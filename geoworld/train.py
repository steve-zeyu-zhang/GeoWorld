"""Supervised post-training and geometric reinforcement learning on cached features."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from geoworld.data import CachedTrajectoryDataset, collate_trajectories
from geoworld.losses import grl_loss, sft_loss
from geoworld.models import HyperbolicPredictor
from geoworld.runtime import atomic_save, load_checkpoint, read_config, seed_everything


class TrainingObjective(nn.Module):
    def __init__(self, predictor, stage, config):
        super().__init__()
        self.predictor, self.stage, self.config = predictor, stage, config

    def forward(self, features, actions, mask):
        if self.stage == "sft":
            return sft_loss(self.predictor, features, actions, mask,
                            lambda_tf=self.config.get("lambda_tf", 0.5))
        return grl_loss(self.predictor, features, actions, mask,
                        gamma=self.config.get("gamma", 0.99), beta=self.config.get("beta", 0.1))


def learning_rate(step, config):
    warmup, hold, decay = (int(config[k]) for k in ("warmup_steps", "hold_steps", "decay_steps"))
    start, peak = float(config["start_lr"]), float(config["lr"])
    if warmup and step < warmup:
        return start + (peak - start) * step / warmup
    if step < warmup + hold:
        return peak
    return peak * max(0.0, 1.0 - (step - warmup - hold) / max(decay, 1))


class CyclingBatches:
    def __init__(self, loader, sampler, seed, consumed=0):
        self.loader, self.sampler, self.seed = loader, sampler, seed
        if not len(loader):
            raise ValueError("The training dataset is empty")
        self.consumed, self.iterator = consumed, None
        self._start_epoch(consumed // len(loader))
        for _ in range(consumed % len(loader)):
            next(self.iterator)

    def _start_epoch(self, epoch):
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)
        else:
            self.sampler.generator.manual_seed(self.seed + epoch)
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self._start_epoch(self.consumed // len(self.loader))
            batch = next(self.iterator)
        self.consumed += 1
        return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("sft", "grl"), required=True)
    parser.add_argument("--manifest", required=True, help="JSONL with cached feature paths")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initialize", help="SFT checkpoint for starting GRL")
    parser.add_argument("--resume", help="Continue the same stage, optimizer, and schedule")
    parser.add_argument("--max-steps", type=int, help="Stop at this total optimizer-step count")
    parser.add_argument("--batch-size", type=int, help="Global batch size")
    parser.add_argument("--micro-batch-size", type=int, help="Maximum batch per GPU")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.initialize and args.resume:
        parser.error("Use either --initialize or --resume")
    if args.stage == "grl" and not (args.initialize or args.resume):
        parser.error("GRL requires --initialize with an SFT checkpoint, or --resume")
    config = read_config(args.config)
    rank, world, local_rank = (int(os.getenv(k, d)) for k, d in
                               (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        seed = int(config.get("seed", 42))
        seed_everything(seed)
        stage_config = config["training"][args.stage]
        batch_size = args.batch_size or int(stage_config["batch_size"])
        micro_size = args.micro_batch_size or int(config["training"].get("micro_batch_size", 8))
        if batch_size < world or batch_size % world:
            raise ValueError("Global batch size must be a positive multiple of WORLD_SIZE")
        micro_size = min(micro_size, batch_size // world)
        if micro_size < 1 or batch_size % (micro_size * world):
            raise ValueError("Global batch size must be divisible by micro-batch-size * WORLD_SIZE")
        accumulation = batch_size // (micro_size * world)
        dataset = CachedTrajectoryDataset(args.manifest, horizon=config.get("data", {}).get("horizon", 4),
                                          num_actions=config["model"]["predictor"]["num_actions"])
        data_signature = {"manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                          "horizon": config.get("data", {}).get("horizon", 4), "seed": seed}
        predictor = HyperbolicPredictor(**config["model"]["predictor"]).to(device)
        objective = TrainingObjective(predictor, args.stage, stage_config).to(device)
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=float(stage_config["lr"]),
                                      weight_decay=float(config["training"].get("weight_decay", 0.04)))
        step, consumed = 0, 0
        restored = None
        if args.resume or args.initialize:
            restored = load_checkpoint(args.resume or args.initialize)
            if restored["config"]["model"]["predictor"] != config["model"]["predictor"]:
                raise ValueError("Checkpoint predictor configuration differs from this config")
            predictor.load_state_dict(restored["predictor"], strict=True)
            if restored.get("vocabulary") != dataset.vocabulary:
                raise ValueError("Checkpoint and manifest action vocabularies differ")
            if args.resume:
                if restored["stage"] != args.stage:
                    raise ValueError("--resume cannot change training stages; use --initialize")
                if restored["config"]["training"][args.stage] != stage_config:
                    raise ValueError("Resume requires the original optimizer schedule")
                if restored.get("data_signature") != data_signature:
                    raise ValueError("Resume requires the original manifest, window horizon, and random seed")
                if (restored["world_size"], restored["global_batch_size"], restored["micro_batch_size"]) != (world, batch_size, micro_size):
                    raise ValueError("Exact resume requires the original world size and batch sizes")
                optimizer.load_state_dict(restored["optimizer"])
                step, consumed = restored["step"], restored["batches_seen"]
        sampler = (DistributedSampler(dataset, shuffle=True, seed=seed) if world > 1 else
                   RandomSampler(dataset, generator=torch.Generator().manual_seed(seed)))
        loader = DataLoader(dataset, batch_size=micro_size, sampler=sampler,
                            collate_fn=collate_trajectories, drop_last=True,
                            num_workers=int(config["data"].get("workers", 0)),
                            pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed + rank))
        batches = CyclingBatches(loader, sampler, seed, consumed)
        if args.resume and "rng_state" in restored:
            torch.set_rng_state(restored["rng_state"][rank]["torch"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(restored["rng_state"][rank]["cuda"], device)
        if world > 1:
            objective = DistributedDataParallel(objective, device_ids=[local_rank] if device.type == "cuda" else None)
        objective.train()
        seed_everything(seed + rank) if not args.resume else None
        total_steps = sum(int(stage_config[k]) for k in ("warmup_steps", "hold_steps", "decay_steps"))
        stop = min(args.max_steps, total_steps) if args.max_steps is not None else total_steps
        if stop < 1:
            raise ValueError("--max-steps must be positive")
        args.output.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        save_every = int(config["training"].get("save_every", 1000))
        use_amp = device.type == "cuda" and config["training"].get("amp", True)
        while step < stop:
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(step, stage_config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            totals = {}
            for micro_step in range(accumulation):
                batch = next(batches)
                features = batch["features"].to(device, non_blocking=True)
                actions = batch["actions"].to(device, non_blocking=True)
                mask = batch["action_mask"].to(device, non_blocking=True)
                if features.shape[-1] != config["model"]["predictor"]["feature_dim"]:
                    raise ValueError("Cached feature width differs from the predictor feature_dim")
                context = (objective.no_sync() if world > 1 and micro_step < accumulation - 1 else contextlib.nullcontext())
                with context, torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    terms = objective(features, actions, mask)
                    loss = terms["loss"] / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {args.stage} loss at step {step}")
                loss.backward()
                for key, value in terms.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        totals[key] = totals.get(key, 0.0) + value.detach().float() / accumulation
            norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), float(config["training"].get("grad_clip", 1.0)), error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if world > 1:
                for value in totals.values():
                    dist.all_reduce(value)
                    value.div_(world)
            if rank == 0:
                record = {"step": step, "stage": args.stage, "lr": lr, "grad_norm": float(norm),
                          "curvature": float(predictor.curvature.detach()), **{k: float(v) for k, v in totals.items()}}
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                if step == 1 or step % int(config["training"].get("log_every", 10)) == 0 or step == stop:
                    print(json.dumps(record), flush=True)
            if step % save_every == 0 or step == stop:
                rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
                rng_states = [None] * world
                if world > 1:
                    dist.all_gather_object(rng_states, rng)
                else:
                    rng_states[0] = rng
                if rank == 0:
                    atomic_save({"format_version": 1, "config": config, "stage": args.stage,
                                 "step": step, "batches_seen": batches.consumed, "world_size": world,
                                 "global_batch_size": batch_size, "micro_batch_size": micro_size,
                                 "vocabulary": dataset.vocabulary,
                                 "data_signature": data_signature,
                                 "predictor": predictor.state_dict(), "optimizer": optimizer.state_dict(),
                                 "rng_state": rng_states}, args.output / "last.pt")
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
