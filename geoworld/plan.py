"""Search a dataset's fixed discrete action vocabulary between two observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from geoworld.planning import CategoricalCEMPlanner
from geoworld.runtime import load_predictor, seed_everything, write_json


def observation_clip(path, frames, end_time=None, clip_seconds=1.0):
    path = Path(path)
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        with Image.open(path) as image:
            pixels = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1).float() / 255
        return pixels[:, None].expand(-1, frames, -1, -1)
    from geoworld.data.video import load_video_clip
    if end_time is None:
        import av
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            duration = float(stream.duration * stream.time_base) if stream.duration is not None else float(container.duration or 0) / av.time_base
        if duration <= 0:
            raise ValueError("Video duration is unavailable; specify --start-time or --goal-time")
        end_time = duration
    return load_video_clip(path, end_time=end_time, frames_per_clip=frames, clip_seconds=clip_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", type=Path, help="A cached [T+1,D] .npy trajectory; use first and last states")
    parser.add_argument("--start", type=Path, help="Start image or video")
    parser.add_argument("--goal", type=Path, help="Goal image or video")
    parser.add_argument("--start-time", type=float, help="End of the start clip in seconds")
    parser.add_argument("--goal-time", type=float, help="End of the goal clip in seconds")
    parser.add_argument("--clip-seconds", type=float, default=1.0)
    parser.add_argument("--encoder-checkpoint", help="Override the frozen V-JEPA 2 checkpoint location")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--num-elites", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.features) == bool(args.start or args.goal) or (not args.features and not (args.start and args.goal)):
        parser.error("Supply --features, or both --start and --goal")
    seed_everything(args.seed)
    predictor, checkpoint = load_predictor(args.checkpoint, args.device)
    if args.features:
        features = torch.from_numpy(np.load(args.features, allow_pickle=False)).float()
        if features.ndim != 2 or len(features) < 2 or not torch.isfinite(features).all():
            raise ValueError("--features must contain finite [T+1,D] Euclidean features")
        start, goal = features[[0, -1]].to(args.device).split(1)
    else:
        from geoworld.models import FrozenVJEPA2Encoder
        encoder_config = dict(checkpoint["config"]["model"]["encoder"])
        if args.encoder_checkpoint:
            encoder_config["checkpoint"] = args.encoder_checkpoint
        encoder = FrozenVJEPA2Encoder(**encoder_config).to(args.device).eval()
        with torch.inference_mode():
            start = encoder(observation_clip(args.start, encoder.frames, args.start_time, args.clip_seconds)[None].to(args.device))
            goal = encoder(observation_clip(args.goal, encoder.frames, args.goal_time, args.clip_seconds)[None].to(args.device))
        del encoder
    options = dict(checkpoint["config"].get("planning", {}))
    for key in ("num_samples", "num_elites", "iterations"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    planner = CategoricalCEMPlanner(predictor.num_actions, args.horizon, **options)
    with torch.inference_mode():
        result = planner.plan(predictor, predictor.encode_states(start), predictor.encode_states(goal),
                              generator=torch.Generator(device=args.device).manual_seed(args.seed))
    actions = result["actions"][0].tolist()
    vocabulary = checkpoint.get("vocabulary") or {}
    labels = vocabulary.get("labels", [])
    output = {"action_ids": actions, "action_labels": [labels[i] for i in actions] if labels else [],
              "energy": float(result["energy"][0]), "horizon": args.horizon,
              "num_actions": predictor.num_actions, "action_space": "full_dataset_vocabulary",
              "seed": args.seed, "curvature": float(predictor.curvature.detach())}
    write_json(output, args.output)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
