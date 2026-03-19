"""Evaluate discrete CEM planning using start/goal features and held-out labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from geoworld.data import CachedTrajectoryDataset
from geoworld.metrics import planning_metrics
from geoworld.planning import CategoricalCEMPlanner
from geoworld.runtime import load_predictor, seed_everything, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--num-elites", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-action-mask", action="store_true", help="Explicitly restrict to known task labels; not the default benchmark setting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    seed_everything(args.seed)
    predictor, checkpoint = load_predictor(args.checkpoint, args.device)
    dataset = CachedTrajectoryDataset(args.manifest, horizon=args.horizon, num_actions=predictor.num_actions)
    if dataset.vocabulary != checkpoint.get("vocabulary"):
        raise ValueError("Evaluation vocabulary differs from the checkpoint labels")
    options = dict(checkpoint["config"].get("planning", {}))
    for key in ("num_samples", "num_elites", "iterations"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    planner = CategoricalCEMPlanner(predictor.num_actions, args.horizon, **options)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    predictions, targets, rows = [], [], []
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    for index in tqdm(range(count), desc="Planning"):
        item = dataset[index]
        if len(item["actions"]) != args.horizon:
            raise ValueError("Evaluation requires exact-length windows for the requested horizon")
        features = item["features"].to(args.device)
        with torch.inference_mode():
            states = predictor.encode_states(features[[0, -1]])
            allowed = item["task_action_mask"].to(args.device) if args.task_action_mask else None
            result = planner.plan(predictor, states[:1], states[1:], action_mask=allowed, generator=generator)
        pred, target = result["actions"][0].cpu(), item["actions"].cpu()
        predictions.append(pred)
        targets.append(target)
        rows.append({"id": item["id"], "window_index": index, "prediction": pred.tolist(), "target": target.tolist(),
                     "energy": float(result["energy"][0])})
    if not rows:
        raise ValueError("No evaluation windows are available")
    report = {"metrics_percent": planning_metrics(torch.stack(predictions), torch.stack(targets)),
              "examples": len(rows), "horizon": args.horizon, "num_actions": predictor.num_actions,
              "action_space": "known_task_labels" if args.task_action_mask else "full_dataset_vocabulary",
              "planning": options, "seed": args.seed, "predictions": rows}
    write_json(report, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}))


if __name__ == "__main__":
    main()
