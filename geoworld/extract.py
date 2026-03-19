"""Cache frozen V-JEPA 2 observation features for prepared video trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from geoworld.data import VideoTrajectoryDataset
from geoworld.data.video import load_video_clips
from geoworld.runtime import read_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clip-seconds", type=float, default=1.0)
    parser.add_argument("--observation-mode", choices=("video", "image"), default="video")
    parser.add_argument("--batch-size", type=int, default=8, help="Clips encoded together")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-errors", action="store_true", help="Explicitly allow missing/corrupt videos; write an error report")
    parser.add_argument("--limit", type=int, help="Only process this many trajectories")
    args = parser.parse_args()
    if args.batch_size < 1 or args.clip_seconds <= 0 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size, clip duration and limit must be positive")
    config = read_config(args.config)
    # Build only the frozen encoder, not the 300M-parameter predictor.
    from geoworld.models import FrozenVJEPA2Encoder
    encoder = FrozenVJEPA2Encoder(**config["model"]["encoder"]).to(args.device).eval()
    dataset = VideoTrajectoryDataset(args.manifest, frames_per_clip=config["model"]["encoder"].get("frames", 16),
                                    clip_seconds=args.clip_seconds)
    args.output.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output / "features"
    feature_dir.mkdir(exist_ok=True)
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    records, errors = [], []
    for index in tqdm(range(count), desc="Encoding videos"):
        record = dict(dataset.records[index])
        try:
            outputs = []
            video_path = args.manifest.parent / record["video"]
            with torch.inference_mode():
                for offset in range(0, len(record["state_times"]), args.batch_size):
                    times = record["state_times"][offset:offset + args.batch_size]
                    chunk = load_video_clips(video_path, times, 1 if args.observation_mode == "image" else encoder.frames,
                                             args.clip_seconds)
                    if args.observation_mode == "image":
                        chunk = chunk.expand(-1, -1, encoder.frames, -1, -1)
                    outputs.append(encoder(chunk.to(args.device)).float().cpu())
            features = torch.cat(outputs).numpy()
            if features.ndim != 2 or features.shape[0] != len(record["actions"]) + 1 or not np.isfinite(features).all():
                raise ValueError("Encoder produced invalid trajectory features")
            name = hashlib.sha256(str(record["id"]).encode()).hexdigest()[:24] + ".npy"
            path = feature_dir / name
            with path.with_suffix(".npy.tmp").open("wb") as stream:
                np.save(stream, features, allow_pickle=False)
            path.with_suffix(".npy.tmp").replace(path)
            record["features"] = str(path.relative_to(args.output))
            source_video = Path(record["video"])
            record["video"] = str(source_video if source_video.is_absolute() else (args.manifest.parent / source_video).resolve())
            records.append(record)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append({"id": record.get("id"), "error": str(exc)})
            write_json(errors, args.output / "extraction_errors.json")
            if not args.skip_errors:
                raise
    temporary = args.output / "manifest.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row) + "\n" for row in records))
    temporary.replace(args.output / "manifest.jsonl")
    vocabulary = args.manifest.parent / "vocabulary.json"
    if vocabulary.exists():
        shutil.copyfile(vocabulary, args.output / "vocabulary.json")
    write_json({"encoder": config["model"]["encoder"], "clip_seconds": args.clip_seconds,
                "observation_mode": args.observation_mode,
                "input_manifest": str(args.manifest.resolve()), "feature_dim": encoder.feature_dim,
                "processed": count, "written": len(records), "errors": errors}, args.output / "extraction.json")
    print(json.dumps({"written": len(records), "failed": len(errors), "manifest": str(args.output / "manifest.jsonl")}))
    if not records:
        raise RuntimeError("No trajectories were encoded")


if __name__ == "__main__":
    main()
