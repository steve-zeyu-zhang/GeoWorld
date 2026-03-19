#!/usr/bin/env python3
"""Build video-disjoint trajectory manifests and a fixed global action vocabulary."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoworld.data.metadata import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["crosstask", "coin"], required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    split = parser.add_mutually_exclusive_group()
    split.add_argument("--split-file", type=Path, help="JSON mapping train/val/test to exact video ID lists")
    split.add_argument("--holdout-fraction", type=float, help="Explicit nonofficial deterministic video holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-policy", choices=["require", "available", "metadata"], default="require",
                        help="require: fail on missing videos; available: explicit partial subset; metadata: metadata audit only")
    args = parser.parse_args()
    report = prepare_dataset(args.dataset, args.metadata, args.videos, args.output,
        split_file=args.split_file, holdout_fraction=args.holdout_fraction,
        seed=args.seed, video_policy=args.video_policy)
    print(json.dumps({k: v for k, v in report.items() if k not in {"missing_videos", "duplicate_video_files"}}, indent=2))


if __name__ == "__main__":
    main()
