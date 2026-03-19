#!/usr/bin/env python3
"""Download official CrossTask/COIN annotations and video URLs."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoworld.data.download import download_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["crosstask", "coin", "all"], default="all")
    parser.add_argument("--output", type=Path, default=Path("data/metadata"))
    args = parser.parse_args()
    print(json.dumps(download_metadata(args.output, args.dataset), indent=2))


if __name__ == "__main__":
    main()
