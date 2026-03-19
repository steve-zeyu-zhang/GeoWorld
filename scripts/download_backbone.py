#!/usr/bin/env python3
"""Download an original V-JEPA 2 encoder checkpoint from Meta."""
import argparse
import os
from pathlib import Path
import requests
from tqdm import tqdm

URLS = {
    "vit_large": "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt",
    "vit_huge": "https://dl.fbaipublicfiles.com/vjepa2/vith.pt",
    "vit_giant": "https://dl.fbaipublicfiles.com/vjepa2/vitg.pt",
    "vit_giant_384": "https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=URLS, default="vit_large")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    with requests.get(URLS[args.model], stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with partial.open("wb") as out, tqdm(total=int(response.headers.get("content-length", 0)), unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if chunk:
                    out.write(chunk)
                    bar.update(len(chunk))
        os.replace(partial, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
