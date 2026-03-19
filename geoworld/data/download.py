"""Download only original public dataset annotations, never replacement videos."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile


CROSSTASK_URL = "https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip"
CROSSTASK_SHA256 = "25ab49242dcd29b411156a6cffa5e5b22760ce2305fcc4b5747508c4ec081300"
COIN_COMMIT = "67e31260e9c25951e27cda8d4336a23eacf62a80"
COIN_URL = f"https://raw.githubusercontent.com/coin-dataset/annotations/{COIN_COMMIT}/COIN.json"
COIN_SHA256 = "c4b2d299328d007be422f58ce65ddf9fdbd9d71e56342c64a6d239b35c5eb111"


def _download(url, destination, expected_sha256):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        data = destination.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(64 * 1024 * 1024 + 1)
        if len(data) > 64 * 1024 * 1024:
            raise ValueError("Annotation download unexpectedly exceeds 64 MiB")
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Metadata SHA256 mismatch for {destination}; source changed or cache is corrupt")
    destination.write_bytes(data)
    return data, {"url": url, "sha256": digest, "bytes": len(data), "file": destination.name}


def download_metadata(output="data/metadata", dataset="all"):
    if dataset not in {"crosstask", "coin", "all"}:
        raise ValueError("dataset must be crosstask, coin, or all")
    output = Path(output)
    records = {}
    if dataset in {"crosstask", "all"}:
        root = output / "crosstask"
        raw, info = _download(CROSSTASK_URL, root / "crosstask_release.zip", CROSSTASK_SHA256)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Unsafe file in CrossTask archive")
                if member.is_dir():
                    continue
                if member.file_size > 16 * 1024 * 1024:
                    raise ValueError("Unexpectedly large annotation entry")
                destination = root / Path(*path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
        info["source"] = "https://github.com/DmZhukov/CrossTask"
        (root / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
        records["crosstask"] = str(root / "crosstask_release")
    if dataset in {"coin", "all"}:
        root = output / "coin"
        _, info = _download(COIN_URL, root / "COIN.json", COIN_SHA256)
        info["commit"] = COIN_COMMIT
        info["source"] = "https://github.com/coin-dataset/annotations"
        (root / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
        records["coin"] = str(root / "COIN.json")
    return records
