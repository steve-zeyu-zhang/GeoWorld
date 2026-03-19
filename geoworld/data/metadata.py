"""Build fixed action vocabularies and video-disjoint trajectory manifests."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def vocabulary_hash(vocabulary):
    return hashlib.sha256(canonical_json(vocabulary).encode()).hexdigest()


def _identifier(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Unsafe dataset identifier: {value!r}")
    return value


def _segments(annotations):
    """Preserve repeated/overlapping actions in chronological start order."""
    result = []
    for action, start, end in annotations:
        start, end = float(start), float(end)
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end <= start:
            raise ValueError(f"Invalid annotation interval [{start}, {end}]")
        result.append((int(action), start, end))
    return sorted(result, key=lambda item: (item[1], item[2]))


def _record(dataset, task, video_id, annotations, url, task_actions, action_count, source, split):
    annotations = _segments(annotations)
    if not annotations:
        raise ValueError(f"{dataset}/{video_id}: empty annotations")
    segments = [[start, end] for _, start, end in annotations]
    # Each state is immediately before its action, plus a final state after all
    # annotated actions. This also preserves nested CrossTask annotations.
    state_times = [start for _, start, _ in annotations] + [max(end for _, _, end in annotations)]
    return {"id": f"{dataset}_{_identifier(task)}_{_identifier(video_id)}", "dataset": dataset,
            "task": str(task), "video_id": video_id, "video_url": url, "split": split,
            "actions": [action for action, _, _ in annotations], "segments": segments,
            "state_times": state_times, "task_action_ids": list(task_actions),
            "action_count": action_count, "source_annotation": str(Path(source).resolve()),
            "overlapping_annotations": any(segments[i][1] > segments[i + 1][0] for i in range(len(segments) - 1))}


def read_task_cards(path):
    lines = [line.strip() for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]
    if len(lines) % 5:
        raise ValueError("CrossTask task cards must contain five nonempty lines per task")
    cards = []
    for offset in range(0, len(lines), 5):
        task, title, url, count, labels = lines[offset:offset + 5]
        labels = [label.strip() for label in labels.split(",")]
        if len(labels) != int(count) or not all(labels):
            raise ValueError(f"CrossTask task {task}: inconsistent step labels")
        cards.append({"id": _identifier(task), "title": title, "url": url, "labels": labels})
    return cards


def read_crosstask(metadata):
    """Read the 18 annotated primary tasks, not the unannotated related tasks.

    Enumerating first-seen exact labels in tasks_primary.txt reproduces the
    official PDPP actions_one_hot.npy mapping (133 local steps ->105 actions).
    """
    root = Path(metadata)
    if not (root / "tasks_primary.txt").is_file() and (root / "crosstask_release").is_dir():
        root = root / "crosstask_release"
    cards = read_task_cards(root / "tasks_primary.txt")
    labels, label_to_id, local_to_global, tasks = [], {}, {}, {}
    for card in cards:
        ids = []
        for local, label in enumerate(card["labels"], 1):
            if label not in label_to_id:
                label_to_id[label] = len(labels)
                labels.append(label)
            value = label_to_id[label]
            local_to_global[f"{card['id']}_{local}"] = value
            ids.append(value)
        tasks[card["id"]] = {"label": card["title"], "action_ids": sorted(set(ids))}
    vocabulary = {"schema_version": 1, "dataset": "crosstask", "action_count": len(labels),
                  "labels": labels, "local_to_global": local_to_global, "tasks": tasks,
                  "mapping_rule": "first-occurrence exact labels in official tasks_primary.txt; PDPP-compatible",
                  "source_sha256": {"tasks_primary.txt": file_sha256(root / "tasks_primary.txt")}}
    urls = {}
    with (root / "videos.csv").open(newline="", encoding="utf8") as stream:
        for task, video_id, url in csv.reader(stream):
            urls[(task, video_id)] = url
    val_path = root / "videos_val.csv"
    if not val_path.is_file():
        raise FileNotFoundError("Official CrossTask videos_val.csv is required; download the complete release")
    with val_path.open(newline="", encoding="utf8") as stream:
        validation_ids = {row[1] for row in csv.reader(stream)}
    records = []
    for annotation in sorted((root / "annotations").glob("*.csv")):
        task, video_id = annotation.stem.split("_", 1)
        if task not in tasks:
            continue
        if (task, video_id) not in urls:
            raise ValueError(f"Annotation {annotation.name} is missing from videos.csv")
        with annotation.open(newline="", encoding="utf8") as stream:
            annotations = [(local_to_global[f"{task}_{int(local)}"], start, end)
                           for local, start, end in csv.reader(stream)]
        record = _record("crosstask", task, video_id, annotations, urls[(task, video_id)],
                         tasks[task]["action_ids"], len(labels), annotation,
                         "val" if video_id in validation_ids else "train")
        record["split_protocol"] = "CrossTask official localization train/validation; not a released GeoWorld planning split"
        records.append(record)
    return records, vocabulary


def read_coin(metadata):
    source = Path(metadata)
    if source.is_dir():
        source = source / "COIN.json"
    database = json.loads(source.read_text(encoding="utf8"))["database"]
    label_by_id, tasks = {}, {}
    for video in database.values():
        task = str(video["recipe_type"])
        info = tasks.setdefault(task, {"label": video["class"], "action_ids": set()})
        for annotation in video["annotation"]:
            source_id = int(annotation["id"])
            if source_id < 1:
                raise ValueError("COIN official action IDs are positive and one-based")
            label = annotation["label"]
            if source_id in label_by_id and label_by_id[source_id] != label:
                raise ValueError(f"COIN action {source_id} has inconsistent names")
            label_by_id[source_id] = label
            info["action_ids"].add(source_id - 1)
    if not label_by_id or sorted(label_by_id) != list(range(1, max(label_by_id) + 1)):
        raise ValueError("COIN metadata must contain the complete contiguous official action vocabulary")
    for info in tasks.values():
        info["action_ids"] = sorted(info["action_ids"])
    labels = [label_by_id[i] for i in range(1, len(label_by_id) + 1)]
    vocabulary = {"schema_version": 1, "dataset": "coin", "action_count": len(labels), "labels": labels,
                  "source_ids": list(range(1, len(labels) + 1)), "tasks": tasks,
                  "mapping_rule": "official COIN annotation id minus one; never merge labels from different IDs",
                  "source_sha256": {"COIN.json": file_sha256(source)}}
    split_map = {"training": "train", "testing": "test", "validation": "val"}
    records = []
    for video_id, video in sorted(database.items()):
        task = str(video["recipe_type"])
        if video["subset"] not in split_map:
            raise ValueError(f"Unknown official COIN subset {video['subset']!r}")
        annotations = [(int(a["id"]) - 1, *a["segment"]) for a in video["annotation"]]
        record = _record("coin", task, video_id, annotations, video["video_url"], tasks[task]["action_ids"],
                         len(labels), source, split_map[video["subset"]])
        record["duration"] = float(video["duration"])
        record["split_protocol"] = "COIN official annotation subset; not a verified GeoWorld planning split"
        records.append(record)
    return records, vocabulary


def apply_split_file(records, split_file):
    """Use explicitly supplied planning IDs, keeping all clips of a video together."""
    split_data = json.loads(Path(split_file).read_text())
    if not isinstance(split_data, dict) or not set(split_data).issubset({"train", "val", "test"}):
        raise ValueError("Split JSON must map train/val/test to lists of video IDs")
    assignments = {}
    for split, ids in split_data.items():
        if not isinstance(ids, list):
            raise ValueError("Each split must contain a JSON list of video IDs")
        for video_id in ids:
            if video_id in assignments:
                raise ValueError(f"Video {video_id!r} occurs more than once/across splits")
            assignments[video_id] = split
    known = {r["video_id"] for r in records}
    if set(assignments) - known:
        raise ValueError(f"Unknown video IDs in provided split: {sorted(set(assignments) - known)[:5]}")
    output = []
    for row in records:
        if row["video_id"] in assignments:
            output.append({**row, "split": assignments[row["video_id"]],
                           "split_protocol": f"provided planning split:{file_sha256(split_file)}"})
    return output


def deterministic_holdout(records, fraction, seed=42):
    """Explicit development fallback, never labeled as an official benchmark split."""
    if not 0 < fraction < 1:
        raise ValueError("Holdout fraction must be between zero and one")
    output = []
    for row in records:
        value = int(hashlib.sha256(f"{seed}:{row['video_id']}".encode()).hexdigest()[:16], 16) / 2**64
        output.append({**row, "split": "test" if value < fraction else "train",
                       "split_protocol": f"nonofficial deterministic video holdout fraction={fraction} seed={seed}"})
    return output


def validate_video_splits(records):
    seen = {}
    for row in records:
        video_id = row["video_id"]
        if video_id in seen and seen[video_id] != row["split"]:
            raise ValueError(f"Video leakage across splits: {video_id}")
        seen[video_id] = row["split"]


def prepare_dataset(dataset, metadata, videos, output, *, split_file=None,
                    holdout_fraction=None, seed=42, video_policy="require"):
    if dataset not in {"crosstask", "coin"}:
        raise ValueError("dataset must be crosstask or coin")
    if video_policy not in {"require", "available", "metadata"}:
        raise ValueError("video_policy must be require, available, or metadata")
    if split_file is not None and holdout_fraction is not None:
        raise ValueError("Choose either provided splits or a holdout, not both")
    records, vocabulary = (read_crosstask(metadata) if dataset == "crosstask" else read_coin(metadata))
    source_count = len(records)
    if split_file is not None:
        records = apply_split_file(records, split_file)
    elif holdout_fraction is not None:
        records = deterministic_holdout(records, holdout_fraction, seed)
    validate_video_splits(records)
    video_root, output = Path(videos).resolve(), Path(output)
    video_index = defaultdict(list)
    if video_root.is_dir():
        for path in sorted(video_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                video_index[path.stem].append(path)
    missing, duplicate = [], []
    result = []
    digest = vocabulary_hash(vocabulary)
    for row in records:
        paths = video_index.get(row["video_id"], [])
        if len(paths) > 1:
            duplicate.append({"video_id": row["video_id"], "paths": [str(p) for p in paths]})
            continue
        if not paths:
            missing.append(row["video_id"])
            if video_policy == "available":
                continue
        video = paths[0] if paths else video_root / (row["video_id"] + ".mp4")
        result.append({**row, "video": str(video), "video_available": bool(paths),
                       "vocabulary_sha256": digest})
    report = {"dataset": dataset, "source_trajectories": source_count,
              "selected_trajectories": len(records), "prepared_trajectories": len(result),
              "action_count": len(vocabulary["labels"]), "task_count": len(vocabulary["tasks"]),
              "split_counts": dict(Counter(r["split"] for r in result)), "video_policy": video_policy,
              "missing_videos": sorted(set(missing)), "duplicate_video_files": duplicate,
              "overlapping_annotation_videos": sum(r["overlapping_annotations"] for r in records),
              "split_protocols": sorted({r["split_protocol"] for r in records}),
              "vocabulary_sha256": digest, "paper_benchmark_split_verified": False}
    output.mkdir(parents=True, exist_ok=True)
    (output / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2, ensure_ascii=False) + "\n")
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if duplicate:
        raise ValueError(f"Ambiguous video files; see {output / 'preparation_report.json'}")
    if missing and video_policy == "require":
        raise FileNotFoundError(f"Missing {len(set(missing))} original videos; see {output / 'preparation_report.json'}")
    if not result:
        raise ValueError("No trajectories remain after video/split filtering")
    for split in ("train", "val", "test"):
        rows = [row for row in result if row["split"] == split]
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return report
