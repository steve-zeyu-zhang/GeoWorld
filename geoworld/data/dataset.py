"""Fixed-vocabulary trajectory loaders for real RGB and frozen encoder features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .metadata import vocabulary_hash, validate_video_splits
from .video import load_video_clips


def safe_cache_name(record):
    """Stable collision-resistant filename; raw IDs never become paths."""
    return hashlib.sha256(str(record["id"]).encode()).hexdigest()[:24] + ".npy"


class _TrajectoryDataset(Dataset):
    def __init__(self, manifest, horizon=None, num_actions=None, window_stride=1):
        self.manifest = Path(manifest)
        self.records = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f"Empty trajectory manifest: {self.manifest}")
        if horizon is not None and horizon < 1:
            raise ValueError("horizon must be positive")
        if window_stride < 1:
            raise ValueError("window_stride must be positive")
        vocab_path = self.manifest.parent / "vocabulary.json"
        self.vocabulary = json.loads(vocab_path.read_text()) if vocab_path.is_file() else None
        self.vocabulary_sha256 = vocabulary_hash(self.vocabulary) if self.vocabulary is not None else None
        counts = {int(row["action_count"]) for row in self.records if "action_count" in row}
        if self.vocabulary is not None:
            counts.add(int(self.vocabulary["action_count"]))
            if len(self.vocabulary["labels"]) != self.vocabulary["action_count"]:
                raise ValueError("Vocabulary labels and action_count disagree")
        if num_actions is not None:
            counts.add(int(num_actions))
        if len(counts) != 1 or next(iter(counts)) < 1:
            raise ValueError("A single fixed global action_count is required via vocabulary.json, metadata, or num_actions")
        self.action_count = self.num_actions = counts.pop()
        self.horizon = horizon
        self.windows = []
        ids = set()
        validate_video_splits(self.records)
        for i, row in enumerate(self.records):
            if row["id"] in ids:
                raise ValueError(f"Duplicate trajectory ID: {row['id']}")
            ids.add(row["id"])
            if self.vocabulary_sha256 is not None and row.get("vocabulary_sha256", self.vocabulary_sha256) != self.vocabulary_sha256:
                raise ValueError("Manifest and fixed action vocabulary hashes disagree")
            actions = row["actions"]
            if not actions or any(type(a) is not int or not 0 <= a < self.action_count for a in actions):
                raise ValueError(f"{row['id']}: invalid global action IDs")
            if "task_action_ids" in row:
                if not set(actions).issubset(row["task_action_ids"]) or any(type(a) is not int or not 0 <= a < self.action_count for a in row["task_action_ids"]):
                    raise ValueError(f"{row['id']}: invalid task action membership")
            if "state_times" in row:
                times = np.asarray(row["state_times"], dtype=float)
                if times.shape != (len(actions) + 1,) or not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) < 0):
                    raise ValueError(f"{row['id']}: invalid state timestamps")
            if horizon is None:
                self.windows.append((i, 0))
            else:
                self.windows.extend((i, start) for start in range(0, len(actions) - horizon + 1, window_stride))
        if not self.windows:
            raise ValueError(f"No trajectories contain the requested horizon {horizon}")

    def __len__(self):
        return len(self.windows)

    def _metadata(self, index):
        record_index, start = self.windows[index]
        row = self.records[record_index]
        stop = start + self.horizon if self.horizon is not None else len(row["actions"])
        task_mask = torch.zeros(self.action_count, dtype=torch.bool)
        task_mask[row.get("task_action_ids", list(range(self.action_count)))] = True
        sample = {"id": row["id"], "task": row["task"], "video_id": row["video_id"],
                  "actions": torch.tensor(row["actions"][start:stop], dtype=torch.long),
                  "mask": torch.ones(stop - start, dtype=torch.bool),
                  "state_mask": torch.ones(stop - start + 1, dtype=torch.bool),
                  "task_action_mask": task_mask, "start_step": start, "record_index": record_index}
        sample["action_mask"] = sample["mask"]
        return row, start, stop, sample


class CachedTrajectoryDataset(_TrajectoryDataset):
    def __init__(self, manifest, horizon=None, num_actions=None, window_stride=1):
        super().__init__(manifest, horizon, num_actions, window_stride)
        self.feature_dim = None
        self._features(self.records[self.windows[0][0]])

    def _features(self, row):
        if not row.get("features"):
            raise ValueError("Manifest does not contain cached encoder features; run feature extraction first")
        path = self.manifest.parent / row["features"]
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                features = np.array(data["features"], dtype=np.float32)
        else:
            features = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != len(row["actions"]) + 1 or features.shape[1] < 1:
            raise ValueError(f"{path}: cached features must have shape [number_of_actions+1,D]")
        if not np.isfinite(features).all():
            raise ValueError(f"{path}: nonfinite encoder features")
        if self.feature_dim is None:
            self.feature_dim = features.shape[1]
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"{path}: feature dimension differs from other cached trajectories")
        return features

    def __getitem__(self, index):
        row, start, stop, sample = self._metadata(index)
        sample["features"] = torch.from_numpy(self._features(row)[start:stop + 1].copy())
        return sample


class VideoTrajectoryDataset(_TrajectoryDataset):
    def __init__(self, manifest, frames_per_clip=16, clip_seconds=1.0, image_size=None,
                 horizon=None, num_actions=None):
        super().__init__(manifest, horizon, num_actions)
        self.frames_per_clip = frames_per_clip
        self.clip_seconds = clip_seconds
        self.image_size = image_size

    def __getitem__(self, index):
        row, start, stop, sample = self._metadata(index)
        video = self.manifest.parent / row["video"]
        sample["clips"] = load_video_clips(video, row["state_times"][start:stop + 1],
            self.frames_per_clip, self.clip_seconds, self.image_size)
        return sample


def collate_trajectories(items):
    if not items:
        raise ValueError("Cannot collate an empty trajectory batch")
    feature_dim = items[0]["features"].shape[-1]
    action_count = len(items[0]["task_action_mask"])
    steps = max(len(item["actions"]) for item in items)
    features = torch.zeros(len(items), steps + 1, feature_dim)
    actions = torch.full((len(items), steps), -1, dtype=torch.long)
    mask = torch.zeros(len(items), steps, dtype=torch.bool)
    state_mask = torch.zeros(len(items), steps + 1, dtype=torch.bool)
    task_mask = torch.zeros(len(items), action_count, dtype=torch.bool)
    for i, item in enumerate(items):
        n = len(item["actions"])
        if item["features"].shape != (n + 1, feature_dim) or len(item["task_action_mask"]) != action_count:
            raise ValueError("Batch feature dimensions or fixed action vocabularies disagree")
        features[i, :n + 1] = item["features"]
        actions[i, :n] = item["actions"]
        mask[i, :n] = True
        state_mask[i, :n + 1] = True
        task_mask[i] = item["task_action_mask"]
    return {"features": features, "actions": actions, "mask": mask, "action_mask": mask,
            "state_mask": state_mask, "task_action_mask": task_mask,
            "id": [item["id"] for item in items], "task": [item["task"] for item in items],
            "video_id": [item["video_id"] for item in items]}
