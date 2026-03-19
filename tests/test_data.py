"""Data contracts tested using temporary metadata, arrays, and encoded video."""

import json
from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from geoworld.data import (
    CachedTrajectoryDataset, VideoTrajectoryDataset, collate_trajectories,
    load_video_clip, load_video_clips, safe_cache_name,
)
from geoworld.data.metadata import (
    apply_split_file, deterministic_holdout, prepare_dataset,
    read_coin, read_crosstask, validate_video_splits, vocabulary_hash,
)


@pytest.fixture
def crosstask_metadata(tmp_path):
    root = tmp_path / "metadata"
    (root / "annotations").mkdir(parents=True)
    (root / "tasks_primary.txt").write_text(
        "10\nFirst task\nhttps://example.org/task1\n2\nopen lid, pour water\n\n"
        "20\nSecond task\nhttps://example.org/task2\n2\npour water, close lid\n"
    )
    (root / "videos.csv").write_text("10,first,https://example.org/first\n20,second,https://example.org/second\n")
    (root / "videos_val.csv").write_text("20,second,https://example.org/second\n")
    # Repeated and nested annotations are intentionally preserved.
    (root / "annotations/10_first.csv").write_text("2,1,3\n1,0,5\n2,4,6\n")
    (root / "annotations/20_second.csv").write_text("1,0,1\n2,1,2\n")
    return root


def test_crosstask_fixed_labels_and_temporal_alignment(crosstask_metadata):
    records, vocabulary = read_crosstask(crosstask_metadata)
    assert vocabulary["labels"] == ["open lid", "pour water", "close lid"]
    assert vocabulary["local_to_global"] == {"10_1": 0, "10_2": 1, "20_1": 1, "20_2": 2}
    assert records[0]["actions"] == [0, 1, 1]
    assert records[0]["state_times"] == [0, 1, 4, 6]
    assert records[0]["overlapping_annotations"]
    assert [(r["split"], r["action_count"]) for r in records] == [("train", 3), ("val", 3)]
    assert records[1]["task_action_ids"] == [1, 2]


def test_coin_keeps_distinct_official_ids_with_equal_names(tmp_path):
    def video(task, subset, ids):
        return {"recipe_type": task, "class": str(task), "duration": 6, "subset": subset,
                "video_url": "https://example.org/video", "annotation": [
                    {"id": str(a), "label": "shared" if a < 3 else "finish", "segment": [i, i + 1]}
                    for i, a in enumerate(ids)]}
    source = tmp_path / "COIN.json"
    source.write_text(json.dumps({"database": {
        "one": video(1, "training", [1, 3]), "two": video(2, "testing", [2]),
    }}))
    records, vocabulary = read_coin(source)
    assert vocabulary["labels"] == ["shared", "shared", "finish"]
    assert vocabulary["source_ids"] == [1, 2, 3]
    assert records[0]["actions"] == [0, 2]
    assert records[1]["actions"] == [1]
    assert [r["split"] for r in records] == ["train", "test"]
    assert vocabulary["tasks"]["2"]["action_ids"] == [1]


def test_video_split_integrity_and_holdout(crosstask_metadata, tmp_path):
    records, _ = read_crosstask(crosstask_metadata)
    assignment = tmp_path / "splits.json"
    assignment.write_text(json.dumps({"train": ["first"], "test": ["first"]}))
    with pytest.raises(ValueError, match="occurs more than once"):
        apply_split_file(records, assignment)
    assignment.write_text(json.dumps({"test": ["second"]}))
    selected = apply_split_file(records, assignment)
    assert len(selected) == 1 and selected[0]["split"] == "test"
    duplicated = records + [{**records[0], "task": "other", "id": "extra"}]
    holdout = deterministic_holdout(duplicated, 0.5, seed=71)
    assert holdout[0]["split"] == holdout[-1]["split"]
    assert holdout == deterministic_holdout(duplicated, 0.5, seed=71)
    validate_video_splits(holdout)
    with pytest.raises(ValueError, match="leakage"):
        validate_video_splits([records[0], {**records[0], "split": "test"}])


def test_preparation_reports_missing_original_videos(crosstask_metadata, tmp_path):
    videos, output = tmp_path / "videos", tmp_path / "prepared"
    with pytest.raises(FileNotFoundError, match="Missing 2 original videos"):
        prepare_dataset("crosstask", crosstask_metadata, videos, output)
    report = json.loads((output / "preparation_report.json").read_text())
    assert report["missing_videos"] == ["first", "second"]
    assert not (output / "train.jsonl").exists()
    prepare_dataset("crosstask", crosstask_metadata, videos, output, video_policy="metadata")
    row = json.loads((output / "train.jsonl").read_text())
    assert row["video_available"] is False
    with pytest.raises(FileNotFoundError, match="Original video is missing"):
        VideoTrajectoryDataset(output / "train.jsonl")[0]
    videos.mkdir()
    (videos / "first.mp4").touch()
    report = prepare_dataset("crosstask", crosstask_metadata, videos, output, video_policy="available")
    assert report["prepared_trajectories"] == 1
    # Partial available-video selection must keep the complete fixed vocabulary.
    assert report["action_count"] == 3
    assert json.loads((output / "vocabulary.json").read_text())["labels"][-1] == "close lid"


def _cached_manifest(tmp_path):
    vocabulary = {"action_count": 4, "labels": ["zero", "one", "two", "three"]}
    (tmp_path / "vocabulary.json").write_text(json.dumps(vocabulary, indent=4))
    rows = []
    for i, actions in enumerate(([0, 1, 0], [2])):
        features = np.arange((len(actions) + 1) * 5, dtype=np.float32).reshape(-1, 5) + i * 100
        np.save(tmp_path / f"{i}.npy", features)
        rows.append({"id": f"trajectory_{i}", "video_id": f"video_{i}", "task": str(i),
                     "split": "train", "actions": list(actions), "features": f"{i}.npy",
                     "action_count": 4, "task_action_ids": [0, 1] if i == 0 else [2, 3],
                     "vocabulary_sha256": vocabulary_hash(vocabulary),
                     "state_times": list(range(len(actions) + 1))})
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, rows


def test_cached_windows_global_mask_and_padding(tmp_path):
    path, rows = _cached_manifest(tmp_path)
    dataset = CachedTrajectoryDataset(path)
    assert dataset.feature_dim == 5 and dataset.action_count == 4
    batch = collate_trajectories([dataset[0], dataset[1]])
    assert batch["features"].shape == (2, 4, 5)
    assert batch["actions"].tolist() == [[0, 1, 0], [2, -1, -1]]
    assert batch["state_mask"].tolist() == [[True] * 4, [True, True, False, False]]
    assert batch["task_action_mask"].tolist() == [[True, True, False, False], [False, False, True, True]]
    windows = CachedTrajectoryDataset(path, horizon=2)
    assert windows.windows == [(0, 0), (0, 1)]
    assert windows[1]["actions"].tolist() == [1, 0]
    torch.testing.assert_close(windows[1]["features"], dataset[0]["features"][1:])
    assert safe_cache_name({"id": "../../escape"}).endswith(".npy")
    assert "/" not in safe_cache_name({"id": "../../escape"})


@pytest.mark.parametrize("change,error", [
    ({"actions": [4]}, "invalid global action"),
    ({"vocabulary_sha256": "incorrect"}, "hashes disagree"),
    ({"action_count": 2}, "single fixed global"),
    ({"task_action_ids": [0]}, "invalid task action"),
    ({"state_times": [0, 2, 1, 3]}, "invalid state timestamps"),
])
def test_cached_metadata_rejects_misalignment(tmp_path, change, error):
    path, rows = _cached_manifest(tmp_path)
    rows[0].update(change)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=error):
        CachedTrajectoryDataset(path)


def test_cached_nonfinite_and_inconsistent_features_fail(tmp_path):
    path, _ = _cached_manifest(tmp_path)
    np.save(tmp_path / "0.npy", np.full((4, 5), np.nan))
    with pytest.raises(ValueError, match="nonfinite"):
        CachedTrajectoryDataset(path)
    np.save(tmp_path / "0.npy", np.zeros((4, 5)))
    np.save(tmp_path / "1.npy", np.zeros((2, 6)))
    dataset = CachedTrajectoryDataset(path)
    with pytest.raises(ValueError, match="feature dimension differs"):
        dataset[1]


@pytest.fixture
def timestamped_video(tmp_path):
    path = tmp_path / "timed.mkv"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 24, 16, "bgr0"
        for index in range(21):
            pixels = np.full((16, 24, 3), index * 10, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 10)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def test_video_decoder_uses_causal_pts_and_start_padding(timestamped_video):
    clips = load_video_clips(timestamped_video, [1.05, 0.1], frames_per_clip=3, clip_seconds=0.2)
    assert clips.shape == (2, 3, 3, 16, 24)
    assert clips.dtype == torch.float32
    observed = (clips[:, 0, :, 0, 0] * 255).round().int().tolist()
    assert observed == [[80, 90, 100], [0, 0, 10]]
    image = load_video_clip(timestamped_video, 1.05, frames_per_clip=1)
    assert image.shape == (3, 1, 16, 24)
    assert (image * 255).round().eq(100).all()
    assert clips.min() >= 0 and clips.max() <= 1


def test_video_decoder_seek_resize_and_truncation(timestamped_video):
    clip = load_video_clips(timestamped_video, [2.1], frames_per_clip=2, clip_seconds=0.3, image_size=12)
    assert clip.shape == (1, 3, 2, 8, 12)
    assert (clip[0, 0, :, 0, 0] * 255).round().int().tolist() == [180, 200]
    with pytest.raises(ValueError, match="beyond decoded video"):
        load_video_clip(timestamped_video, 3.0)
    with pytest.raises(ValueError, match="nonnegative"):
        load_video_clip(timestamped_video, -0.1)
