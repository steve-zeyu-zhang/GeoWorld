"""Exercise video decoding, feature caching, training/resume and discrete planning."""
import json
import os
from pathlib import Path
import subprocess
import sys

import av
import numpy as np
import torch
import yaml

from geoworld.data.metadata import vocabulary_hash


ROOT = Path(__file__).resolve().parents[1]


def run(module, *arguments):
    result = subprocess.run([sys.executable, "-m", module, *map(str, arguments)], cwd=ROOT,
                            env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_video_to_both_training_stages_resume_and_planning(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "fixture.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 48, 32, "yuv420p"
        for index in range(16):
            pixels = np.zeros((32, 48, 3), dtype=np.uint8)
            pixels[..., 0], pixels[..., 1] = index * 15, np.arange(48)[None] * 4
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    vocabulary = {"schema_version": 1, "dataset": "test_fixture", "action_count": 3,
                  "labels": ["first", "second", "third"]}
    (source / "vocabulary.json").write_text(json.dumps(vocabulary))
    row = {"id": "fixture", "video_id": "fixture", "task": "test", "split": "train", "video": str(video),
           "actions": [0, 1, 2], "state_times": [0, 1, 2, 3], "task_action_ids": [0, 1, 2],
           "action_count": 3, "vocabulary_sha256": vocabulary_hash(vocabulary)}
    (source / "train.jsonl").write_text(json.dumps(row) + "\n")
    schedule = {"batch_size": 2, "start_lr": 0.0001, "lr": 0.0002,
                "warmup_steps": 1, "hold_steps": 5, "decay_steps": 1}
    config = {"seed": 7,
              "model": {"encoder": {"name": "tiny", "pretrained": False, "frames": 4, "image_size": 32,
                                     "tiny_dim": 48, "tiny_depth": 1, "tiny_heads": 4},
                        "predictor": {"feature_dim": 48, "hidden_dim": 48, "depth": 1, "num_heads": 4,
                                      "num_actions": 3, "tangent_scale": 0.01}},
              "data": {"horizon": 2, "workers": 0},
              "training": {"micro_batch_size": 1, "save_every": 1, "log_every": 1,
                           "sft": {**schedule, "lambda_tf": 0.5}, "grl": {**schedule, "gamma": 0.99, "beta": 0.1}},
              "planning": {"num_samples": 12, "num_elites": 3, "iterations": 2}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cache = tmp_path / "cache"
    run("geoworld.extract", "--config", config_path, "--manifest", source / "train.jsonl",
        "--output", cache, "--device", "cpu", "--batch-size", 2)
    manifest = cache / "manifest.jsonl"
    cached_row = json.loads(manifest.read_text())
    features = np.load(cache / cached_row["features"])
    assert features.shape == (4, 48) and np.isfinite(features).all()
    common = ("--config", config_path, "--manifest", manifest, "--device", "cpu")
    sft, uninterrupted, grl = tmp_path / "sft", tmp_path / "continuous", tmp_path / "grl"
    run("geoworld.train", *common, "--stage", "sft", "--output", sft, "--max-steps", 2)
    run("geoworld.train", *common, "--stage", "sft", "--output", sft, "--max-steps", 3,
        "--resume", sft / "last.pt")
    run("geoworld.train", *common, "--stage", "sft", "--output", uninterrupted, "--max-steps", 3)
    resumed = torch.load(sft / "last.pt", weights_only=True)
    continuous = torch.load(uninterrupted / "last.pt", weights_only=True)
    for key in resumed["predictor"]:
        torch.testing.assert_close(resumed["predictor"][key], continuous["predictor"][key], rtol=0, atol=0)
    assert resumed["step"] == 3 and resumed["vocabulary"] == vocabulary
    run("geoworld.train", *common, "--stage", "grl", "--output", grl, "--max-steps", 1,
        "--initialize", sft / "last.pt")
    plan = tmp_path / "plan.json"
    run("geoworld.plan", "--checkpoint", grl / "last.pt", "--features", cache / cached_row["features"],
        "--horizon", 2, "--output", plan, "--device", "cpu")
    result = json.loads(plan.read_text())
    assert len(result["action_ids"]) == 2
    assert result["action_labels"] == [vocabulary["labels"][i] for i in result["action_ids"]]
    assert result["action_space"] == "full_dataset_vocabulary"
    evaluation = tmp_path / "evaluation.json"
    run("geoworld.evaluate", "--checkpoint", grl / "last.pt", "--manifest", manifest,
        "--horizon", 2, "--output", evaluation, "--device", "cpu")
    report = json.loads(evaluation.read_text())
    assert report["examples"] == 2
    assert all(0 <= value <= 100 for value in report["metrics_percent"].values())
