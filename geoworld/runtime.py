"""Configuration, checkpoints, and reproducible runtime setup."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def read_config(path):
    with open(path) as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError("Configuration must contain a model mapping")
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, device="cpu"):
    # Our checkpoints contain only tensors and primitive configuration values.
    return torch.load(path, map_location=device, weights_only=True)


def load_predictor(path, device="cpu"):
    from geoworld.models import HyperbolicPredictor
    checkpoint = load_checkpoint(path)
    predictor = HyperbolicPredictor(**checkpoint["config"]["model"]["predictor"])
    predictor.load_state_dict(checkpoint["predictor"], strict=True)
    return predictor.to(device).eval(), checkpoint


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
