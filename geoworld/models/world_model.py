"""Frozen perception and trainable hyperbolic dynamics."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn

from .encoder import FrozenVJEPA2Encoder
from .predictor import HyperbolicPredictor


class GeoWorldModel(nn.Module):
    def __init__(self, config: Mapping) -> None:
        super().__init__()
        self.encoder = FrozenVJEPA2Encoder(**dict(config.get("encoder", {})))
        predictor_config = dict(config.get("predictor", {}))
        predictor_config.setdefault("feature_dim", self.encoder.feature_dim)
        if predictor_config["feature_dim"] != self.encoder.feature_dim:
            raise ValueError("Predictor feature_dim must match the frozen encoder output")
        self.predictor = HyperbolicPredictor(**predictor_config)
        self.feature_dim = self.encoder.feature_dim

    def preprocess(self, video: Tensor) -> Tensor:
        return self.encoder.preprocess(video)

    def encode(self, video: Tensor, *, preprocess: bool = True) -> Tensor:
        """Return frozen Euclidean features [B,D] from raw RGB [B,3,F,H,W]."""
        return self.encoder(video, preprocess=preprocess)

    def forward(self, video: Tensor, actions: Tensor) -> Tensor:
        return self.predictor(self.predictor.encode_states(self.encode(video)), actions)
