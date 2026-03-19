"""Frozen original V-JEPA 2 encoders and deterministic video preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from third_party.vjepa2.src.models import vision_transformer as official_vit


CHECKPOINT_URLS = {
    "vit_large": "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt",
    "vit_huge": "https://dl.fbaipublicfiles.com/vjepa2/vith.pt",
    "vit_giant": "https://dl.fbaipublicfiles.com/vjepa2/vitg.pt",
    "vit_giant_384": "https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt",
}
_ARCHITECTURES = {
    "vit_large": "vit_large",
    "vit_huge": "vit_huge",
    "vit_giant": "vit_giant_xformers",
    "vit_giant_384": "vit_giant_xformers",
}


def _clean_encoder_keys(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    clean = {}
    for name, value in state.items():
        while name.startswith(("module.", "backbone.")):
            name = name.split(".", 1)[1]
        if name in clean:
            raise ValueError(f"Duplicate encoder key after prefix removal: {name}")
        clean[name] = value
    return clean


class FrozenVJEPA2Encoder(nn.Module):
    """Load the 2025 V-JEPA 2 architecture, freeze it, and pool patch features.

    ``forward`` accepts raw RGB clips [B,3,F,H,W], uint8 or float in [0,1].
    The explicitly named ``tiny`` architecture is a random, test-only encoder;
    production architectures require a checkpoint or ``pretrained=True``.
    """

    def __init__(
        self,
        name: str = "vit_large",
        checkpoint: str | Path | None = None,
        frames: int = 16,
        image_size: int = 256,
        pretrained: bool = True,
        spatial_pool: str = "mean",
        temporal_pool: str = "mean",
        tiny_dim: int = 64,
        tiny_depth: int = 2,
        tiny_heads: int = 4,
    ) -> None:
        super().__init__()
        if frames < 2 or frames % 2:
            raise ValueError("frames must be an even integer >= 2 for V-JEPA 2 tubelets")
        if image_size < 16 or image_size % 16:
            raise ValueError("image_size must be a positive multiple of 16")
        if spatial_pool != "mean" or temporal_pool not in {"mean", "last"}:
            raise ValueError("Supported pooling: spatial_pool='mean', temporal_pool='mean' or 'last'")
        self.name, self.frames, self.image_size = name, frames, image_size
        self.spatial_pool, self.temporal_pool = spatial_pool, temporal_pool
        self.patch_size, self.tubelet_size = 16, 2
        self.checkpoint_source: str | None = None
        self.load_report: dict = {}
        kwargs = dict(
            img_size=image_size,
            patch_size=16,
            num_frames=frames,
            tubelet_size=2,
            use_sdpa=True,
            use_silu=False,
            uniform_power=False,
            use_rope=True,
        )
        if name == "tiny":
            if pretrained:
                raise ValueError("The tiny encoder is test-only; set pretrained=False explicitly")
            if tiny_dim % tiny_heads or tiny_dim // tiny_heads < 12:
                raise ValueError("tiny_dim/tiny_heads must be an integer >= 12 for checkpoint-compatible 3D RoPE")
            self.backbone = official_vit.VisionTransformer(
                embed_dim=tiny_dim, depth=tiny_depth, num_heads=tiny_heads, **kwargs
            )
        else:
            if name not in _ARCHITECTURES:
                raise ValueError(f"Unknown V-JEPA 2 encoder {name!r}; choose {list(_ARCHITECTURES)}")
            if checkpoint is None and not pretrained:
                raise ValueError("Production encoders require pretrained weights; use 'tiny' for random tests")
            self.backbone = getattr(official_vit, _ARCHITECTURES[name])(**kwargs)
        self.feature_dim = self.backbone.embed_dim
        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None, None], persistent=False)
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None, None], persistent=False)
        if checkpoint is not None:
            path = Path(checkpoint).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"V-JEPA 2 checkpoint not found: {path}")
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            self._load_encoder(state)
            self.checkpoint_source = str(path.resolve())
        elif pretrained:
            url = CHECKPOINT_URLS[name]
            state = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=True)
            self._load_encoder(state)
            self.checkpoint_source = url
        self.requires_grad_(False)
        self.train(False)

    def _load_encoder(self, checkpoint: Mapping) -> None:
        if not isinstance(checkpoint, Mapping) or "encoder" not in checkpoint:
            raise ValueError("Expected an official checkpoint mapping containing an 'encoder' state dict")
        clean = _clean_encoder_keys(checkpoint["encoder"])
        ignored = []
        # Original hub code explicitly permits this obsolete absolute embedding
        # when loading the checkpoint into the official rotary-embedding model.
        if "pos_embed" in clean and self.backbone.pos_embed is None:
            del clean["pos_embed"]
            ignored.append("pos_embed")
        reference = self.backbone.state_dict()
        missing = sorted(set(reference) - set(clean))
        unexpected = sorted(set(clean) - set(reference))
        mismatched = {
            key: {"checkpoint": list(clean[key].shape), "model": list(reference[key].shape)}
            for key in set(clean) & set(reference)
            if clean[key].shape != reference[key].shape
        }
        self.load_report = {
            "loaded_tensors": len(clean),
            "ignored_absolute_position_embeddings": ignored,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatches": mismatched,
        }
        if missing or unexpected or mismatched:
            raise RuntimeError(f"V-JEPA 2 checkpoint architecture mismatch: {self.load_report}")
        self.backbone.load_state_dict(clean, strict=True)

    def train(self, mode: bool = True) -> "FrozenVJEPA2Encoder":
        # Parent model.train() must never enable stochastic encoder operations.
        super().train(False)
        return self

    def preprocess(self, video: Tensor) -> Tensor:
        """Single-view official evaluation recipe, using tensor bilinear resize.

        Resize the shortest edge to floor(image_size*256/224), center crop,
        then normalize RGB with ImageNet statistics. Temporal sampling belongs
        to the video dataset; this method preserves all supplied frames.
        """
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("Expected video [B,3,F,H,W]")
        if video.shape[2] < 2 or video.shape[2] % self.tubelet_size:
            raise ValueError("Video must contain an even number of frames >= 2")
        if video.dtype == torch.uint8:
            video = video.float().div(255)
        elif not video.is_floating_point():
            raise TypeError("Video pixels must be uint8 or floating-point RGB in [0,1]")
        else:
            video = video.float()
        if not bool(torch.isfinite(video).all()) or bool((video < 0).any()) or bool((video > 1).any()):
            raise ValueError("Expected finite, unnormalized RGB pixels in [0,1]")
        batch, channels, frames, height, width = video.shape
        if min(height, width) < 1:
            raise ValueError("Video spatial dimensions must be positive")
        short_edge = int(self.image_size * 256 / 224)
        if height <= width:
            resize_h, resize_w = short_edge, int(width * short_edge / height)
        else:
            resize_h, resize_w = int(height * short_edge / width), short_edge
        pixels = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        pixels = F.interpolate(pixels, (resize_h, resize_w), mode="bilinear", align_corners=False, antialias=True)
        top, left = (resize_h - self.image_size) // 2, (resize_w - self.image_size) // 2
        pixels = pixels[..., top : top + self.image_size, left : left + self.image_size]
        video = pixels.reshape(batch, frames, channels, self.image_size, self.image_size).permute(0, 2, 1, 3, 4)
        return (video - self.pixel_mean.to(video)) / self.pixel_std.to(video)

    @torch.no_grad()
    def encode_tokens(self, video: Tensor, *, preprocess: bool = True) -> Tensor:
        """Return official final-layer tokens [B,F/2,H/16,W/16,D]."""
        video = self.preprocess(video) if preprocess else video
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("Expected video [B,3,F,H,W]")
        if video.shape[2] < 2 or video.shape[2] % 2:
            raise ValueError("Video must contain an even number of frames >= 2")
        if video.shape[-2] % 16 or video.shape[-1] % 16:
            raise ValueError("Preprocessed height/width must be divisible by patch size 16")
        batch, _, frames, height, width = video.shape
        tokens = self.backbone(video)
        return tokens.reshape(batch, frames // 2, height // 16, width // 16, self.feature_dim)

    @torch.no_grad()
    def forward(self, video: Tensor, *, preprocess: bool = True) -> Tensor:
        tokens = self.encode_tokens(video, preprocess=preprocess).float()
        spatial = tokens.mean(dim=(2, 3))
        return spatial.mean(dim=1) if self.temporal_pool == "mean" else spatial[:, -1]
