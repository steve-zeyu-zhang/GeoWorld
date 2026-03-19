# Portions adapted from Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in third_party/vjepa2/LICENSE.

"""Discrete-action H-JEPA using the original V-JEPA 2-AC transformer blocks.

The action/state token layout and initialization adapt Meta's MIT-licensed
src/models/ac_predictor.py. The exact upstream attention blocks are vendored in
third_party/vjepa2; local changes for symbolic actions and ball-valued states
are described in docs/MODEL.md.
"""

from __future__ import annotations

import math
from functools import partial

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from geoworld.geometry import expmap0, project
from third_party.vjepa2.src.models.utils.modules import ACBlock, build_action_block_causal_attention_mask
from third_party.vjepa2.src.utils.tensors import trunc_normal_


class HyperbolicPredictor(nn.Module):
    """Autoregressive predictor with one action and one visual token per step."""

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        num_actions: int = 105,
        curvature_init: float = 1.0,
        curvature_min: float = 0.1,
        curvature_max: float = 10.0,
        tangent_scale: float = 0.01,
        mlp_ratio: float = 4.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if min(feature_dim, hidden_dim, depth, num_heads, num_actions) < 1:
            raise ValueError("Dimensions, depth, and action count must be positive")
        if hidden_dim % num_heads or hidden_dim // num_heads < 12:
            raise ValueError("hidden_dim/num_heads must be an integer >= 12 for checkpoint-compatible 3D RoPE")
        if not (0 < curvature_min <= curvature_init <= curvature_max < float("inf")):
            raise ValueError("Require 0 < curvature_min <= curvature_init <= curvature_max < infinity")
        if not math.isfinite(tangent_scale) or tangent_scale <= 0:
            raise ValueError("tangent_scale must be finite and positive")
        self.feature_dim, self.hidden_dim = feature_dim, hidden_dim
        self.num_actions, self.depth, self.num_heads = num_actions, depth, num_heads
        self.curvature_min, self.curvature_max = curvature_min, curvature_max
        self.tangent_scale = tangent_scale
        self.activation_checkpointing = activation_checkpointing
        self.log_c = nn.Parameter(torch.tensor(math.log(curvature_init), dtype=torch.float32))
        self.predictor_embed = nn.Linear(feature_dim, hidden_dim)
        self.action_encoder = nn.Embedding(num_actions, hidden_dim)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.predictor_blocks = nn.ModuleList([
            ACBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
                act_layer=nn.GELU,
                use_sdpa=True,
                use_rope=True,
                grid_size=1,
                is_causal=False,
            )
            for _ in range(depth)
        ])
        self.predictor_norm = norm_layer(hidden_dim)
        self.predictor_proj = nn.Linear(hidden_dim, feature_dim)
        self.apply(self._init_weights)
        with torch.no_grad():
            for index, layer in enumerate(self.predictor_blocks, start=1):
                layer.attn.proj.weight.div_(math.sqrt(2 * index))
                layer.mlp.fc2.weight.div_(math.sqrt(2 * index))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            trunc_normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @property
    def curvature(self) -> Tensor:
        """Positive magnitude c of sectional curvature -c."""
        return self.log_c.clamp(math.log(self.curvature_min), math.log(self.curvature_max)).exp()

    def encode_states(self, features: Tensor) -> Tensor:
        """Map frozen Euclidean features into the ball with fixed input scale."""
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature dimension {self.feature_dim}, got {features.shape[-1]}")
        # Geometry stays float32 under mixed-precision transformer execution.
        dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
        return expmap0(features.to(dtype) * self.tangent_scale, self.curvature.to(dtype))

    def forward_sequence(self, states: Tensor, actions: Tensor) -> Tensor:
        """Predict [B,T,D] next states with causal access to preceding pairs.

        State tokens contain ball coordinates, matching the paper's P(z,a).
        Output linear coordinates are mapped through Exp_0 to remain inside
        the ball. Future state/action tokens are blocked from attention.
        """
        if states.ndim != 3 or states.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected states [B,T,{self.feature_dim}]")
        if actions.shape != states.shape[:2] or actions.dtype != torch.long:
            raise ValueError("Actions must be int64 [B,T] aligned with states")
        if states.shape[1] < 1:
            raise ValueError("forward_sequence requires at least one state-action pair")
        if bool((actions < 0).any()) or bool((actions >= self.num_actions).any()):
            raise ValueError(f"Action IDs must be in [0,{self.num_actions})")
        batch, steps, _ = states.shape
        geometry_dtype = torch.float64 if states.dtype == torch.float64 else torch.float32
        states = project(states.to(geometry_dtype), self.curvature.to(geometry_dtype))
        visual = self.predictor_embed(states.to(self.predictor_embed.weight.dtype))
        action = self.action_encoder(actions)
        tokens = torch.stack((action, visual), dim=2).reshape(batch, 2 * steps, self.hidden_dim)
        attn_mask = build_action_block_causal_attention_mask(steps, 1, 1, add_tokens=1).to(states.device)
        for block in self.predictor_blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                def apply_block(x: Tensor, layer: nn.Module = block) -> Tensor:
                    return layer(x, attn_mask=attn_mask, T=steps, H=1, W=1, action_tokens=1)
                tokens = checkpoint(apply_block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens, attn_mask=attn_mask, T=steps, H=1, W=1, action_tokens=1)
        visual = tokens.reshape(batch, steps, 2, self.hidden_dim)[:, :, 1]
        tangent = self.predictor_proj(self.predictor_norm(visual)).to(geometry_dtype)
        return expmap0(tangent * self.tangent_scale, self.curvature.to(geometry_dtype))

    def forward(self, state_on_ball: Tensor, actions: Tensor) -> Tensor:
        """One transition: [B,D] ball state and int64 [B] action -> [B,D]."""
        if state_on_ball.ndim != 2 or actions.ndim != 1:
            raise ValueError("Expected state [B,D] and actions [B]")
        return self.forward_sequence(state_on_ball[:, None], actions[:, None])[:, 0]

    def rollout(self, initial: Tensor, actions: Tensor) -> Tensor:
        """Return [B,T+1,D], retaining causal state/action history at each step."""
        if initial.ndim != 2 or actions.ndim != 2 or actions.shape[0] != initial.shape[0]:
            raise ValueError("Expected initial [B,D] and actions [B,T]")
        if actions.dtype != torch.long:
            raise ValueError("Actions must use int64 IDs")
        states = [initial]
        for index in range(actions.shape[1]):
            history = torch.stack(states, dim=1)
            prediction = self.forward_sequence(history, actions[:, : index + 1])[:, -1]
            states.append(prediction)
        return torch.stack(states, dim=1)
