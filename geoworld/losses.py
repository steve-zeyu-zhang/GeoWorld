"""Hyperbolic JEPA SFT and the literal GeoWorld GRL objective."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .geometry import distance, expmap0


def _prepare(predictor, features: Tensor, actions: Tensor, mask: Tensor | None):
    if features.ndim != 3 or actions.ndim != 2 or features.shape[:2] != (actions.shape[0], actions.shape[1] + 1):
        raise ValueError("Expected features [B,T+1,D] and actions [B,T]")
    if actions.dtype != torch.long or actions.shape[1] < 1:
        raise ValueError("Actions must be int64 and contain at least one transition")
    if mask is None:
        mask = torch.ones_like(actions, dtype=torch.bool)
    if mask.shape != actions.shape or mask.dtype != torch.bool:
        raise ValueError("Action mask must be boolean [B,T], True for valid transitions")
    if not bool(mask.any()):
        raise ValueError("At least one valid transition is required")
    if bool((actions[mask] < 0).any()):
        raise ValueError("Valid actions cannot be negative")
    safe_actions = actions.masked_fill(~mask, 0)
    states = predictor.encode_states(features) if hasattr(predictor, "encode_states") else expmap0(features, predictor.curvature)
    return states, safe_actions, mask


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if not bool(mask.any()):
        return values.sum() * 0
    # Each trajectory has its own 1/T normalization before the batch average.
    counts = mask.sum(-1)
    per_example = values.masked_fill(~mask, 0).sum(-1) / counts.clamp_min(1)
    return per_example[counts > 0].mean()


def triangle_regularization(states: Tensor, c: Tensor | float, mask: Tensor | None = None) -> Tensor:
    """Literal Eq. 17: [d(s0,s2)-d(s0,s1)-d(s1,s2)]_+.

    This is mathematically zero for a genuine metric. It is retained exactly,
    without changing the sign to create an unreported path-length penalty.
    """
    if states.ndim != 3:
        raise ValueError("Triangle states must have shape [B,L,D]")
    if states.shape[1] < 3:
        return states.sum() * 0
    if mask is None:
        mask = torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
    if mask.shape != states.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Triangle mask must be boolean [B,L]")
    violation = (distance(states[:, :-2], states[:, 2:], c)
                 - distance(states[:, :-2], states[:, 1:-1], c)
                 - distance(states[:, 1:-1], states[:, 2:], c)).relu()
    return _masked_mean(violation, mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:])


def sft_loss(predictor, features: Tensor, actions: Tensor, mask: Tensor | None = None,
             lambda_tf: float = 0.5) -> dict[str, Tensor]:
    """Eq. 10: mean one-step error and mean two-step error over valid windows."""
    if not math.isfinite(lambda_tf) or not 0 <= lambda_tf <= 1:
        raise ValueError("lambda_tf must be in [0,1]")
    states, actions, valid = _prepare(predictor, features, actions, mask)
    batch, horizon, width = states.shape[0], actions.shape[1], states.shape[-1]
    if hasattr(predictor, "forward_sequence"):
        first = predictor.forward_sequence(states[:, :-1], actions)
    else:
        first = predictor(states[:, :-1].reshape(-1, width), actions.reshape(-1)).reshape(batch, horizon, width)
    teacher = _masked_mean(distance(first, states[:, 1:], predictor.curvature), valid)
    if horizon >= 2:
        pairs = actions.unfold(1, 2, 1).reshape(-1, 2)
        second = predictor.rollout(states[:, :-2].reshape(-1, width), pairs)[:, -1].reshape(batch, horizon - 1, width)
        rollout = _masked_mean(distance(second, states[:, 2:], predictor.curvature), valid[:, :-1] & valid[:, 1:])
    else:
        rollout = first.sum() * 0
    return {"loss": lambda_tf * teacher + (1 - lambda_tf) * rollout,
            "teacher_forcing": teacher, "rollout": rollout}


def grl_loss(predictor, features: Tensor, actions: Tensor, mask: Tensor | None = None,
             gamma: float = 0.99, beta: float = 0.1) -> dict[str, Tensor]:
    """Eq. 18 discounted rollout energy, using observed offline action sequences.

    The objective averages over recorded action trajectories and differentiates
    through the predictor rollout. There is no separate policy or reward model.
    """
    if not math.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0,1)")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    states, actions, valid = _prepare(predictor, features, actions, mask)
    # An autoregressive horizon cannot continue after an unknown padded action.
    if bool(((~valid[:, :-1]) & valid[:, 1:]).any()):
        raise ValueError("GRL action masks must be contiguous valid prefixes")
    predicted = predictor.rollout(states[:, 0], actions)[:, 1:]
    if predicted.shape != states[:, 1:].shape:
        raise ValueError("predictor.rollout must return [B,T+1,D], including initial state")
    costs = distance(predicted, states[:, 1:], predictor.curvature)
    discount = gamma ** torch.arange(actions.shape[1], dtype=costs.dtype, device=costs.device)
    # Preserve the paper's SUM over time, then average over nonempty examples.
    energy = ((costs * discount).masked_fill(~valid, 0).sum(-1))[valid.any(-1)].mean()
    triangle = triangle_regularization(predicted, predictor.curvature, valid)
    return {"loss": energy + beta * triangle, "energy": energy,
            "triangle": triangle, "return": -energy}
