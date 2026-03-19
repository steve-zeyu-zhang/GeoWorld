"""Poincare ball geometry with sectional curvature -c, c > 0.

Vectors occupy the last axis. Curvature is a differentiable positive scalar.
The origin metric is 4I, so d(0, expmap0(v)) = 2 ||v||.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _curvature(c: Tensor | float, reference: Tensor) -> Tensor:
    if reference.dtype not in (torch.float32, torch.float64):
        raise TypeError("Hyperbolic geometry requires float32 or float64 tensors")
    curvature = torch.as_tensor(c, dtype=reference.dtype, device=reference.device)
    if curvature.numel() != 1 or not bool(torch.isfinite(curvature).all()) or bool((curvature <= 0).any()):
        raise ValueError("Curvature magnitude c must be a finite positive scalar")
    return curvature.reshape(())


def _margin(reference: Tensor, eps: float | None) -> float:
    margin = (1e-5 if reference.dtype == torch.float32 else 1e-10) if eps is None else eps
    if not 0 < margin < 1:
        raise ValueError("Ball boundary margin must be in (0,1)")
    return margin


def project(x: Tensor, c: Tensor | float = 1.0, eps: float | None = None) -> Tensor:
    """Clip to ||x|| <= (1-eps)/sqrt(c); preserve all interior points exactly."""
    curvature = _curvature(c, x)
    radius = (1 - _margin(x, eps)) / curvature.sqrt()
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    scale = (radius / norm.clamp_min(torch.finfo(x.dtype).tiny)).clamp_max(1)
    return x * scale


def expmap0(v: Tensor, c: Tensor | float = 1.0) -> Tensor:
    """GeoWorld Eq. 2: tanh(sqrt(c)||v||) v / (sqrt(c)||v||)."""
    curvature = _curvature(c, v)
    r2 = curvature * v.square().sum(-1, keepdim=True)
    r = r2.clamp_min(1e-16).sqrt()
    coefficient = torch.where(r2 < 1e-6,
                              1 - r2 / 3 + 2 * r2.square() / 15,
                              torch.tanh(r) / r)
    return project(coefficient * v, curvature)


def logmap0(x: Tensor, c: Tensor | float = 1.0) -> Tensor:
    """Inverse exponential map on the numerically represented open ball."""
    curvature = _curvature(c, x)
    x = project(x, curvature)
    r2 = curvature * x.square().sum(-1, keepdim=True)
    r = r2.clamp_min(1e-16).sqrt().clamp_max(1 - _margin(x, None) / 2)
    coefficient = torch.where(r2 < 1e-6,
                              1 + r2 / 3 + r2.square() / 5,
                              torch.atanh(r) / r)
    return coefficient * x


def distance(x: Tensor, y: Tensor, c: Tensor | float = 1.0) -> Tensor:
    """Poincare distance (Eq. 6), with an asinh form stable at x == y.

    acosh(1+2u^2) = 2 asinh(u). This avoids acosh's singular derivative
    at its identity argument and produces an exact zero for identical points.
    torch.linalg.vector_norm supplies the zero subgradient there.
    """
    curvature = _curvature(c, x)
    x, y = project(x, curvature), project(y, curvature)
    x_factor = (1 - curvature * x.square().sum(-1)).clamp_min(torch.finfo(x.dtype).eps)
    y_factor = (1 - curvature * y.square().sum(-1)).clamp_min(torch.finfo(x.dtype).eps)
    separation = torch.linalg.vector_norm(x - y, dim=-1)
    argument = curvature.sqrt() * separation / (x_factor * y_factor).sqrt()
    return 2 * torch.asinh(argument) / curvature.sqrt()
