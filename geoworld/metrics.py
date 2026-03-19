"""Procedure-planning metrics reported as percentages."""
from __future__ import annotations
import torch


def planning_metrics(predictions, targets, mask=None):
    predictions, targets = torch.as_tensor(predictions), torch.as_tensor(targets)
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.numel() == 0:
        raise ValueError("Predictions and targets must have identical [B,T] shapes")
    valid = torch.ones_like(targets, dtype=torch.bool) if mask is None else torch.as_tensor(mask, dtype=torch.bool)
    if valid.shape != targets.shape or not valid.any(dim=1).all():
        raise ValueError("Each example must contain at least one valid target action")
    equal = predictions.eq(targets)
    success = (equal | ~valid).all(dim=1).float().mean()
    accuracy = ((equal & valid).sum(dim=1) / valid.sum(dim=1)).mean()
    overlaps = []
    for pred, target, keep in zip(predictions, targets, valid):
        p, t = set(pred[keep].tolist()), set(target[keep].tolist())
        overlaps.append(len(p & t) / len(p | t))
    return {"SR": 100 * float(success), "mAcc": 100 * float(accuracy),
            "mIoU": 100 * sum(overlaps) / len(overlaps)}
