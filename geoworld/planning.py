"""Cross-Entropy Method search over discrete procedural action sequences."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .geometry import distance


class CategoricalCEMPlanner:
    """Refit per-step categorical action probabilities to low-energy elites.

    The objective is the final predicted state's hyperbolic distance to the
    goal (Eq. 20), never a ground-truth intermediate trajectory. Optional task
    action masks require known task identity and must be reported at evaluation.
    """

    def __init__(self, num_actions: int, horizon: int, num_samples: int = 800,
                 num_elites: int = 80, iterations: int = 10,
                 exploration: float = 0.05, evaluation_batch_size: int = 256):
        integers = (num_actions, horizon, num_samples, num_elites, iterations, evaluation_batch_size)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in integers):
            raise ValueError("Action count, horizon, samples, elites, iterations and batch size must be positive integers")
        if num_elites > num_samples:
            raise ValueError("num_elites cannot exceed num_samples")
        if not math.isfinite(exploration) or not 0 <= exploration < 1:
            raise ValueError("exploration must be in [0,1)")
        self.num_actions, self.horizon = num_actions, horizon
        self.num_samples, self.num_elites, self.iterations = num_samples, num_elites, iterations
        self.exploration, self.evaluation_batch_size = exploration, evaluation_batch_size

    def _allowed(self, action_mask: Tensor | None, batch: int, device) -> Tensor:
        shape = (batch, self.horizon, self.num_actions)
        if action_mask is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        allowed = torch.as_tensor(action_mask, device=device)
        if allowed.dtype != torch.bool:
            raise ValueError("Action mask must be boolean, True for permitted actions")
        if allowed.shape == (self.num_actions,):
            allowed = allowed[None, None].expand(shape)
        elif allowed.shape == (batch, self.num_actions):
            allowed = allowed[:, None].expand(shape)
        elif allowed.shape != shape:
            raise ValueError("Action mask must be [A], [B,A] or [B,T,A]")
        if not bool(allowed.any(-1).all()):
            raise ValueError("Every example/time step needs at least one permitted action")
        return allowed

    def _energies(self, predictor, initial: Tensor, goal: Tensor, actions: Tensor) -> Tensor:
        batch, samples, horizon = actions.shape
        starts = initial[:, None].expand(-1, samples, -1).reshape(batch * samples, -1)
        targets = goal[:, None].expand(-1, samples, -1).reshape(batch * samples, -1)
        flattened = actions.reshape(batch * samples, horizon)
        costs = []
        for offset in range(0, batch * samples, self.evaluation_batch_size):
            stop = offset + self.evaluation_batch_size
            predicted = predictor.rollout(starts[offset:stop], flattened[offset:stop])
            if predicted.shape != (len(starts[offset:stop]), horizon + 1, initial.shape[-1]):
                raise ValueError("predictor.rollout must return [B,T+1,D], including initial state")
            costs.append(distance(predicted[:, -1], targets[offset:stop], predictor.curvature))
        energy = torch.cat(costs).reshape(batch, samples)
        if not bool(torch.isfinite(energy).all()):
            raise FloatingPointError("Planning encountered a nonfinite energy")
        return energy

    @torch.no_grad()
    def plan(self, predictor, initial: Tensor, goal: Tensor, action_mask: Tensor | None = None,
             generator: torch.Generator | None = None) -> dict[str, Tensor]:
        if initial.ndim != 2 or goal.shape != initial.shape or initial.shape[0] == 0:
            raise ValueError("Initial and goal ball states must have identical nonempty [B,D] shapes")
        if not bool(torch.isfinite(initial).all() & torch.isfinite(goal).all()):
            raise ValueError("Planning states must be finite")
        batch = initial.shape[0]
        allowed = self._allowed(action_mask, batch, initial.device)
        uniform = allowed.to(initial.dtype) / allowed.sum(-1, keepdim=True)
        probabilities = uniform.clone()
        best_actions = None
        best_energy = initial.new_full((batch,), torch.inf)
        history = []
        for _ in range(self.iterations):
            sampled = torch.multinomial(probabilities.reshape(-1, self.num_actions),
                                        self.num_samples, replacement=True, generator=generator)
            candidates = sampled.reshape(batch, self.horizon, self.num_samples).transpose(1, 2).contiguous()
            if best_actions is not None:
                candidates[:, 0] = best_actions
            energy = self._energies(predictor, initial, goal, candidates)
            values, indices = energy.topk(self.num_elites, dim=1, largest=False, sorted=True)
            elites = candidates.gather(1, indices[..., None].expand(-1, -1, self.horizon))
            improve = values[:, 0] < best_energy
            if best_actions is None:
                best_actions = elites[:, 0].clone()
            else:
                best_actions = torch.where(improve[:, None], elites[:, 0], best_actions)
            best_energy = torch.minimum(best_energy, values[:, 0])
            frequencies = F.one_hot(elites, self.num_actions).to(initial.dtype).mean(1)
            probabilities = (1 - self.exploration) * frequencies + self.exploration * uniform
            probabilities = probabilities.masked_fill(~allowed, 0)
            probabilities = probabilities / probabilities.sum(-1, keepdim=True)
            history.append(best_energy.clone())
        trajectory = predictor.rollout(initial, best_actions)
        return {"actions": best_actions, "energy": best_energy, "trajectory": trajectory,
                "probabilities": probabilities, "energy_history": torch.stack(history, 1)}
