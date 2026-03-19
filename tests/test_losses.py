import pytest
import torch
from torch import nn

from geoworld.geometry import distance, expmap0, logmap0
from geoworld.losses import grl_loss, sft_loss, triangle_regularization


class AdditivePredictor(nn.Module):
    """Exact tangent translations make known trajectories analytically testable."""
    def __init__(self, step=0.2):
        super().__init__()
        self.step = nn.Parameter(torch.tensor(step, dtype=torch.float64))
        self.log_c = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.used_sequence = False

    @property
    def curvature(self):
        return self.log_c.exp()

    def encode_states(self, features):
        return expmap0(features, self.curvature)

    def forward(self, state, action):
        direction = torch.nn.functional.one_hot(action, state.shape[-1]).to(state)
        return expmap0(logmap0(state, self.curvature) + self.step * direction, self.curvature)

    def forward_sequence(self, states, actions):
        self.used_sequence = True
        return self(states, actions)

    def rollout(self, initial, actions):
        states = [initial]
        for action in actions.unbind(1):
            states.append(self(states[-1], action))
        return torch.stack(states, 1)


def example():
    actions = torch.tensor([[0, 1, 0], [1, 0, 1]])
    increments = torch.nn.functional.one_hot(actions, 2).double() * 0.2
    features = torch.cat((torch.zeros(2, 1, 2, dtype=torch.float64), increments.cumsum(1)), 1)
    return features, actions


def test_sft_exact_one_and_two_step_teacher_trajectory():
    features, actions = example()
    predictor = AdditivePredictor()
    result = sft_loss(predictor, features, actions)
    assert predictor.used_sequence
    assert result["loss"].item() < 1e-12
    assert result["rollout"].item() < 1e-12


def test_two_step_rollout_is_autoregressive_and_masked():
    features, actions = example()
    predictor = AdditivePredictor(step=0.1)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    padded_actions = actions.masked_fill(~mask, -1)
    result = sft_loss(predictor, features, padded_actions, mask, lambda_tf=0.25)
    states = predictor.encode_states(features)
    first = predictor.forward_sequence(states[:, :-1], actions)
    errors = distance(first, states[:, 1:], predictor.curvature)
    expected_teacher = (errors[0].mean() + errors[1, 0]) / 2
    independent_two_step = predictor.rollout(states[0, :2], actions[0].unfold(0, 2, 1))[:, -1]
    expected_rollout = distance(independent_two_step, states[0, 2:], predictor.curvature).mean()
    torch.testing.assert_close(result["teacher_forcing"], expected_teacher)
    torch.testing.assert_close(result["rollout"], expected_rollout)
    torch.testing.assert_close(result["loss"], .25 * expected_teacher + .75 * expected_rollout)
    result["loss"].backward()
    assert torch.isfinite(predictor.step.grad) and predictor.step.grad.abs() > 0
    assert torch.isfinite(predictor.log_c.grad)


def test_grl_uses_discounted_sum_and_literal_zero_triangle_hinge():
    features, actions = example()
    predictor = AdditivePredictor(step=0.1)
    states = predictor.encode_states(features)
    predicted = predictor.rollout(states[:, 0], actions)[:, 1:]
    errors = distance(predicted, states[:, 1:], predictor.curvature)
    expected = (errors * torch.tensor([1., .99, .99**2], dtype=torch.float64)).sum(-1).mean()
    result = grl_loss(predictor, features, actions)
    torch.testing.assert_close(result["energy"], expected)
    torch.testing.assert_close(result["return"], -expected)
    assert result["triangle"].item() < 1e-12
    result["loss"].backward()
    assert predictor.step.grad.abs() > 0


def test_triangle_hinge_is_zero_for_non_geodesic_curve():
    # A right-angle path is not a geodesic, yet a metric still satisfies Eq17.
    states = expmap0(torch.tensor([[[0.,0.],[.3,0.],[.3,.3]]], dtype=torch.float64))
    assert triangle_regularization(states, 1).item() == 0
    assert distance(states[:,0],states[:,1]) + distance(states[:,1],states[:,2]) > distance(states[:,0],states[:,2])


def test_masking_invalid_padding_preserves_valid_losses():
    features, actions = example()
    predictor = AdditivePredictor(step=0.15)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    baseline = grl_loss(predictor, features, actions, mask)
    changed_features = features.clone()
    changed_features[1, 2:] += 5
    changed_actions = actions.masked_fill(~mask, -1)
    changed = grl_loss(predictor, changed_features, changed_actions, mask)
    torch.testing.assert_close(changed["energy"], baseline["energy"])
    with pytest.raises(ValueError, match="contiguous"):
        grl_loss(predictor, features, actions, torch.tensor([[True, False, True], [True, True, True]]))


def test_one_transition_has_no_two_step_term_and_invalid_masks_fail():
    features, actions = example()
    predictor = AdditivePredictor(step=.1)
    result = sft_loss(predictor, features[:,:2], actions[:,:1])
    assert result["rollout"].item() == 0
    torch.testing.assert_close(result["loss"], .5 * result["teacher_forcing"])
    with pytest.raises(ValueError, match="valid transition"):
        sft_loss(predictor, features, actions, torch.zeros_like(actions, dtype=torch.bool))
