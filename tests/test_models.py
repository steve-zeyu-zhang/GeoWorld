from pathlib import Path

import pytest
import torch

from geoworld.geometry import distance
from geoworld.models import FrozenVJEPA2Encoder, GeoWorldModel, HyperbolicPredictor
from third_party.vjepa2.src.models.utils.modules import ACBlock, ACRoPEAttention


@pytest.fixture(autouse=True)
def deterministic_small_models():
    torch.manual_seed(11)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_predictor(**kwargs):
    defaults = dict(feature_dim=32, hidden_dim=48, depth=2, num_heads=4, num_actions=7)
    return HyperbolicPredictor(**(defaults | kwargs))


def make_encoder(**kwargs):
    defaults = dict(name="tiny", pretrained=False, image_size=32, frames=4, tiny_dim=48, tiny_heads=4)
    return FrozenVJEPA2Encoder(**(defaults | kwargs))


def test_encoder_uses_original_transformer_and_remains_frozen():
    encoder = make_encoder()
    encoder.train()
    assert not encoder.training and not encoder.backbone.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert encoder.backbone.__class__.__module__.startswith("third_party.vjepa2")
    pixels = torch.rand(2, 3, 4, 40, 60, requires_grad=True)
    features = encoder(pixels)
    tokens = encoder.encode_tokens(pixels)
    assert features.shape == (2, 48)
    assert tokens.shape == (2, 2, 2, 2, 48)
    assert not features.requires_grad
    torch.testing.assert_close(features, tokens.mean((1, 2, 3)))
    torch.testing.assert_close(features, encoder(pixels))


def test_preprocessing_normalizes_rgb_and_preserves_temporal_axis():
    encoder = make_encoder()
    pixels = torch.full((1, 3, 4, 60, 40), 255, dtype=torch.uint8)
    result = encoder.preprocess(pixels)
    assert result.shape == (1, 3, 4, 32, 32)
    expected = (torch.ones(3) - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(result[0, :, 0, 0, 0], expected)
    with pytest.raises(ValueError, match="unnormalized"):
        encoder.preprocess(torch.full((1, 3, 4, 32, 32), -1.0))


def test_last_temporal_pool_matches_last_tubelet():
    encoder = make_encoder(temporal_pool="last")
    pixels = torch.rand(1, 3, 4, 32, 32)
    tokens = encoder.encode_tokens(pixels)
    torch.testing.assert_close(encoder(pixels), tokens[:, -1].mean((1, 2)))


def test_checkpoint_prefix_cleanup_is_exact_and_reported(tmp_path: Path):
    encoder = make_encoder()
    state = {"module.backbone." + key: value for key, value in encoder.backbone.state_dict().items()}
    state["module.backbone.pos_embed"] = torch.zeros(1, 8, 48)
    path = tmp_path / "encoder.pt"
    torch.save({"encoder": state, "predictor": {"unused": torch.zeros(1)}}, path)
    loaded = make_encoder(checkpoint=path)
    assert loaded.load_report["ignored_absolute_position_embeddings"] == ["pos_embed"]
    assert loaded.load_report["missing_keys"] == []
    assert loaded.load_report["unexpected_keys"] == []
    for key, value in encoder.backbone.state_dict().items():
        torch.testing.assert_close(value, loaded.backbone.state_dict()[key], rtol=0, atol=0)
    del state[next(iter(state))]
    torch.save({"encoder": state}, path)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        make_encoder(checkpoint=path)


def test_random_production_encoder_is_rejected_before_allocation():
    with pytest.raises(ValueError, match="Production encoders"):
        FrozenVJEPA2Encoder(name="vit_large", pretrained=False)
    with pytest.raises(ValueError, match="test-only"):
        FrozenVJEPA2Encoder(name="tiny", pretrained=True)


def test_too_narrow_rotary_heads_are_rejected():
    with pytest.raises(ValueError, match=">= 12"):
        make_predictor(hidden_dim=32, num_heads=4)
    with pytest.raises(ValueError, match=">= 12"):
        make_encoder(tiny_dim=32, tiny_heads=4)


def test_default_feature_scale_keeps_pretrained_sized_norms_inside_ball():
    model = make_predictor()
    features = torch.ones(2, 32) * (61.5 / 32**0.5)
    states = model.encode_states(features)
    expected_radius = torch.tanh(torch.tensor(0.615))
    torch.testing.assert_close(states.norm(dim=-1), expected_radius.expand(2))
    assert bool((states.norm(dim=-1) < 0.8).all())


def test_action_conditioning_and_causal_history():
    model = make_predictor().eval()
    assert isinstance(model.predictor_blocks[0], ACBlock)
    assert isinstance(model.predictor_blocks[0].attn, ACRoPEAttention)
    states = model.encode_states(torch.randn(2, 4, 32))
    actions = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 6]])
    predictions = model.forward_sequence(states, actions)
    changed_states, changed_actions = states.clone(), actions.clone()
    changed_states[:, 2:] = model.encode_states(torch.randn(2, 2, 32) * 2)
    changed_actions[:, 2:] = (changed_actions[:, 2:] + 1) % model.num_actions
    changed = model.forward_sequence(changed_states, changed_actions)
    torch.testing.assert_close(predictions[:, :2], changed[:, :2], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(predictions[:, 2:], changed[:, 2:])
    changed_actions = actions.clone()
    changed_actions[:, 0] = (changed_actions[:, 0] + 1) % model.num_actions
    conditioned = model.forward_sequence(states, changed_actions)
    assert not torch.allclose(predictions[:, 0], conditioned[:, 0])
    # The effect of the first action persists into later states through causal attention.
    assert not torch.allclose(predictions[:, -1], conditioned[:, -1])


def test_rollout_uses_history_has_valid_ball_outputs_and_curvature_gradients():
    model = make_predictor()
    initial = model.encode_states(torch.randn(2, 32))
    actions = torch.tensor([[0, 1, 2], [3, 4, 5]])
    trajectory = model.rollout(initial, actions)
    assert trajectory.shape == (2, 4, 32)
    torch.testing.assert_close(trajectory[:, 0], initial)
    torch.testing.assert_close(trajectory[:, 1], model(initial, actions[:, 0]))
    expected_second = model.forward_sequence(trajectory[:, :2], actions[:, :2])[:, -1]
    torch.testing.assert_close(trajectory[:, 2], expected_second)
    assert bool((trajectory.norm(dim=-1) * model.curvature.sqrt() < 1).all())
    target = model.encode_states(torch.randn(2, 32))
    distance(trajectory[:, -1], target, model.curvature).mean().backward()
    assert torch.isfinite(model.log_c.grad) and model.log_c.grad.abs() > 0
    assert model.action_encoder.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    torch.testing.assert_close(model.rollout(initial, actions[:, :0]), initial[:, None])


def test_activation_checkpointing_preserves_outputs_and_gradients():
    reference = make_predictor()
    checkpointed = make_predictor(activation_checkpointing=True)
    checkpointed.load_state_dict(reference.state_dict())
    features = torch.randn(2, 3, 32)
    actions = torch.tensor([[0, 1, 2], [2, 3, 4]])
    expected = reference.forward_sequence(reference.encode_states(features), actions)
    actual = checkpointed.forward_sequence(checkpointed.encode_states(features), actions)
    torch.testing.assert_close(expected, actual)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for left, right in zip(reference.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(left.grad, right.grad)


def test_world_model_connects_frozen_encoder_and_predictor():
    model = GeoWorldModel({
        "encoder": dict(name="tiny", pretrained=False, frames=4, image_size=32, tiny_dim=48, tiny_heads=4),
        "predictor": dict(hidden_dim=48, depth=2, num_heads=4, num_actions=7),
    })
    model.train()
    assert model.predictor.training and not model.encoder.training
    assert model.feature_dim == 48
    result = model(torch.rand(2, 3, 4, 32, 32), torch.tensor([1, 2]))
    result.square().sum().backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert model.predictor.predictor_proj.weight.grad is not None
