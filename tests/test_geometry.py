import pytest
import torch

from geoworld.geometry import distance, expmap0, logmap0, project


@pytest.mark.parametrize("curvature", [0.01, 1.0, 3.0])
def test_exp_log_roundtrip_and_radial_distance(curvature):
    torch.manual_seed(1)
    tangent = torch.randn(7, 5, dtype=torch.float64) * 0.15
    ball = expmap0(tangent, curvature)
    torch.testing.assert_close(logmap0(ball, curvature), tangent, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(distance(torch.zeros_like(ball), ball, curvature),
                               2 * tangent.norm(dim=-1), atol=1e-12, rtol=1e-12)
    assert (curvature * ball.square().sum(-1) < 1).all()


def test_distance_matches_paper_acosh_formula():
    x = torch.tensor([[0.1, 0.2], [-0.3, 0.1]], dtype=torch.float64)
    y = torch.tensor([[-0.2, 0.1], [0.2, 0.25]], dtype=torch.float64)
    c = torch.tensor(0.7, dtype=torch.float64)
    argument = 1 + 2 * c * (x - y).square().sum(-1) / ((1 - c*x.square().sum(-1)) * (1 - c*y.square().sum(-1)))
    expected = torch.acosh(argument) / c.sqrt()
    torch.testing.assert_close(distance(x, y, c), expected, atol=1e-12, rtol=1e-12)


def test_geometry_has_finite_gradients_at_zero_and_identity():
    for dtype in (torch.float32, torch.float64):
        v = torch.zeros(3, 4, dtype=dtype, requires_grad=True)
        c = torch.tensor(0.8, dtype=dtype, requires_grad=True)
        point = expmap0(v, c)
        loss = point.sum() + logmap0(point, c).sum() + distance(point, point, c).sum()
        loss.backward()
        assert torch.isfinite(v.grad).all()
        assert torch.isfinite(c.grad)
        torch.testing.assert_close(distance(point, point, c), torch.zeros(3, dtype=dtype))


def test_differentiable_curvature_and_geometry_gradcheck():
    x = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([-0.2, 0.02, 0.15], dtype=torch.float64, requires_grad=True)
    c = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(expmap0, (x, c))
    assert torch.autograd.gradcheck(logmap0, (x, c))
    assert torch.autograd.gradcheck(distance, (x, y, c))
    distance(x, y, c).backward()
    assert c.grad.abs() > 0


def test_projection_near_boundary_and_triangle_inequality():
    torch.manual_seed(2)
    points = project(torch.randn(30, 3, dtype=torch.float64) * 5, 1.4)
    assert (1.4 * points.square().sum(-1) < 1).all()
    energy = distance(points, -points, 1.4)
    assert torch.isfinite(energy).all()
    a, b, c = expmap0(torch.randn(3, 40, 3, dtype=torch.float64) * .4)
    excess = distance(a, c) - distance(a, b) - distance(b, c)
    assert excess.max() <= 1e-10


@pytest.mark.parametrize("curvature", [0, -1, float("nan"), float("inf")])
def test_invalid_curvature_is_rejected(curvature):
    with pytest.raises(ValueError, match="finite positive scalar"):
        expmap0(torch.zeros(2, 3), curvature)
