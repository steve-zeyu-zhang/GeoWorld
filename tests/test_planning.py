import pytest
import torch
from torch import nn

from geoworld.geometry import distance, expmap0, logmap0
from geoworld.planning import CategoricalCEMPlanner


class OrderedActionWorld(nn.Module):
    """Different action orders produce distinct terminal states."""
    def __init__(self):
        super().__init__()
        self.curvature = torch.tensor(0.5, dtype=torch.float64)

    def rollout(self, initial, actions):
        states = [initial]
        for action in actions.unbind(1):
            tangent = logmap0(states[-1], self.curvature) * .4
            tangent = tangent + torch.nn.functional.one_hot(action, 3).double() * .2
            states.append(expmap0(tangent, self.curvature))
        return torch.stack(states, 1)


def test_cem_finds_known_discrete_plan_and_is_reproducible():
    world = OrderedActionWorld().eval()
    start = torch.zeros(2, 3, dtype=torch.float64)
    truth = torch.tensor([[0, 2, 1], [2, 1, 0]])
    goal = world.rollout(start, truth)[:, -1]
    planner = CategoricalCEMPlanner(3, 3, num_samples=128, num_elites=8, iterations=5, evaluation_batch_size=29)
    result = planner.plan(world, start, goal, generator=torch.Generator().manual_seed(2))
    repeated = planner.plan(world, start, goal, generator=torch.Generator().manual_seed(2))
    assert torch.equal(result["actions"], truth)
    assert torch.equal(result["actions"], repeated["actions"])
    assert result["energy"].max() < 1e-12
    torch.testing.assert_close(result["probabilities"].sum(-1), torch.ones(2, 3, dtype=torch.float64))
    assert (result["energy_history"][:, 1:] <= result["energy_history"][:, :-1]).all()
    assert result["trajectory"].shape == (2, 4, 3)


@pytest.mark.parametrize("per_step", [False, True])
def test_optional_action_masks_exclude_illegal_labels(per_step):
    world = OrderedActionWorld()
    start = torch.zeros(1, 3, dtype=torch.float64)
    desired = torch.tensor([[2, 2, 2]])
    goal = world.rollout(start, desired)[:, -1]
    allowed = torch.tensor([True, True, False])
    if per_step:
        allowed = allowed[None,None].expand(1,3,3).clone()
        allowed[:,1,0] = False
    planner = CategoricalCEMPlanner(3, 3, num_samples=64, num_elites=8, iterations=3)
    result = planner.plan(world, start, goal, allowed, torch.Generator().manual_seed(9))
    assert not (result["actions"] == 2).any()
    assert not result["probabilities"][...,2].any()
    if per_step:
        assert result["actions"][0,1] == 1


def test_default_search_uses_whole_fixed_label_vocabulary():
    world = OrderedActionWorld()
    start = torch.zeros(1,3,dtype=torch.float64)
    goal = world.rollout(start, torch.tensor([[2]]))[:,-1]
    result = CategoricalCEMPlanner(3,1,num_samples=32,num_elites=4,iterations=2).plan(
        world,start,goal,generator=torch.Generator().manual_seed(10))
    assert result["actions"].item() == 2
    assert (result["probabilities"] > 0).all()


def test_planning_rejects_empty_action_spaces_and_invalid_hyperparameters():
    with pytest.raises(ValueError,match="positive integers"):
        CategoricalCEMPlanner(3,0)
    with pytest.raises(ValueError,match="cannot exceed"):
        CategoricalCEMPlanner(3,2,num_samples=4,num_elites=8)
    planner = CategoricalCEMPlanner(3,1,num_samples=4,num_elites=2,iterations=1)
    with pytest.raises(ValueError,match="at least one permitted"):
        planner.plan(OrderedActionWorld(),torch.zeros(1,3,dtype=torch.float64),
                     torch.zeros(1,3,dtype=torch.float64),torch.zeros(3,dtype=torch.bool))
