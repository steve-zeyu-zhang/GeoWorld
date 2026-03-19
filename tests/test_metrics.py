import pytest
import torch

from geoworld.metrics import planning_metrics


def test_sequence_success_position_accuracy_and_set_overlap_are_distinct():
    # Reordering changes SR/mAcc, but not the unordered procedure label set.
    predicted = torch.tensor([[0, 1, 2], [1, 2, 0], [0, 0, 2]])
    targets = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    result = planning_metrics(predicted, targets)
    assert result["SR"] == pytest.approx(100 / 3)
    assert result["mAcc"] == pytest.approx(100 * 5 / 9)
    assert result["mIoU"] == pytest.approx(100 * 8 / 9)


def test_intersection_over_union_uses_unique_action_labels():
    # A repeated label counts once in intersection/union, regardless of position.
    result = planning_metrics([[0, 0, 1, 1]], [[0, 2, 2, 2]])
    assert result["SR"] == 0
    assert result["mAcc"] == 25
    assert result["mIoU"] == pytest.approx(100 / 3)


def test_masked_accuracy_averages_trajectories_before_batch():
    predicted = torch.tensor([[1, 8, 9], [0, 1, 2]])
    targets = torch.tensor([[1, -1, -1], [0, 0, 0]])
    mask = torch.tensor([[True, False, False], [True, True, True]])
    result = planning_metrics(predicted, targets, mask)
    assert result["SR"] == 50
    assert result["mAcc"] == pytest.approx(100 * 2 / 3)
    assert result["mIoU"] == pytest.approx(100 * 2 / 3)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_empty_batches_and_empty_sequences_are_rejected(shape):
    with pytest.raises(ValueError):
        planning_metrics(torch.empty(shape, dtype=torch.long), torch.empty(shape, dtype=torch.long))


def test_mismatched_shapes_and_fully_padded_examples_are_rejected():
    with pytest.raises(ValueError, match="identical"):
        planning_metrics([[0, 1]], [[0]])
    with pytest.raises(ValueError, match="valid"):
        planning_metrics([[0, 1], [1, 2]], [[0, 1], [-1, -1]], [[True, True], [False, False]])
