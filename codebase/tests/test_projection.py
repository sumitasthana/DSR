import pytest
import torch

from src.projection import SubspaceProjector


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recursive_projection_and_exhaustion(config, seed):
    torch.manual_seed(seed)
    tracker = SubspaceProjector(64, config["projection"])
    for task in range(16):
        old = tracker.matrix
        update = old @ torch.randn(64, 4, dtype=torch.float64) @ torch.randn(4, 100, dtype=torch.float64)
        u, _, _ = torch.linalg.svd(update, full_matrices=False)
        assert tracker.reserve(update) == 4
        p = tracker.matrix
        torch.testing.assert_close(p, old - u[:, :4] @ u[:, :4].T, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(p, p.T)
        torch.testing.assert_close(p @ p, p, atol=1e-8, rtol=1e-8)
        assert tracker.free_rank == 64 - 4 * (task + 1)
        assert int(torch.linalg.matrix_rank(p, atol=1e-8, rtol=0)) == tracker.free_rank
        torch.testing.assert_close(p @ update, torch.zeros_like(update), atol=1e-8, rtol=0)
    # A numerical residual after exhaustion must not reserve arbitrary SVD vectors.
    residual = tracker.matrix @ torch.randn(64, 4, dtype=torch.float64) @ torch.randn(4, 100, dtype=torch.float64)
    assert tracker.reserve(residual) == 0
    assert tracker.free_rank == 0


def test_zero_and_rank_deficient_updates(config):
    tracker = SubspaceProjector(8, config["projection"])
    assert tracker.reserve(torch.zeros(8, 5)) == 0
    update = torch.zeros(8, 5, dtype=torch.float64)
    update[0, 0] = 2
    assert tracker.reserve(update) == 1
    assert tracker.free_rank == 7
    with pytest.raises(ValueError, match="outside"):
        tracker.reserve(update)


def test_posthoc_projection_and_nonfinite_rejection(config):
    tracker = SubspaceProjector(8, config["projection"])
    prior = torch.zeros(8, 5, dtype=torch.float64)
    prior[0, 0] = 1
    tracker.reserve(prior)
    a = torch.arange(8, dtype=torch.float64).reshape(8, 1) + 1
    b = torch.ones(1, 5, dtype=torch.float64)
    assert tracker.reserve((tracker.matrix @ a) @ b) == 1
    with pytest.raises(ValueError, match="finite"):
        tracker.reserve(torch.full((8, 5), float("nan")))
