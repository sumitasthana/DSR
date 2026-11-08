import pytest
import torch

from src.diagnostics import budget_trace, interference


@pytest.mark.parametrize("rank", [2, 4, 8])
def test_budget_exhausts_at_width_over_rank(config, rank):
    """T4 and T2 arithmetic: capacity runs out at floor(d_in / rank)."""
    budget = 64 // rank
    rows = budget_trace(64, 100, rank, budget + 4, config["projection"], seed=0)

    for row in rows[:budget]:
        assert row["reserved"] == rank
        assert row["free_rank_after"] == 64 - rank * row["task"]
        assert row["update_norm"] > 0.0

    # Past the budget the projector is zero, so every further update is zero.
    for row in rows[budget:]:
        assert row["free_rank_before"] == 0
        assert row["reserved"] == 0
        assert row["update_norm"] == pytest.approx(0.0, abs=1e-10)


def test_free_rank_curve_is_independent_of_scale(config):
    """T7: scaling an update cannot change its column space, so lambda cannot
    move the saturation curve even if it entered training."""
    curves = {
        scale: [row["free_rank_after"]
                for row in budget_trace(64, 100, 4, 20, config["projection"], seed=0, scale=scale)]
        for scale in (0.5, 1.0, 2.0)
    }
    assert curves[0.5] == curves[1.0] == curves[2.0]


def test_orthogonal_input_subspaces_still_change_predictions(config):
    """T5: the geometric claim holds but the behavioural claim does not."""
    torch.manual_seed(0)
    weight = torch.randn(64, 100, dtype=torch.float64) * 0.1
    features = torch.randn(512, 64, dtype=torch.float64)

    result = interference(weight, features, 4, config["projection"], seed=0, strength=1.0)

    # The paper's geometry is correct: the two updates are orthogonal.
    assert result["input_subspace_overlap"] == pytest.approx(0.0, abs=1e-8)
    assert result["frobenius_inner_product"] == pytest.approx(0.0, abs=1e-8)
    # Their behaviour is not independent: adding the second update moves logits
    # and flips predictions the first update had already set.
    assert result["mean_logit_shift"] > 0.0
    assert result["argmax_change_fraction"] > 0.0
