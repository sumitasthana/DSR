"""Checks on the M4 sweep logic, with no training and no dataset.

The one that matters is the prefix claim. M4 trains a single 30-task sequence
and reads it at N = 5, 10, 20 and 30, which is only sound if the first k tasks
of a long sequence are exactly the tasks a k-task run would have trained.
"""

import math

import pytest
import torch

from src.run_m3 import evaluate_sequence
from src.run_m4 import rank_sweep
from src.unlearn import run_sequence, to_projection_dtype


CFG = {
    "projection": {"rank": 2, "dtype": "float64", "rank_atol": 1e-12,
                   "rank_rtol": 1e-8, "feasibility_atol": 1e-8,
                   "feasibility_rtol": 1e-6},
    "lora": {"init_std": 0.01},
    "unlearn": {"epochs": 2, "lr": 0.05, "batch_size": 8},
    "m4": {"rank_sweep": [2, 4], "rank_sweep_margin": 1},
}


def _head(d_in=8, d_out=8):
    torch.manual_seed(0)
    return to_projection_dtype(
        {"weight": torch.randn(d_in, d_out), "bias": torch.zeros(d_out)}, CFG)


def _subsets(count, d_in=8, rows=16):
    torch.manual_seed(1)
    return [to_projection_dtype(
        {"features": torch.randn(rows, d_in),
         "labels": torch.full((rows,), task % 8, dtype=torch.long)}, CFG)
        for task in range(count)]


@pytest.mark.parametrize("method", ["static", "posthoc", "intraining"])
def test_long_sequence_prefix_equals_a_short_run(method):
    """Reading task 3 off a 5-task run must equal running 3 tasks."""
    head, subsets = _head(), _subsets(5)
    long_run = run_sequence(method, subsets, head, CFG, seed=3)
    short_run = run_sequence(method, subsets[:3], head, CFG, seed=3)
    for task in range(3):
        torch.testing.assert_close(long_run["updates"][task], short_run["updates"][task])


def test_metrics_at_a_checkpoint_match_a_dedicated_run():
    """The same claim, carried all the way through to the reported numbers."""
    head, subsets = _head(), _subsets(5)
    test = to_projection_dtype(
        {"features": torch.randn(32, 8), "labels": torch.arange(32) % 8}, CFG)
    classes = [0, 1, 2, 3, 4]   # of 8, so some always remain

    long_rows = evaluate_sequence("intraining", run_sequence("intraining", subsets, head, CFG, 3),
                                  None, test, head, classes, [0.5, 1.0])
    short_rows = evaluate_sequence("intraining",
                                   run_sequence("intraining", subsets[:3], head, CFG, 3),
                                   None, test, head, classes[:3], [0.5, 1.0])
    at_three = [r for r in long_rows if r["tasks"] == 3]
    assert at_three == [r for r in short_rows if r["tasks"] == 3]


def test_rank_sweep_exhausts_at_the_predicted_budget():
    """T2 in miniature: d_in = 8, so rank 2 must run out at task 4 and rank 4 at 2."""
    head = _head()
    rows = rank_sweep(head, _subsets(6), CFG, seed=5)
    for rank, expected in ((2, 4), (4, 2)):
        part = [r for r in rows if r["rank"] == rank]
        assert part[0]["predicted_budget"] == expected
        assert next(r["task"] for r in part if r["free_rank_after"] == 0) == expected
        # Past the budget there is nothing left to read, so the update is zero.
        assert all(math.isclose(r["update_norm"], 0.0, abs_tol=1e-20)
                   for r in part if r["task"] > expected)


def test_rank_sweep_refuses_to_report_a_budget_it_cannot_reach():
    with pytest.raises(ValueError, match="to show exhaustion"):
        rank_sweep(_head(), _subsets(2), CFG, seed=5)
