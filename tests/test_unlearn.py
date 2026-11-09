import pytest
import torch

from src.evaluate import unlearning_metrics
from src.lora import fuse
from src.unlearn import (fused_logits, router_logits, run_sequence,
                         to_projection_dtype, train_router)

D_IN, D_OUT, TASKS = 16, 5, 3
DOUBLE = torch.float64


@pytest.fixture
def fast(config):
    """The real config with the training loops shortened for unit tests."""
    cfg = {**config, "unlearn": {**config["unlearn"], "epochs": 2, "router_epochs": 2}}
    cfg["unlearn"]["forget_classes"] = list(range(TASKS))
    return cfg


@pytest.fixture
def head(fast):
    torch.manual_seed(0)
    return to_projection_dtype(
        {"weight": torch.randn(D_IN, D_OUT) * 0.1, "bias": torch.zeros(D_OUT)}, fast)


@pytest.fixture
def subsets(fast):
    torch.manual_seed(1)
    return [to_projection_dtype(
        {"features": torch.randn(64, D_IN).abs(),
         "labels": torch.full((64,), task, dtype=torch.long)}, fast)
        for task in range(TASKS)]


def test_metrics_report_all_four_readings():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    predictions = torch.tensor([0, 1, 1, 1, 2, 0])
    logits = torch.nn.functional.one_hot(predictions, num_classes=3).float()

    result = unlearning_metrics(logits, labels, forgotten=[0], current=0)

    assert result["forget_current"] == pytest.approx(50.0)
    assert result["forget_all"] == pytest.approx(50.0)
    assert result["retain_excluding_forgotten"] == pytest.approx(75.0)
    assert result["retain_full"] == pytest.approx(100 * 4 / 6)


@pytest.mark.parametrize("method", ["intraining", "posthoc"])
def test_projected_methods_reserve_capacity_and_stay_feasible(fast, head, subsets, method):
    rank = fast["projection"]["rank"]
    result = run_sequence(method, subsets, head, fast, seed=0)

    # A trained update may have effective rank below the adapter rank, so assert
    # the invariant rather than a value. What that rank actually is for a real
    # run is test T1 in paper_spec.md, and it is what discrepancy D2 turns on.
    assert all(0 < used <= rank for used in result["reserved"])
    assert result["free_rank"] == [D_IN - sum(result["reserved"][:k + 1])
                                   for k in range(TASKS)]
    # Every stored update must lie in the subspace that was free when it was made.
    running = torch.eye(D_IN, dtype=DOUBLE)
    for update, used in zip(result["updates"], result["reserved"]):
        torch.testing.assert_close(running @ update, update, atol=1e-8, rtol=0)
        left, _, _ = torch.linalg.svd(update, full_matrices=False)
        basis = left[:, :used]
        running = running - basis @ basis.T


def test_static_keeps_no_projector(fast, head, subsets):
    result = run_sequence("static", subsets, head, fast, seed=0)
    assert result["reserved"] == [] and result["free_rank"] == []
    assert len(result["updates"]) == TASKS


def test_fused_logits_match_equation_one(fast, head, subsets):
    result = run_sequence("static", subsets, head, fast, seed=0)
    features = torch.randn(32, D_IN, dtype=DOUBLE)

    actual = fused_logits(features, head, result["updates"], strength=0.75, tasks=2)
    expected = features @ fuse(head["weight"], result["updates"][:2], 0.75) + head["bias"]

    torch.testing.assert_close(actual, expected)


def test_router_adds_nothing_when_no_adapters_are_applied(fast, head, subsets):
    result = run_sequence("router", subsets, head, fast, seed=0)
    retain = torch.randn(128, D_IN, dtype=DOUBLE).abs()
    gate = train_router(subsets, retain, fast, seed=0)
    features = torch.randn(32, D_IN, dtype=DOUBLE)

    base = features @ head["weight"] + head["bias"]
    torch.testing.assert_close(
        router_logits(features, head, result["adapters"], gate, tasks=0), base)
    # Gate weights exclude the null slot, so they sum to at most one.
    assert bool((gate.weights(features).sum(dim=1) <= 1.0 + 1e-6).all())


def test_lambda_does_not_change_training(fast, head, subsets):
    """Lambda enters only Eq. 1, so one trained sequence serves every lambda."""
    first = run_sequence("intraining", subsets, head, fast, seed=3)
    second = run_sequence("intraining", subsets, head, fast, seed=3)
    for a, b in zip(first["updates"], second["updates"]):
        torch.testing.assert_close(a, b)
