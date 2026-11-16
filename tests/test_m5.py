"""CPU synthetic checks for dense measurement and seed recovery. No downloads."""
from copy import deepcopy
import json

import pytest
import torch

from src.run_m3 import evaluate_sequence, forget_subsets
from src.run_m5 import measure_seed, run_m5, run_identity
from src.unlearn import run_sequence


def fixture(config):
    cfg = deepcopy(config)
    cfg["seeds"] = [0, 1]
    cfg["m5"]["tasks"] = 5
    cfg["projection"]["rank"] = 2
    cfg["unlearn"].update(epochs=2, batch_size=8)
    torch.manual_seed(7)
    head = {"weight": torch.randn(4, 6, dtype=torch.float64),
            "bias": torch.zeros(6, dtype=torch.float64)}
    cache = {"features": torch.randn(48, 4, dtype=torch.float64),
             "labels": torch.arange(48) % 6}
    return cfg, head, {"train": cache, "test": cache}


def test_dense_metrics_match_existing_evaluator(config):
    cfg, head, caches = fixture(config)
    rows = measure_seed(cfg, head, caches["train"], caches["test"], 0)
    classes = list(range(cfg["m5"]["tasks"]))
    result = run_sequence("intraining", forget_subsets(caches["train"], classes), head, cfg, 0)
    expected = evaluate_sequence("intraining", result, None, caches["test"], head,
                                 classes, [cfg["m5"]["lambda"]])
    assert [r["task"] for r in rows] == list(range(6))
    assert rows[0]["forget_all"] is None
    for actual, old in zip(rows[1:], expected):
        for key in ("retain_full", "retain_excluding_forgotten", "forget_all", "forget_current"):
            assert actual[key] == old[key]
    assert rows[2]["free_rank"] == 0
    assert all(r["weight_drift_since_exhaustion"] <= cfg["m5"]["weight_drift_atol"]
               for r in rows[2:])


def test_completed_seed_resume_and_identity_rejection(config, tmp_path, monkeypatch):
    cfg, head, caches = fixture(config)
    identity = run_identity(cfg, {"commit": "test"}, {"checkpoint": "test"})
    expected = run_m5(cfg, head, caches, identity, tmp_path)
    def no_training(*args, **kwargs):
        raise AssertionError("Completed seeds must skip training")
    monkeypatch.setattr("src.run_m5.measure_seed", no_training)
    assert run_m5(cfg, head, caches, identity, tmp_path) == expected
    changed = deepcopy(identity)
    changed["configuration"]["m5"]["lambda"] += 1
    with pytest.raises(ValueError, match="differs"):
        run_m5(cfg, head, caches, changed, tmp_path)
    path = tmp_path / "seeds" / "seed-0.json"
    saved = json.loads(path.read_text())
    saved["rows"][1]["retain_full"] += 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Invalid"):
        run_m5(cfg, head, caches, identity, tmp_path)


def test_interrupted_seed_replays_and_repairs_partial_progress(config, tmp_path):
    cfg, head, caches = fixture(config)
    identity = run_identity(cfg, {"commit": "test"}, {"checkpoint": "test"})
    def interrupt(folder):
        if (folder / "progress" / "seed-0" / "task-002.jsonl").exists():
            raise RuntimeError("simulated interruption")
    with pytest.raises(RuntimeError, match="simulated"):
        run_m5(cfg, head, caches, identity, tmp_path / "resumed", sync=interrupt)
    broken = tmp_path / "resumed" / "progress" / "seed-0" / "task-003.jsonl"
    broken.write_text('{"identity":')
    recovered = run_m5(cfg, head, caches, identity, tmp_path / "resumed")
    fresh = run_m5(cfg, head, caches, identity, tmp_path / "fresh")
    assert recovered == fresh
