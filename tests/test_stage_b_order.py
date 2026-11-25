import pandas as pd
import pytest
import torch

from src import unlearn
from src.stage_b_order import measure_orders, paired_results, request_orders
from src.stage_b_start import development_split


def test_pair_changes_order_but_preserves_per_class_rng_and_measurement(config, tmp_path, monkeypatch):
    config["stage_b_start"].update(tasks=4, fixed_retain_start=5, dev_per_class=4)
    config["stage_b_order"]["swap_count"] = 1
    config["projection"]["rank"] = 1
    config["unlearn"]["epochs"] = 2
    rng = torch.Generator().manual_seed(7)
    cache = {"features": torch.randn(96, 4, generator=rng), "labels": torch.arange(96) % 6}
    head = {"weight": torch.randn(4, 6, generator=rng), "bias": torch.zeros(6)}
    fit, dev, manifest = development_split(cache, config["stage_b_start"], 6)
    calls = []
    train = unlearn.train_adapter

    def trace(subset, weight, bias, projector, cfg, seed):
        calls.append((int(subset["labels"][0]), seed))
        return train(subset, weight, bias, projector, cfg, seed)

    monkeypatch.setattr(unlearn, "train_adapter", trace)
    args = config, head, fit, dev, {"split": manifest, "configuration": config}, tmp_path
    history, targets, scores = measure_orders(*args)
    orders = request_orders(4, 1)
    assert calls == [(c, seed*100+c) for order in orders.values() for seed in config["seeds"] for c in order]
    assert scores.exhausted_at_boundary.all()
    lookup = history.set_index(["order", "seed", "task", "class_id"]).accuracy
    for row in targets.itertuples():
        assert orders[row.order][row.request-1] == row.class_id
        assert row.before == lookup[row.order, row.seed, row.request-1, row.class_id]
        assert row.after == lookup[row.order, row.seed, row.request, row.class_id]
    pairs, _, _ = paired_results(targets, config)
    assert pairs.request_early.eq(1).all() and pairs.request_late.eq(4).all()
    calls.clear()
    measure_orders(*args)
    assert calls == []


def test_judges_gaps_within_class_and_seed(config):
    rows = []
    opts = config["stage_b_start"]
    for arm, order in request_orders(opts["tasks"], 2).items():
        for seed in config["seeds"]:
            for request, label in enumerate(order, 1):
                after = 30 if request <= 2 else 50
                rows.append(dict(order=arm, seed=seed, class_id=label, request=request, before=90,
                                 after=after, drop_pp=90-after, at_boundary=after, reversal_pp=0))
    targets = pd.DataFrame(rows)
    pairs, _, decision = paired_results(targets, config)
    assert len(pairs) == 12 and pairs.late_minus_early_pp.eq(20).all()
    assert all(decision.values())  # Difficulty in both positions can coexist with an order effect.
    targets.loc[(targets.seed == 1) & (targets.class_id == 14) & (targets.order == "original"), "after"] = 32
    _, _, mixed = paired_results(targets, config)
    assert not mixed["all_moved_classes_show_late_penalty"]
    with pytest.raises(ValueError, match="Missing"):
        paired_results(targets.iloc[1:], config)
