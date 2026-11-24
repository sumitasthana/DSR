import torch

from src.stage_b_start import development_split, run_start_check, summarize


def test_same_split_across_seeds_and_paired_measurements(config, tmp_path, monkeypatch):
    opts = config["stage_b_start"]
    opts.update(tasks=2, fixed_retain_start=3, dev_per_class=3)
    config["seeds"] = [0, 1]
    config["unlearn"]["epochs"] = 2
    config["projection"]["rank"] = 2
    rng = torch.Generator().manual_seed(41)
    cache = {"labels": torch.arange(60) % 5, "features": torch.randn(60, 4, generator=rng),
             "ids": torch.arange(60)}
    fit, dev, manifest = development_split(cache, opts, 5)
    assert len(set(dev["ids"].tolist()) & set(fit["ids"].tolist())) == 0
    assert len(dev["ids"]) + len(fit["ids"]) == 60
    assert development_split(cache, opts, 5)[2] == manifest
    head = {"weight": torch.randn(4, 5, generator=rng), "bias": torch.zeros(5)}
    identity = {"manifest": manifest, "config": config}
    rows = run_start_check(config, head, fit, dev, identity, tmp_path)
    history, targets, scores = summarize(rows, opts)
    assert len(history) == 2 * 2 * 3 * 5
    assert len(targets) == 2 * 2 * 2
    assert scores.exhausted_at_boundary.all()
    clean = (dev["features"].double() @ head["weight"].double()).argmax(1) == dev["labels"]
    expected = 100 * clean[dev["labels"] >= 3].sum().item() / 6
    assert (scores.retain_clean == expected).all()
    assert (scores.next_clean == 100 * clean[dev["labels"] == 2].sum().item() / 3).all()
    def unexpected_training(*args, **kwargs):
        raise AssertionError("a completed seed was retrained")
    monkeypatch.setattr("src.stage_b_start.run_sequence", unexpected_training)
    assert run_start_check(config, head, fit, dev, identity, tmp_path) == rows


def test_one_old_class_can_fail_even_when_retention_and_next_target_pass(config):
    # Targets 0 and 1. Class 0 rebounds by 25 points. Class 2 is next, 3 and 4 stay fixed.
    by_task = [[4, 3, 4, 2, 4], [0, 2, 4, 2, 4], [1, 0, 3, 2, 4]]
    rows = [{"seed": 0, "task": t, "free_rank": 4-2*t, "counts": [4]*5,
             "measurements": [{"lambda": 0.5, "correct": c}]} for t, c in enumerate(by_task)]
    opts = {**config["stage_b_start"], "tasks": 2, "fixed_retain_start": 3}
    _, targets, scores = summarize(rows, opts)
    assert list(targets.before) == [100, 50]
    assert list(targets.after) == [0, 0]
    assert list(targets.at_boundary) == [25, 0]
    s = scores.iloc[0]
    assert s.retain_clean == s.retain_final == s.next_at_boundary == 75
    assert s.old_after_ok and s.next_ok and s.retain_ok and s.exhausted_at_boundary
    assert not s.old_final_ok and not s.reversal_ok and not s.passes
