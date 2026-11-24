"""CIFAR-100 development check immediately before any protected directions are released."""

import pandas as pd
import torch

from .common import digest
from .run_m3 import forget_subsets
from .run_m5 import run_seed_records
from .unlearn import fused_logits, run_sequence, to_projection_dtype


def development_split(cache, options, classes):
    """Reserve an equal number per class from the official training split."""
    labels = cache["labels"]
    generator = torch.Generator().manual_seed(options["split_seed"])
    take = options["dev_per_class"]
    if sorted(labels.unique().tolist()) != list(range(classes)):
        raise ValueError("Training cache must cover every class")
    dev_mask = torch.zeros_like(labels, dtype=torch.bool)
    for label in range(classes):
        indices = torch.where(labels == label)[0]
        if not 0 < take < len(indices):
            raise ValueError("Each class needs both fitting and development examples")
        selected = indices[torch.randperm(len(indices), generator=generator)[:take]]
        dev_mask[selected] = True

    def select(mask):
        return {key: value[mask] for key, value in cache.items()}

    index = torch.where(dev_mask)[0].tolist()
    manifest = {"source": "CIFAR-100 official train, original row order",
                "split_seed": options["split_seed"], "dev_per_class": take,
                "dev_index_sha256": digest(index), "dev_rows": len(index),
                "fit_pool_rows": int((~dev_mask).sum())}
    return select(~dev_mask), select(dev_mask), manifest


def run_start_check(cfg, raw_head, fit, development, identity, folder, sync=None):
    """Train once per seed. Record correct counts by class at both strengths per task."""
    options = cfg["stage_b_start"]
    tasks = options["tasks"]
    head = to_projection_dtype(raw_head, cfg)
    width, classes = head["weight"].shape
    if not 0 < tasks < options["fixed_retain_start"] < classes:
        raise ValueError("Earlier targets, next target and fixed retained classes must be disjoint")
    strengths = options["lambdas"]
    if len(strengths) != len(set(strengths)) or not strengths:
        raise ValueError("Fusion strengths must be nonempty and distinct")
    subsets = forget_subsets(to_projection_dtype(fit, cfg), list(range(tasks)))
    dev = to_projection_dtype(development, cfg)
    labels = dev["labels"]
    counts = torch.bincount(labels, minlength=classes).tolist()
    if counts != [options["dev_per_class"]] * classes:
        raise ValueError("Development set differs from the configured per-class sample count")
    torch.set_num_threads(cfg["runtime"]["threads"])

    def measure(seed, persist):
        rows = []

        @torch.inference_mode()
        def record(task, result):
            measurements = []
            for strength in strengths:
                logits = fused_logits(dev["features"], head, result["updates"], strength, task)
                correct = labels[logits.argmax(1) == labels]
                measurements.append({"lambda": strength,
                                     "correct": torch.bincount(correct, minlength=classes).tolist()})
            free_rank = result["free_rank"][-1] if task else width
            row = {"seed": seed, "task": task, "free_rank": free_rank,
                   "counts": counts, "measurements": measurements}
            persist(row)
            rows.append(row)

        record(0, {"updates": [], "free_rank": []})
        run_sequence("intraining", subsets, head, cfg, seed, on_task=record)
        return rows

    return run_seed_records(folder, identity, cfg["seeds"], tasks, measure, sync)


def summarize(rows, options):
    """Score each strength within seed using the same class IDs and images over time."""
    tasks = options["tasks"]
    history = []
    for row in rows:
        for measurement in row["measurements"]:
            for label, (correct, count) in enumerate(zip(measurement["correct"], row["counts"], strict=True)):
                history.append({"seed": row["seed"], "task": row["task"], "free_rank": row["free_rank"],
                                "lambda": measurement["lambda"], "class_id": label, "correct": correct,
                                "count": count, "accuracy": 100 * correct / count})
    history = pd.DataFrame(history)
    targets, scores = [], []
    for (seed, strength), group in history.groupby(["seed", "lambda"]):
        acc = group.pivot(index="task", columns="class_id", values="accuracy")
        ranks = group.groupby("task").free_rank.first()
        exhausted = ranks.index[ranks == 0].tolist()
        earlier = []
        for label in range(tasks):
            before, after, final = acc.at[label, label], acc.at[label+1, label], acc.at[tasks, label]
            earlier.append({"seed": seed, "lambda": strength, "class_id": label, "request": label + 1,
                            "before": before, "after": after, "at_boundary": final,
                            "drop_pp": before - after, "reversal_pp": max(0.0, final - after)})
        targets.extend(earlier)
        retained = group[group.class_id >= options["fixed_retain_start"]].groupby("task")[["correct", "count"]].sum()
        retain = 100 * retained.correct / retained["count"]
        loss = retain.at[0] - retain.at[tasks]
        worst_after = max(r["after"] for r in earlier)
        worst_final = max(r["at_boundary"] for r in earlier)
        rebound = max(r["reversal_pp"] for r in earlier)
        next_acc = acc.at[tasks, tasks]
        checks = {"exhausted_at_boundary": exhausted == [tasks],
                  "retain_ok": loss <= options["max_retain_loss_pp"],
                  "old_after_ok": worst_after <= options["max_old_accuracy_pct"],
                  "old_final_ok": worst_final <= options["max_old_accuracy_pct"],
                  "reversal_ok": rebound <= options["max_reversal_pp"],
                  "next_ok": next_acc >= options["min_next_accuracy_pct"]}
        scores.append({"seed": seed, "lambda": strength, "free_rank": ranks.at[tasks],
                       "retain_clean": retain.at[0], "retain_final": retain.at[tasks],
                       "retain_loss_pp": loss, "worst_after": worst_after, "worst_final": worst_final,
                       "worst_reversal_pp": rebound, "next_clean": acc.at[0, tasks],
                       "next_at_boundary": next_acc, **checks, "passes": all(checks.values())})
    return history, pd.DataFrame(targets), pd.DataFrame(scores)
