"""M4: CIFAR-100 across N in {5, 10, 20, 30}: python -m src.run_m4

This is the run that crosses the capacity budget. M3 measured that every trained
adapter consumes exactly 4 of the 64 input directions, so the 17th deletion has
nothing left to read and its correction must be exactly zero. The paper reports
results at N = 20 and N = 30. This run says what is actually happening there.

Two shortcuts that are not approximations:

One sequence covers every N. Deletions are sequential and causal, so the first k
tasks of a 30-task run are the same tasks a k-task run would train. We train 30
and read the metrics at 5, 10, 20 and 30. The router is the one exception: its
gate is trained over all N adapters at once, so it gets a separate gate per N.

Lambda costs nothing. It enters only at fusion time (Eq. 1), never during
training, so one trained sequence is evaluated at every lambda.

Test T2 needs a different shape and runs separately at the bottom: rank 2, 4 and
8 each taken past their own predicted budget of 32, 16 and 8 tasks.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from .common import load_config, provenance, save_json
from .evaluate import accuracy
from .run_m0 import baseline_caches
from .run_m3 import (evaluate_sequence, forget_subsets, fusion_norms,
                     retain_features, subspace_rows)
from .unlearn import (METHODS, fused_logits, run_sequence, to_projection_dtype,
                      train_router)


def rank_sweep(head, subsets, cfg, seed):
    """T2: does the free space run out exactly where floor(d_in / r) says?

    Each rank gets its own sequence, run `rank_sweep_margin` tasks past its own
    predicted budget so the collapse is visible rather than merely reached.
    """
    d_in = head["weight"].shape[0]
    rows = []
    for rank in cfg["m4"]["rank_sweep"]:
        budget = d_in // rank
        tasks = min(budget + cfg["m4"]["rank_sweep_margin"], len(subsets))
        if tasks <= budget:
            raise ValueError(
                f"rank {rank} needs {budget + 1} tasks to show exhaustion, "
                f"only {len(subsets)} classes are configured")
        local = deepcopy(cfg)
        local["projection"]["rank"] = rank
        result = run_sequence("intraining", subsets[:tasks], head, local, seed)
        for task in range(tasks):
            rows.append({
                "rank": rank, "predicted_budget": budget, "task": task + 1,
                "reserved_rank": result["reserved"][task],
                "free_rank_after": result["free_rank"][task],
                "update_norm": float(torch.linalg.norm(result["updates"][task])),
            })
        print(f"  rank {rank}: budget {budget}, ran {tasks} tasks, "
              f"free rank ends at {result['free_rank'][-1]}", flush=True)
    return rows


@torch.inference_mode()
def silent_failures(result, test_cache, head, classes, budget, strength):
    """What happened to each class, split by whether its deletion fit the budget.

    Past task `budget` the projection has no free directions left, so the update
    is exactly zero and the deletion request does nothing at all. The caller gets
    back per-class accuracy before and after, which is the difference between a
    deletion that worked and one that was silently ignored.
    """
    logits = fused_logits(test_cache["features"], head, result["updates"], strength, len(classes))
    after, before = logits.argmax(dim=1), (
        test_cache["features"] @ head["weight"] + head["bias"]).argmax(dim=1)
    labels = test_cache["labels"]

    rows = []
    for position, label in enumerate(classes):
        mask = labels == label
        rows.append({
            "task": position + 1, "forget_class": label,
            "within_budget": position < budget,
            "accuracy_before": 100.0 * float((before[mask] == label).double().mean()),
            "accuracy_after": 100.0 * float((after[mask] == label).double().mean()),
        })
    # Exactly zero, not merely small, is the claim worth stating.
    inside = sum(u for u in result["updates"][:budget])
    everything = sum(u for u in result["updates"])
    return rows, float((everything - inside).abs().max())


def run_m4(config, seeds=None):
    """Train one 30-task sequence per method and seed, read at four values of N."""
    cfg = deepcopy(config)
    torch.set_num_threads(cfg["runtime"]["threads"])
    context = provenance()
    seeds = seeds or cfg["seeds"]
    options = cfg["m4"]
    classes = list(range(options["tasks"]))
    checkpoints = options["checkpoints"]
    lambdas = cfg["unlearn"]["lambdas"]
    if max(checkpoints) > len(classes):
        raise ValueError("A checkpoint asks for more deletions than there are tasks")

    # The rank sweep needs its own length: rank 2 has a 32-task budget, which is
    # longer than the 30-task sequence the rest of the run uses. Prepare enough
    # classes for whichever is larger.
    d_in = cfg["model"]["feature_dim"]
    sweep_tasks = max(d_in // rank + options["rank_sweep_margin"]
                      for rank in options["rank_sweep"])
    prepared = list(range(max(options["tasks"], sweep_tasks)))
    if len(prepared) > cfg["model"]["classes"]:
        raise ValueError(f"Need {len(prepared)} classes, the dataset has "
                         f"{cfg['model']['classes']}")

    folder, identity, raw_head, raw_caches = baseline_caches(cfg, context)
    head = to_projection_dtype(raw_head, cfg)
    train_cache = to_projection_dtype(raw_caches["train"], cfg)
    test_cache = to_projection_dtype(raw_caches["test"], cfg)
    clean = accuracy(test_cache, head)["accuracy_percent"]
    budget = cfg["model"]["feature_dim"] // cfg["projection"]["rank"]
    print(f"baseline before any deletion: {clean:.2f}%")
    print(f"deleting classes {classes[0]} to {classes[-1]} in order, "
          f"budget is {budget} tasks, checkpoints at {checkpoints}")

    prepared_subsets = forget_subsets(train_cache, prepared)
    subsets = prepared_subsets[:len(classes)]

    # Runs first: it is quick, it is the T2 answer, and a configuration mistake
    # here should fail in seconds rather than after every method has trained.
    print("rank sweep (T2), one seed:", flush=True)
    sweep = rank_sweep(head, prepared_subsets, cfg, seeds[0])

    metrics, subspace, norms, per_class, drift = [], [], [], [], None
    for seed in seeds:
        # One gate per N: a gate trained over 30 adapters is not the gate a
        # 5-adapter run would have produced, so the router cannot share.
        gates = {n: train_router(subsets[:n], retain_features(train_cache, classes[:n]), cfg, seed)
                 for n in checkpoints}
        print(f"seed {seed}: {len(gates)} routers trained", flush=True)

        for method in METHODS:
            result = run_sequence(method, subsets, head, cfg, seed)
            if result["reserved"]:
                subspace += [{"seed": seed, **row}
                             for row in subspace_rows(method, result, classes)]
            if method == "intraining":
                norms += [{"seed": seed, **row}
                          for row in fusion_norms(result, lambdas, classes)]
            if method == "router":
                # The gate differs per N, so the router is evaluated once per
                # checkpoint with the gate that was trained for it.
                for n in checkpoints:
                    prefix = {"updates": result["updates"][:n],
                              "adapters": result["adapters"][:n]}
                    metrics += [{"seed": seed, "n": n, **row}
                                for row in evaluate_sequence(method, prefix, gates[n], test_cache,
                                                             head, classes[:n], lambdas)
                                if row["tasks"] == n]
            else:
                # Every other method reads straight off the one sequence: the
                # row at task k is exactly what a k-task run would have produced.
                metrics += [{"seed": seed, "n": row["tasks"], **row}
                            for row in evaluate_sequence(method, result, None, test_cache,
                                                         head, classes, lambdas)
                            if row["tasks"] in checkpoints]
            if method == "intraining" and seed == seeds[0]:
                per_class, drift = silent_failures(result, test_cache, head, classes,
                                                   budget, strength=1.0)
            print(f"seed {seed} {method:11s} done", flush=True)

    return {
        "status": "M4_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity, "environment": context,
        "classes": classes, "seeds": seeds, "lambdas": lambdas,
        "checkpoints": checkpoints, "rank": cfg["projection"]["rank"], "capacity": budget,
        "clean_accuracy_percent": clean,
        "metrics": metrics, "subspace": subspace, "fusion_norms": norms,
        "rank_sweep": sweep, "per_class": per_class, "update_drift_past_budget": drift,
        "limitations": [
            "Public checkpoint at 68.83% clean, not the paper's 60.39% starting point.",
            "Learning rate, epochs, optimiser and batch size for unlearning are ours.",
            "The deleted classes are the first 30 by index; the paper names none.",
            "The rank sweep runs one seed, because it measures a deterministic "
            "property of the projection rather than a trained outcome.",
        ],
    }


def summarise(report):
    """Print the three pre-registered questions and what the run says about them."""
    budget = report["capacity"]
    subspace = report["subspace"]

    print()
    print("T2: does the free space run out where the budget says?")
    for r in sorted({s["rank"] for s in report["rank_sweep"]}):
        rows = [s for s in report["rank_sweep"] if s["rank"] == r]
        predicted = rows[0]["predicted_budget"]
        zero_at = next((s["task"] for s in rows if s["free_rank_after"] == 0), None)
        after = [s["update_norm"] for s in rows if s["task"] > predicted]
        print(f"  rank {r}: predicted {predicted}, free rank hit 0 at task {zero_at}, "
              f"norms past the budget {['%.2e' % v for v in after]}")

    print()
    print(f"T3: does the correction stop growing at task {budget}?")
    for strength in (1.0,):
        rows = [n for n in report["fusion_norms"]
                if n["seed"] == report["seeds"][0] and n["lambda"] == strength]
        marks = [budget - 1, budget, budget + 1, len(report["classes"])]
        shown = {r["task"]: r["fused_delta_norm"] for r in rows if r["task"] in marks}
        print(f"  lambda {strength}: " + ", ".join(f"task {k} -> {v:.3f}" for k, v in sorted(shown.items())))

    print()
    print(f"D3: are the projected methods frozen past task {budget}?")
    for method in ("intraining", "posthoc"):
        rows = [s for s in subspace if s["method"] == method and s["seed"] == report["seeds"][0]]
        past = [s["update_norm"] for s in rows if s["task"] > budget]
        if not past:
            print(f"  {method:11s} not reached: the run stops at task {len(rows)}")
            continue
        print(f"  {method:11s} update norms past the budget: "
              f"max {max(past):.3e} over {len(past)} tasks")

    print()
    print(f"How many deletions actually did anything? (in-training, lambda 1.0, seed "
          f"{report['seeds'][0]})")
    inside = [r for r in report["per_class"] if r["within_budget"]]
    outside = [r for r in report["per_class"] if not r["within_budget"]]
    print(f"  the fused weight at task {len(report['classes'])} differs from task {budget} by "
          f"{report['update_drift_past_budget']}")
    for name, rows in (("within the budget", inside), ("past the budget", outside)):
        if not rows:
            continue
        print(f"  {name:18s} {len(rows):2d} classes: "
              f"{np.mean([r['accuracy_before'] for r in rows]):5.2f}% before -> "
              f"{np.mean([r['accuracy_after'] for r in rows]):5.2f}% after")
    if outside:
        print(f"  {len(outside)} deletion requests changed the model by exactly nothing.")

    print()
    print("Table I analogue, mean over seeds, retain excluding forgotten / forget all")
    for n in report["checkpoints"]:
        print(f"  N = {n}")
        for method in METHODS:
            for strength in (0.5, 1.0):
                rows = [m for m in report["metrics"]
                        if m["n"] == n and m["method"] == method and m["lambda"] == strength]
                if not rows:
                    continue
                print(f"    {method:11s} lambda {strength}: "
                      f"retain {np.mean([r['retain_excluding_forgotten'] for r in rows]):6.2f}  "
                      f"forget {np.mean([r['forget_all'] for r in rows]):6.2f}")


def write_outputs(report, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "report.json", report)
    for name in ("metrics", "subspace", "fusion_norms", "rank_sweep", "per_class"):
        rows = report[name]
        if not rows:
            continue
        columns = list(rows[0])
        (folder / f"{name}.csv").write_text(
            ",".join(columns) + "\n"
            + "".join(",".join(str(row[c]) for c in columns) + "\n" for row in rows),
            encoding="utf-8")
    print(f"\nwrote {folder / 'report.json'} and {len(report['metrics'])} metric rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "config.yaml")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = run_m4(cfg)
    summarise(report)
    write_outputs(report, args.out or Path(cfg["cache"]["root"]).parent / "m4")


if __name__ == "__main__":
    main()
