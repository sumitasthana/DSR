"""M3: CIFAR-100, five deletions, four methods: python -m src.run_m3

This is the first time the paper's own dataset carries a full unlearning
sequence. M2 did the same thing on MNIST, where the capacity budget is 32 tasks
and three deletions could not reach it. Here d_in is 64 and rank is 4, so the
budget is 16, and five deletions is the first real look at how fast it is spent.

The measurement that matters is test T1 in paper_spec.md: how many directions
each trained update actually consumes. The paper's Fig. 3 implies about 2.56 per
task. Its own equations imply 4. Five tasks are enough to tell those apart.

Training never depends on lambda, so one trained sequence is evaluated at every
lambda. The whole run is a linear head on cached features and takes minutes on a
laptop CPU.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from .common import load_config, provenance, save_json
from .data import class_subset
from .evaluate import accuracy, unlearning_metrics
from .lora import fuse
from .run_m0 import baseline_caches
from .unlearn import (METHODS, fused_logits, router_logits, run_sequence,
                      to_projection_dtype, train_router)


def forget_subsets(train_cache, classes):
    """One task per class, in the configured order."""
    return [class_subset(train_cache, label) for label in classes]


def retain_features(train_cache, classes):
    """Training rows that survive every deletion, used to teach the gate "none"."""
    kept = ~torch.isin(train_cache["labels"],
                       torch.tensor(classes, dtype=train_cache["labels"].dtype))
    return train_cache["features"][kept]


def subspace_rows(method, result, classes):
    """T1: directions consumed per task, and the free rank left after each."""
    return [{"method": method, "task": task + 1, "forget_class": classes[task],
             "reserved_rank": result["reserved"][task],
             "free_rank_after": result["free_rank"][task],
             "update_norm": float(torch.linalg.norm(result["updates"][task]))}
            for task in range(len(result["reserved"]))]


def fusion_norms(result, lambdas, classes):
    """T3: size of the accumulated correction, which must stop growing at N_max."""
    zero = torch.zeros_like(result["updates"][0])
    return [{"task": task + 1, "forget_class": classes[task], "lambda": strength,
             "fused_delta_norm": float(torch.linalg.norm(
                 fuse(zero, result["updates"][:task + 1], strength)))}
            for strength in lambdas for task in range(len(result["updates"]))]


def evaluate_sequence(method, result, gate, test_cache, head, classes, lambdas):
    """Every (task count, lambda) pair, with all four readings of the metrics."""
    rows = []
    for task in range(1, len(classes) + 1):
        forgotten = classes[:task]
        for strength in lambdas:
            if method == "router":
                # The router has no lambda: its formula adds adapter outputs
                # directly (Sec. II). The column is kept so the table lines up,
                # and the identical values across lambda are the point of D16.
                logits = router_logits(test_cache["features"], head,
                                       result["adapters"], gate, task)
            else:
                logits = fused_logits(test_cache["features"], head,
                                      result["updates"], strength, task)
            rows.append({"method": method, "tasks": task, "lambda": strength,
                         "forget_class": classes[task - 1],
                         **unlearning_metrics(logits, test_cache["labels"],
                                              forgotten, classes[task - 1])})
    return rows


def run_m3(config, seeds=None):
    """Train and evaluate the four methods over the CIFAR-100 deletion sequence."""
    cfg = deepcopy(config)
    torch.set_num_threads(cfg["runtime"]["threads"])
    context = provenance()
    seeds = seeds or cfg["seeds"]
    classes = cfg["cifar100"]["forget_classes"]
    lambdas = cfg["unlearn"]["lambdas"]

    folder, identity, raw_head, raw_caches = baseline_caches(cfg, context)
    head = to_projection_dtype(raw_head, cfg)
    train_cache = to_projection_dtype(raw_caches["train"], cfg)
    test_cache = to_projection_dtype(raw_caches["test"], cfg)
    clean = accuracy(test_cache, head)["accuracy_percent"]
    print(f"baseline before any deletion: {clean:.2f}% on {len(test_cache['labels'])} test images")

    subsets = forget_subsets(train_cache, classes)
    retain = retain_features(train_cache, classes)
    print(f"deleting classes {classes}: "
          f"{[len(s['labels']) for s in subsets]} training images each")

    metrics, subspace, norms = [], [], []
    for seed in seeds:                                   # the whole sequence, once per seed
        gate = train_router(subsets, retain, cfg, seed)
        for method in METHODS:                           # four methods, each trained from scratch
            result = run_sequence(method, subsets, head, cfg, seed)
            if result["reserved"]:                       # only the two projected methods track it
                subspace += [{"seed": seed, **row}
                             for row in subspace_rows(method, result, classes)]
            if method == "intraining":
                norms += [{"seed": seed, **row}
                          for row in fusion_norms(result, lambdas, classes)]
            metrics += [{"seed": seed, **row}
                        for row in evaluate_sequence(method, result, gate, test_cache,
                                                     head, classes, lambdas)]
            print(f"seed {seed} {method:11s} done", flush=True)

    report = {
        "status": "M3_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity, "environment": context,
        "classes": classes, "seeds": seeds, "lambdas": lambdas,
        "rank": cfg["projection"]["rank"],
        "capacity": cfg["model"]["feature_dim"] // cfg["projection"]["rank"],
        "clean_accuracy_percent": clean,
        "metrics": metrics, "subspace": subspace, "fusion_norms": norms,
        "limitations": [
            "Public checkpoint at 68.83% clean, not the paper's 60.39% starting point.",
            "N = 5 of a 16-task budget, so this cannot show saturation, only the rate.",
            "Learning rate, epochs, optimiser and batch size for unlearning are ours; "
            "the paper states none of them.",
            "The five deleted classes are the first five by index; the paper names none.",
        ],
    }
    return report


def summarise(report):
    """Print the two things this run exists to answer."""
    rank = report["rank"]
    reserved = {row["reserved_rank"] for row in report["subspace"]}
    print()
    print("T1: directions consumed per task")
    for method in sorted({row["method"] for row in report["subspace"]}):
        rows = [r for r in report["subspace"] if r["method"] == method and r["seed"] == report["seeds"][0]]
        print(f"  {method:11s} reserved {[r['reserved_rank'] for r in rows]}  "
              f"free rank after {[r['free_rank_after'] for r in rows]}")
    print(f"  distinct values across every seed and method: {sorted(reserved)}")
    if reserved == {rank}:
        print(f"  Every task consumed the full rank {rank}. Budget is "
              f"{report['capacity']} tasks, so D2 and D3 stand.")
    else:
        print(f"  Not always {rank}. Fig. 3's ~2.56 per task becomes reachable; "
              f"D2 needs rewriting.")

    print()
    print("Retain / forget at the end of the sequence, mean over seeds")
    final = [r for r in report["metrics"] if r["tasks"] == len(report["classes"])]
    for method in METHODS:
        for strength in (0.5, 1.0):
            rows = [r for r in final if r["method"] == method and r["lambda"] == strength]
            if not rows:
                continue
            retain = np.mean([r["retain_excluding_forgotten"] for r in rows])
            forget = np.mean([r["forget_all"] for r in rows])
            print(f"  {method:11s} lambda {strength}: retain {retain:6.2f}  forget {forget:6.2f}")


def write_outputs(report, folder):
    """One JSON report plus flat CSVs, so a notebook can read either."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "report.json", report)
    for name in ("metrics", "subspace", "fusion_norms"):
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
    parser.add_argument("--out", type=Path, default=None,
                        help="Output folder. Defaults to outputs/m3.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = run_m3(cfg)
    summarise(report)
    write_outputs(report, args.out or Path(cfg["cache"]["root"]).parent / "m3")


if __name__ == "__main__":
    main()
