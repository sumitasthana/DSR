"""Prepare a frozen CIFAR-100 baseline: python -m src.run_m0 --check-only."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from .common import digest, load_config, provenance, save_json, save_tensors, seed_everything
from .cache import load_or_create_cache
from .data import cifar100_loader, extract_features, validate_cache
from .evaluate import accuracy
from .models import checkpoint_hash, head_state, load_backbone


def run(config_path, check_only=False):
    """File-based entry point shared by the CLI and existing callers."""
    return prepare_baseline(load_config(config_path), check_only=check_only)


def baseline_caches(cfg, context):
    """Frozen head and verified train/test feature caches for the M0 backbone.

    The folder name is a hash of the configuration, the checkpoint and the
    environment, so a cache is only reused by a run that would have produced an
    identical one. Later milestones call this to get the same frozen starting
    point M0 reported on, rather than rebuilding it themselves.
    """
    model = load_backbone(cfg)
    identity = {"config": cfg, "checkpoint_sha256": checkpoint_hash(model), "environment": context}
    folder = Path(cfg["cache"]["root"]) / digest(identity)
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "identity.json", identity)
    head = head_state(model)
    save_tensors(folder / "head.pt", head)
    caches = {}
    for split in ("train", "test"):
        loader = cifar100_loader(cfg, split)
        print(f"Preparing {split}: {len(loader.dataset)} images", flush=True)
        cache, resumed = load_or_create_cache(
            folder, split, identity,
            create=lambda: extract_features(model, loader, cfg),
            validate=lambda value: validate_cache(value, cfg, len(loader.dataset)),
        )
        print(f"{'Resumed' if resumed else 'Saved'} verified {split} cache", flush=True)
        caches[split] = cache
    return folder, identity, head, caches


def prepare_baseline(config, check_only=False):
    """Run M0 from a resolved configuration, including from a notebook.

    Load configuration with load_config first to resolve storage paths. A copy
    prevents the runner from changing the caller's settings between experiments.
    """
    cfg = deepcopy(config)
    torch.set_num_threads(cfg["runtime"]["threads"])
    context = provenance()
    print(json.dumps({"configuration": cfg, "environment": context}, indent=2))
    device = torch.device(cfg["runtime"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    print("Plan: fixed public checkpoint, train/test caches, no training; runtime estimate unknown.")
    print("Resume boundary: completed dataset split. An interrupted split is recomputed.")
    if check_only:
        return {"status": "environment_checked"}
    seed_everything(cfg["seeds"][0])
    folder, identity, head, caches = baseline_caches(cfg, context)
    rows = []
    for seed in cfg["seeds"]:
        seed_everything(seed)
        rows.append({"seed": seed, **accuracy(caches["test"], head)})
    values = [row["accuracy_percent"] for row in rows]
    report = {
        "status": "M0_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity, "rows": rows,
        "mean_accuracy_percent": float(np.mean(values)),
        "std_accuracy_percent": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "limitations": [
            "Public checkpoint, not the paper's undisclosed checkpoint.",
            "Seeds repeat evaluation of one fixed checkpoint and cache; they are not independent training runs.",
            "Clean accuracy covers the full test set, not a task-dependent retained subset.",
            "No unlearning or projection experiment has run.",
            "Local output is not persistent storage in Colab; configure a mounted Drive destination there.",
        ],
    }
    save_json(folder / "report.json", report)
    csv_path = folder / "accuracy.csv"
    csv_path.write_text("seed,correct,count,accuracy_percent\n" + "".join(
        f'{r["seed"]},{r["correct"]},{r["count"]},{r["accuracy_percent"]}\n' for r in rows), encoding="utf-8")
    import csv
    with csv_path.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == len(rows)
    print(json.dumps(report, indent=2))
    print(f"Verified report: {folder / 'report.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.yaml")
    parser.add_argument("--check-only", action="store_true", help="Check environment without downloading or running inference")
    args = parser.parse_args()
    run(args.config, args.check_only)


if __name__ == "__main__":
    main()
