"""M5: per-deletion saturation measurements with immutable, seed-level recovery."""

from copy import deepcopy
import json
from pathlib import Path

import torch

from .common import digest, save_json
from .evaluate import accuracy, unlearning_metrics
from .lora import fuse
from .run_m3 import forget_subsets
from .unlearn import fused_logits, run_sequence, to_projection_dtype


def run_identity(cfg, context, cache_identity):
    """Bind resume to code, environment, settings and the input checkpoint/cache."""
    return {"protocol": "m5-saturation-v1", "configuration": deepcopy(cfg),
            "environment": context, "cache_identity": cache_identity}


def write_once(path, value):
    """Reopen an atomic JSON record; refuse changed results under an old label."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Existing result differs: {path}. Use a new run label.")
    else:
        save_json(path, value)
    if json.loads(path.read_text(encoding="utf-8")) != value:
        raise IOError(f"Could not verify {path}")


def _save_record(path, identity, rows):
    write_once(path, {"identity": digest(identity), "rows": rows, "sha256": digest(rows)})


def load_seed(path, identity, seed, tasks):
    """Only a complete, matching, intact seed is eligible for skipping training."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = record["rows"]
    if (record["identity"] != digest(identity) or record["sha256"] != digest(rows)
            or [r["task"] for r in rows] != list(range(tasks + 1))
            or any(r["seed"] != seed for r in rows)):
        raise ValueError("Invalid completed seed or changed run identity; use a new run label")
    return rows


def resume_status(folder, identity, seeds, tasks):
    folder = Path(folder)
    write_once(folder / "identity.json", identity)
    complete = []
    for seed in seeds:
        path = folder / "seeds" / f"seed-{seed}.json"
        if path.exists():
            load_seed(path, identity, seed, tasks)
            complete.append(seed)
    return {"complete": complete, "remaining": [s for s in seeds if s not in complete]}


def measure_seed(cfg, head, train_cache, test_cache, seed, on_row=None):
    """Use existing training and metric functions; evaluate after each task."""
    options = cfg["m5"]
    classes = list(range(options["tasks"]))
    if not 0 < len(classes) < head["weight"].shape[1]:
        raise ValueError("M5 needs deletion classes and at least one retained class")
    width = head["weight"].shape[0]
    strength = options["lambda"]
    baseline = accuracy(test_cache, head)["accuracy_percent"]
    rows = [{"seed": seed, "task": 0, "forget_class": None, "lambda": strength,
             "reserved_rank": 0, "free_rank": width, "update_norm": 0.0,
             "weight_step_max_abs": 0.0, "weight_drift_since_exhaustion": None,
             "retain_excluding_forgotten": baseline, "retain_full": baseline,
             "forget_current": None, "forget_all": None}]
    previous = head["weight"].clone()
    exhausted_weight = None
    if on_row:
        on_row(rows[0])

    def record(task, result):
        nonlocal previous, exhausted_weight
        with torch.inference_mode():
            weight = fuse(head["weight"], result["updates"], strength)
            if exhausted_weight is None and result["free_rank"][-1] == 0:
                exhausted_weight = weight.clone()
            row = {"seed": seed, "task": task, "forget_class": classes[task - 1],
                   "lambda": strength, "reserved_rank": result["reserved"][-1],
                   "free_rank": result["free_rank"][-1],
                   "update_norm": float(torch.linalg.norm(result["updates"][-1])),
                   "weight_step_max_abs": float((weight - previous).abs().max()),
                   "weight_drift_since_exhaustion": None if exhausted_weight is None else
                       float((weight - exhausted_weight).abs().max()),
                   **unlearning_metrics(
                       fused_logits(test_cache["features"], head, result["updates"], strength, task),
                       test_cache["labels"], classes[:task], classes[task - 1])}
            previous = weight.clone()
        rows.append(row)
        if on_row:
            on_row(row)

    run_sequence("intraining", forget_subsets(train_cache, classes), head, cfg, seed,
                 on_task=record)
    return rows


def run_seed_records(folder, identity, seeds, tasks, measure, sync=None):
    """Persist task records from measure(seed, on_row); replay incomplete seeds."""
    folder = Path(folder)
    status = resume_status(folder, identity, seeds, tasks)
    print("Resume:", status)
    all_rows = []
    for seed in seeds:
        path = folder / "seeds" / f"seed-{seed}.json"
        if seed not in status["complete"]:
            def persist(row):
                # One immutable JSONL file per task avoids replacing an uploaded log.
                target = folder / "progress" / f"seed-{seed}" / f"task-{row['task']:03d}.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                record = {"identity": digest(identity), "row": row, "sha256": digest(row)}
                payload = json.dumps(record, sort_keys=True) + "\n"
                if target.exists() and target.read_text(encoding="utf-8") != payload:
                    # A truncated local record can be repaired by full seed replay;
                    # valid but different results must not be silently overwritten.
                    try:
                        json.loads(target.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass
                    else:
                        raise ValueError("Replayed task differs; use a new run label")
                temporary = target.with_suffix(".tmp")
                temporary.write_text(payload, encoding="utf-8")
                temporary.replace(target)
                if json.loads(target.read_text(encoding="utf-8")) != record:
                    raise IOError("Progress record failed verification")
                if sync:
                    sync(folder)
            rows = measure(seed, persist)
            _save_record(path, identity, rows)
            if sync:
                sync(folder)
        all_rows.extend(load_seed(path, identity, seed, tasks))
        print(f"seed {seed}: verified {tasks + 1} rows")
    return all_rows


def run_m5(cfg, raw_head, raw_caches, identity, folder, sync=None):
    """Measure M5 on the full train/test caches using completed-seed recovery."""
    head = to_projection_dtype(raw_head, cfg)
    train = to_projection_dtype(raw_caches["train"], cfg)
    test = to_projection_dtype(raw_caches["test"], cfg)
    torch.set_num_threads(cfg["runtime"]["threads"])

    def measure(seed, persist):
        return measure_seed(cfg, head, train, test, seed, on_row=persist)

    return run_seed_records(folder, identity, cfg["seeds"], cfg["m5"]["tasks"], measure, sync)
