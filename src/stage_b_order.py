"""Paired comparison of the same targets at early and late request positions."""

from copy import deepcopy

import pandas as pd

from .stage_b_start import run_start_check, summarize


def request_orders(tasks, swap_count):
    original = list(range(tasks))
    if not 0 < swap_count <= tasks // 2:
        raise ValueError("The early and late groups must be nonempty and disjoint")
    return {"original": original,
            "swapped": original[-swap_count:] + original[swap_count:-swap_count] + original[:swap_count]}


def measure_orders(cfg, head, fit, development, identity, folder, sync=None):
    settings = deepcopy(cfg)
    design = cfg["stage_b_order"]
    options = settings["stage_b_start"]
    options["lambdas"] = [design["lambda"]]
    histories, targets, scores = [], [], []
    for name, order in request_orders(options["tasks"], design["swap_count"]).items():
        rows = run_start_check(settings, head, fit, development, identity, folder / name,
                               sync=sync, class_order=order)
        h, t, s = summarize(rows, options, class_order=order)
        histories.append(h.assign(order=name))
        targets.append(t.assign(order=name))
        scores.append(s.assign(order=name))
    return tuple(pd.concat(parts, ignore_index=True) for parts in (histories, targets, scores))


def paired_results(targets, cfg):
    options = cfg["stage_b_start"]
    design = cfg["stage_b_order"]
    tasks, count = options["tasks"], design["swap_count"]
    early_ids = list(range(count))
    late_ids = list(range(tasks - count, tasks))
    moved = targets[targets.class_id.isin(early_ids + late_ids)]
    keys = ["seed", "class_id"]
    measures = ["order", "request", "before", "after", "drop_pp", "at_boundary", "reversal_pp"]
    early = moved[moved.request <= count][keys + measures]
    late = moved[moved.request > tasks - count][keys + measures]
    pairs = early.merge(late, on=keys, suffixes=("_early", "_late"), validate="one_to_one")
    if set(map(tuple, pairs[keys].values)) != {(s, c) for s in cfg["seeds"] for c in early_ids + late_ids}:
        raise ValueError("Missing a matched early/late observation")
    pairs["late_minus_early_pp"] = pairs.after_late - pairs.after_early
    pairs["extra_drop_early_pp"] = pairs.drop_pp_early - pairs.drop_pp_late
    stats = pairs.groupby("class_id").late_minus_early_pp.agg(["mean", "std", "min"]).reset_index()
    # Compare variation across paired seeds, never treating the images or classes as extra seeds.
    stats["consistent_late_penalty"] = ((stats["min"] >= design["min_position_gap_pp"]) &
                                         (stats["mean"] > stats["std"]))
    hard = pairs[pairs.class_id.isin(late_ids)]
    limit = options["max_old_accuracy_pct"]
    decision = {"all_moved_classes_show_late_penalty": bool(stats.consistent_late_penalty.all()),
                "original_late_classes_above_limit_in_both_positions":
                    bool(((hard.after_early > limit) & (hard.after_late > limit)).all())}
    return pairs.sort_values(keys).reset_index(drop=True), stats, decision
