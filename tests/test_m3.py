"""Checks on the M3 orchestration helpers, with no training and no dataset."""

import math

import torch

from src.run_m3 import fusion_norms, retain_features, subspace_rows


def _cache(labels):
    return {"features": torch.arange(len(labels) * 2, dtype=torch.float64).reshape(len(labels), 2),
            "labels": torch.tensor(labels)}


def test_retain_features_drops_every_deleted_class():
    cache = _cache([0, 1, 2, 3, 0, 2])
    kept = retain_features(cache, [0, 2])
    assert len(kept) == 2
    torch.testing.assert_close(kept, cache["features"][torch.tensor([1, 3])])


def test_subspace_rows_pair_each_task_with_its_class():
    result = {"reserved": [4, 4], "free_rank": [60, 56],
              "updates": [torch.zeros(3, 2, dtype=torch.float64),
                          torch.ones(3, 2, dtype=torch.float64)]}
    rows = subspace_rows("intraining", result, [7, 9])
    assert [r["task"] for r in rows] == [1, 2]
    assert [r["forget_class"] for r in rows] == [7, 9]
    assert [r["reserved_rank"] for r in rows] == [4, 4]
    assert rows[0]["update_norm"] == 0.0
    assert rows[1]["update_norm"] == math.sqrt(6.0)


def test_fusion_norm_is_linear_in_lambda_and_cumulative_in_tasks():
    update = torch.ones(2, 2, dtype=torch.float64)
    result = {"updates": [update, update]}
    rows = fusion_norms(result, [0.5, 1.0], [3, 4])
    by_key = {(r["lambda"], r["task"]): r["fused_delta_norm"] for r in rows}
    # Doubling lambda doubles the correction; a second identical update doubles it again.
    assert by_key[(1.0, 1)] == 2 * by_key[(0.5, 1)]
    assert by_key[(0.5, 2)] == 2 * by_key[(0.5, 1)]
