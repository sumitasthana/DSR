"""Synthetic checks on the paper's projection math.

Covers tests T4, T5 and T7 in `day-zero/paper_spec.md`. These run on random or
cached inputs and need no training. They test the paper AS WRITTEN, that is,
Eqs. 1 to 3 and the Lemma 1 discussion. They cannot show what the authors' own
code did, which the paper does not publish.
"""

import torch

from .lora import fuse
from .projection import SubspaceProjector


def _options_for(options, rank):
    """Projector options with the adapter rank overridden for a sweep."""
    return {**options, "rank": rank}


def feasible_update(projector, d_out, rank, generator):
    """Random rank-`rank` update whose input directions lie inside range(P).

    Mirrors the in-training construction dW = (P A) B, so the update is always
    feasible for the current projector. Once P has rank 0 this returns zeros.
    """
    p = projector.matrix
    a = torch.randn(p.shape[0], rank, generator=generator, dtype=p.dtype)
    b = torch.randn(rank, d_out, generator=generator, dtype=p.dtype)
    return (p @ a) @ b


def column_basis(matrix, rank):
    """Orthonormal basis of the update's input directions (its column space)."""
    left, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return left[:, :rank]


def budget_trace(d_in, d_out, rank, tasks, options, seed, scale=1.0):
    """Record capacity and update size for a sequence of synthetic tasks.

    `scale` multiplies each update before it is reserved. Scaling a matrix never
    changes its column space, so the free-rank curve must be identical for every
    nonzero scale. That is test T7: even if lambda entered training, which the
    paper's Eq. 1 says it does not, the saturation curve could not depend on it.
    """
    generator = torch.Generator().manual_seed(seed)
    projector = SubspaceProjector(d_in, _options_for(options, rank))
    rows = []
    for task in range(1, tasks + 1):
        free_before = projector.free_rank
        update = scale * feasible_update(projector, d_out, rank, generator)
        rows.append({
            "task": task,
            "free_rank_before": free_before,
            "reserved": projector.reserve(update),
            "free_rank_after": projector.free_rank,
            "update_norm": float(torch.linalg.norm(update)),
        })
    return rows


def interference(weight, features, rank, options, seed, strength):
    """Do two updates with orthogonal input subspaces act independently?

    Builds update 1, reserves it, then builds update 2 inside the remaining
    subspace, so their input directions are orthogonal by construction. Then it
    measures whether adding update 2 changes behaviour that update 1 produced.
    A nonzero `argmax_change_fraction` contradicts the paper's claim of "zero
    interference with historical tasks" (paper_spec.md D5).
    """
    d_in, d_out = weight.shape
    generator = torch.Generator().manual_seed(seed)
    projector = SubspaceProjector(d_in, _options_for(options, rank))

    first = feasible_update(projector, d_out, rank, generator)
    projector.reserve(first)
    second = feasible_update(projector, d_out, rank, generator)
    projector.reserve(second)

    inputs = features.to(first)
    after_first = inputs @ fuse(weight.to(first), [first], strength)
    after_both = inputs @ fuse(weight.to(first), [first, second], strength)
    overlap = torch.linalg.svdvals(
        column_basis(first, rank).T @ column_basis(second, rank)
    ).max()

    return {
        "input_subspace_overlap": float(overlap),
        "frobenius_inner_product": float((first * second).sum()),
        "argmax_change_fraction": float(
            (after_first.argmax(dim=1) != after_both.argmax(dim=1)).double().mean()
        ),
        "mean_logit_shift": float(
            torch.linalg.norm(after_both - after_first, dim=1).mean()
        ),
        "mean_logit_norm_after_first": float(
            torch.linalg.norm(after_first, dim=1).mean()
        ),
    }
