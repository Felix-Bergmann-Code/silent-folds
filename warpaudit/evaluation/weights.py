"""Group-first sampling weights (specification §10.1).

    The target population is sampled group-first: select a group uniformly,
    then a pair within the group, then a pipeline within the declared common
    block.

Implementing that literally matters because groups contain very different
numbers of pairs. Row-weighted metrics are reported as a secondary summary,
never as the primary estimate, and synthetic corruptions and seed repeats
never become additional independent subjects -- they share their pair's
weight.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

__all__ = [
    "group_first_weights",
    "inverse_group_size_weights",
    "normalise",
    "row_weights",
]


def normalise(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must have a positive finite sum")
    return w / total


def row_weights(n: int) -> np.ndarray:
    """Uniform row weighting -- the secondary summary of §10.1."""
    if int(n) <= 0:
        raise ValueError("n must be positive")
    return np.full(int(n), 1.0 / int(n), dtype=np.float64)


def group_first_weights(
    groups: Sequence[str],
    pairs: Sequence[str],
    pipelines: Sequence[str] | None = None,
) -> np.ndarray:
    """Weights implementing uniform group -> uniform pair -> uniform pipeline.

    ``pipelines=None`` collapses the third stage, which is correct for a
    single-pipeline cell. Rows sharing a ``(group, pair, pipeline)`` triple --
    seeds, corruption severities, reverse runs -- split that triple's weight
    equally, so repeating a run cannot inflate its influence.
    """
    groups = list(groups)
    pairs = list(pairs)
    n = len(groups)
    if n == 0:
        raise ValueError("at least one row is required")
    if len(pairs) != n:
        raise ValueError("groups and pairs must have equal length")
    pipelines = list(pipelines) if pipelines is not None else ["*"] * n
    if len(pipelines) != n:
        raise ValueError("pipelines must have the same length as groups")

    pairs_per_group: dict[str, set[str]] = defaultdict(set)
    pipelines_per_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows_per_triple: dict[tuple[str, str, str], int] = defaultdict(int)
    for g, p, q in zip(groups, pairs, pipelines, strict=False):
        pairs_per_group[g].add(p)
        pipelines_per_pair[(g, p)].add(q)
        rows_per_triple[(g, p, q)] += 1

    n_groups = len(pairs_per_group)
    w = np.empty(n, dtype=np.float64)
    for i, (g, p, q) in enumerate(zip(groups, pairs, pipelines, strict=False)):
        w[i] = (
            (1.0 / n_groups)
            * (1.0 / len(pairs_per_group[g]))
            * (1.0 / len(pipelines_per_pair[(g, p)]))
            * (1.0 / rows_per_triple[(g, p, q)])
        )
    return normalise(w)


def inverse_group_size_weights(
    groups: Sequence[str], pipelines: Sequence[str] | None = None
) -> np.ndarray:
    """Learner sample weights (spec §7.4).

    Inverse group size, with equal total pipeline weight within a group, so a
    subject with many pairs cannot dominate the fit and a pipeline that
    returns more rows cannot outvote the others.
    """
    groups = list(groups)
    n = len(groups)
    if n == 0:
        raise ValueError("at least one row is required")
    pipelines = list(pipelines) if pipelines is not None else ["*"] * n
    if len(pipelines) != n:
        raise ValueError("pipelines must have the same length as groups")

    rows_per_group_pipeline: dict[tuple[str, str], int] = defaultdict(int)
    pipelines_per_group: dict[str, set[str]] = defaultdict(set)
    for g, q in zip(groups, pipelines, strict=False):
        rows_per_group_pipeline[(g, q)] += 1
        pipelines_per_group[g].add(q)

    n_groups = len(pipelines_per_group)
    w = np.empty(n, dtype=np.float64)
    for i, (g, q) in enumerate(zip(groups, pipelines, strict=False)):
        w[i] = (
            (1.0 / n_groups)
            * (1.0 / len(pipelines_per_group[g]))
            * (1.0 / rows_per_group_pipeline[(g, q)])
        )
    return normalise(w)
