"""Weighted discrimination and calibration metrics (specification §10.1, §9.1).

Everything here takes explicit weights, because every primary estimate in this
study is group-first weighted (§10.1). Everything here also refuses to invent
a number when the data cannot support one: a one-class cell returns ``nan``
and is *marked undefined*, never quietly dropped so the remaining cells can be
averaged (§10.1).

Orientation convention, fixed once for the whole package:

    ``score`` is a RISK score. Higher means more likely to be a failure.
    A case is accepted when ``score <= threshold``.

Binary failure is the positive class. Score orientation is recorded from
theory or source development data only; a target AUROC below 0.5 is never
flipped after inspecting target labels (§7.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "BinaryCounts",
    "ReliabilityBin",
    "average_precision",
    "auroc",
    "brier_score",
    "calibration_in_the_large",
    "class_bearing_groups",
    "expected_calibration_error",
    "prevalence",
    "reliability_bins",
]


def _check(scores, labels, weights):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if s.shape != y.shape or s.ndim != 1:
        raise ValueError("scores and labels must be 1-D arrays of equal length")
    if weights is None:
        w = np.full(s.shape, 1.0 / max(len(s), 1), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != s.shape:
            raise ValueError("weights must match scores")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    finite = np.isfinite(s) & np.isfinite(y) & np.isfinite(w)
    s, y, w = s[finite], y[finite], w[finite]
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be binary (0/1) where defined")
    return s, y, w


@dataclass(frozen=True)
class BinaryCounts:
    """Support behind a discrimination estimate (§10.1, §10.3)."""

    n: int
    n_positive: int
    n_negative: int
    weight_positive: float
    weight_negative: float
    prevalence: float
    n_failure_bearing_groups: int = -1
    n_success_bearing_groups: int = -1

    @property
    def both_classes_present(self) -> bool:
        return self.n_positive > 0 and self.n_negative > 0

    @property
    def low_information(self, threshold: int = 10) -> bool:
        """Screening rule of §10.3. Not a power guarantee."""
        if self.n_failure_bearing_groups < 0:
            return not self.both_classes_present
        return (
            self.n_failure_bearing_groups < threshold or self.n_success_bearing_groups < threshold
        )


def prevalence(labels, weights=None) -> float:
    _, y, w = _check(np.zeros(len(labels)), labels, weights)
    total = w.sum()
    return float((w * y).sum() / total) if total > 0 else float("nan")


def class_bearing_groups(labels, groups) -> tuple[int, int]:
    """Independent groups bearing at least one failure / at least one success.

    Membership may overlap: a group containing both outcomes counts in both
    (§10.3). Group counts, not pair totals, are what the information gate
    reads.
    """
    y = np.asarray(labels, dtype=np.float64)
    g = np.asarray(groups)
    fail = {gi for gi, yi in zip(g, y, strict=False) if yi == 1}
    ok = {gi for gi, yi in zip(g, y, strict=False) if yi == 0}
    return len(fail), len(ok)


def auroc(scores, labels, weights=None) -> float:
    """Weighted AUROC with explicit half-credit for ties.

    ``AUROC = [ sum_{i in pos, j in neg} w_i w_j ( 1[s_i > s_j] + 0.5 * 1[s_i = s_j] ) ]
    / (W_pos * W_neg)``

    Returns ``nan`` when either class is absent, so the caller can mark the
    cell undefined instead of averaging over the cells that happened to work.
    """
    s, y, w = _check(scores, labels, weights)
    pos = y == 1
    neg = ~pos
    W_pos, W_neg = w[pos].sum(), w[neg].sum()
    if W_pos <= 0 or W_neg <= 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    s_sorted, y_sorted, w_sorted = s[order], y[order], w[order]

    # Cumulative negative weight strictly below, and tied with, each block.
    neg_w = np.where(y_sorted == 0, w_sorted, 0.0)
    pos_w = np.where(y_sorted == 1, w_sorted, 0.0)

    boundaries = np.flatnonzero(np.diff(s_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(s_sorted)]))

    cum_neg_below = 0.0
    total = 0.0
    for a, b in zip(starts, ends, strict=False):
        block_neg = neg_w[a:b].sum()
        block_pos = pos_w[a:b].sum()
        total += block_pos * (cum_neg_below + 0.5 * block_neg)
        cum_neg_below += block_neg
    return float(total / (W_pos * W_neg))


def average_precision(scores, labels, weights=None) -> float:
    """Weighted average precision -- the specified PR summary (§10.1).

    ``AP = sum_k (R_k - R_{k-1}) * P_k`` over descending score thresholds,
    with tied scores collapsed into one threshold block. This is deliberately
    *not* trapezoidal PR-AUC; the two are not mixed anywhere in this project.
    """
    s, y, w = _check(scores, labels, weights)
    pos_total = (w * (y == 1)).sum()
    if pos_total <= 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted, w_sorted = s[order], y[order], w[order]
    tp = np.cumsum(w_sorted * (y_sorted == 1))
    fp = np.cumsum(w_sorted * (y_sorted == 0))

    boundaries = np.flatnonzero(np.diff(s_sorted)) + 1
    ends = np.concatenate((boundaries, [len(s_sorted)])) - 1

    recall_prev = 0.0
    ap = 0.0
    for k in ends:
        precision = tp[k] / (tp[k] + fp[k]) if (tp[k] + fp[k]) > 0 else 0.0
        recall = tp[k] / pos_total
        ap += (recall - recall_prev) * precision
        recall_prev = recall
    return float(ap)


def brier_score(probabilities, labels, weights=None) -> float:
    p, y, w = _check(probabilities, labels, weights)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    total = w.sum()
    return float((w * (p - y) ** 2).sum() / total) if total > 0 else float("nan")


def calibration_in_the_large(probabilities, labels, weights=None) -> float:
    """Mean predicted failure probability minus observed failure rate (§9.1)."""
    p, y, w = _check(probabilities, labels, weights)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    total = w.sum()
    if total <= 0:
        return float("nan")
    return float(((w * p).sum() - (w * y).sum()) / total)


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    n: int
    weight: float
    mean_predicted: float
    observed_rate: float


def reliability_bins(probabilities, labels, weights=None, n_bins: int = 5):
    """Fixed equal-width probability bins with their counts (§9.1).

    Counts travel with every bin because small datasets cannot support an
    elaborate reliability diagram, and a bin holding two cases must be visible
    as such.
    """
    p, y, w = _check(probabilities, labels, weights)
    if int(n_bins) < 2:
        raise ValueError("n_bins must be at least 2")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    out: list[ReliabilityBin] = []
    for i in range(int(n_bins)):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        wt = w[sel].sum()
        out.append(
            ReliabilityBin(
                lower=float(lo),
                upper=float(hi),
                n=int(sel.sum()),
                weight=float(wt),
                mean_predicted=float((w[sel] * p[sel]).sum() / wt) if wt > 0 else float("nan"),
                observed_rate=float((w[sel] * y[sel]).sum() / wt) if wt > 0 else float("nan"),
            )
        )
    return out


def expected_calibration_error(probabilities, labels, weights=None, n_bins: int = 5) -> float:
    """Secondary metric. Always reported with bin counts (§9.1)."""
    bins = reliability_bins(probabilities, labels, weights, n_bins)
    total = sum(b.weight for b in bins)
    if total <= 0:
        return float("nan")
    return float(
        sum(
            b.weight * abs(b.mean_predicted - b.observed_rate)
            for b in bins
            if b.weight > 0 and np.isfinite(b.mean_predicted)
        )
        / total
    )
