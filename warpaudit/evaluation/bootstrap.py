"""Paired group bootstrap (specification §10.2).

Rules implemented here rather than described:

* resample the **highest identifiable independent group** with replacement and
  keep all of its rows -- pipelines, directions, conditions, seeds -- together;
* compare detectors and policies **inside the same resample**, so a contrast is
  paired and its interval reflects the paired quantity, not two marginal ones;
* count **invalid one-class resamples**; if more than 5% are invalid the
  interval is flagged unreliable and the cell is reported as low-information
  rather than silently conditioned on the convenient subset;
* one-sided bounds are percentile bounds at the declared alpha; two-sided
  intervals are reported alongside for estimation.

Conditional bootstrapping (the default here) excludes uncertainty from source
and reference *fitting and calibration*. The primary H4 claim additionally
requires :func:`refit_bootstrap`, which resamples the complete folded
fitting/calibration/evaluation procedure while preserving group identity, fold
membership, and the separation of fitting/calibration/test roles.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BootstrapResult",
    "group_bootstrap",
    "monte_carlo_stability",
    "refit_bootstrap",
]


@dataclass
class BootstrapResult:
    """Distribution of one statistic over resamples, with its validity record."""

    name: str
    estimate: float
    samples: np.ndarray
    n_resamples: int
    n_invalid: int
    alpha: float = 0.05
    notes: list[str] = field(default_factory=list)

    @property
    def valid_samples(self) -> np.ndarray:
        return self.samples[np.isfinite(self.samples)]

    @property
    def invalid_fraction(self) -> float:
        return float(self.n_invalid) / float(self.n_resamples) if self.n_resamples else 1.0

    @property
    def reliable(self) -> bool:
        """False when more than 5% of resamples were invalid (§10.2)."""
        return self.invalid_fraction <= 0.05 and len(self.valid_samples) >= 100

    def lower_one_sided(self) -> float:
        """One-sided ``1 - alpha`` lower bound (percentile method)."""
        s = self.valid_samples
        return float(np.percentile(s, 100 * self.alpha)) if len(s) else float("nan")

    def upper_one_sided(self) -> float:
        s = self.valid_samples
        return float(np.percentile(s, 100 * (1 - self.alpha))) if len(s) else float("nan")

    def two_sided(self) -> tuple[float, float]:
        s = self.valid_samples
        if not len(s):
            return float("nan"), float("nan")
        return (
            float(np.percentile(s, 100 * self.alpha / 2)),
            float(np.percentile(s, 100 * (1 - self.alpha / 2))),
        )

    def as_row(self) -> dict[str, float | int | bool | str]:
        lo, hi = self.two_sided()
        return {
            "quantity": self.name,
            "estimate": self.estimate,
            "lower_one_sided": self.lower_one_sided(),
            "upper_one_sided": self.upper_one_sided(),
            "ci_low": lo,
            "ci_high": hi,
            "n_resamples": self.n_resamples,
            "n_invalid": self.n_invalid,
            "invalid_fraction": self.invalid_fraction,
            "reliable": self.reliable,
            "notes": "; ".join(self.notes),
        }


def _group_index(groups: Sequence[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    g = np.asarray(groups)
    unique = np.unique(g)
    return unique, {u: np.flatnonzero(g == u) for u in unique}


def group_bootstrap(
    groups: Sequence[str],
    statistics: Mapping[str, Callable[[np.ndarray], float]],
    *,
    n_resamples: int = 10_000,
    seed: int = 20260907,
    alpha: float = 0.05,
) -> dict[str, BootstrapResult]:
    """Paired group bootstrap of one or more statistics.

    Each statistic receives the row indices of one resample, so every
    statistic in ``statistics`` sees the *same* resample and their differences
    are paired by construction. A statistic returning ``nan`` (a one-class
    resample, an undefined risk) marks that resample invalid for that
    statistic only.
    """
    unique, index = _group_index(groups)
    n_groups = len(unique)
    if n_groups == 0:
        raise ValueError("no groups to resample")

    rng = np.random.default_rng(seed)
    samples = {k: np.full(n_resamples, np.nan) for k in statistics}

    for b in range(n_resamples):
        drawn = rng.integers(0, n_groups, size=n_groups)
        rows = np.concatenate([index[unique[i]] for i in drawn])
        for name, fn in statistics.items():
            try:
                samples[name][b] = fn(rows)
            except (ValueError, ZeroDivisionError, FloatingPointError):
                samples[name][b] = np.nan

    all_rows = np.arange(sum(len(v) for v in index.values()))
    out: dict[str, BootstrapResult] = {}
    for name, fn in statistics.items():
        try:
            point = float(fn(all_rows))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            point = float("nan")
        arr = samples[name]
        n_invalid = int(np.count_nonzero(~np.isfinite(arr)))
        notes: list[str] = []
        if n_invalid / n_resamples > 0.05:
            notes.append(
                f"{n_invalid}/{n_resamples} resamples were invalid (one-class or undefined); "
                "the interval is unreliable and this cell is low-information (§10.2)"
            )
        out[name] = BootstrapResult(
            name=name,
            estimate=point,
            samples=arr,
            n_resamples=n_resamples,
            n_invalid=n_invalid,
            alpha=alpha,
            notes=notes,
        )
    return out


def refit_bootstrap(
    fold_groups: Mapping[str, Sequence[str]],
    procedure: Callable[[Mapping[str, Sequence[str]]], Mapping[str, float]],
    *,
    n_resamples: int = 1_000,
    seed: int = 20260907,
    alpha: float = 0.05,
    scope_of: Callable[[str], str] | None = None,
) -> dict[str, BootstrapResult]:
    """Bootstrap the complete folded fitting/calibration/evaluation procedure.

    ``fold_groups`` maps a role -- e.g. ``source_train``, ``source_calibration``,
    ``target_train``, ``target_calibration``, ``target_test`` -- to its group
    ids. Each role is resampled **within itself**, which preserves fold
    membership and guarantees that a repeated copy of a group can never enter
    two roles of the same evaluated fold (§10.2).

    With several evaluated folds, pass ``scope_of`` to say which fold each key
    belongs to. Roles must be disjoint *within* a fold, which is the separation
    §10.2 requires; a group that tests in one fold and trains in another is the
    normal cross-validation design, not a leak, so disjointness is checked per
    scope rather than globally.

    ``procedure`` refits the declared hyperparameter-selection procedure on the
    resampled roles and returns the quantities of interest. Everything it
    returns is bootstrapped jointly, so ``Delta_policy``'s two policy outcomes
    stay paired within each resampled target group.
    """
    roles = {k: np.asarray(sorted(set(v))) for k, v in fold_groups.items()}
    for role, ids in roles.items():
        if len(ids) == 0:
            raise ValueError(f"role {role!r} has no groups")
    scope = scope_of or (lambda _key: "")
    by_scope: dict[str, list[str]] = {}
    for key in roles:
        by_scope.setdefault(scope(key), []).append(key)
    overlap = [
        (a, b)
        for keys in by_scope.values()
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
        if set(roles[a]) & set(roles[b])
    ]
    if overlap:
        raise ValueError(
            "roles must be disjoint within each evaluated fold before resampling; "
            f"overlapping: {overlap}"
        )

    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {}

    for _ in range(n_resamples):
        resampled = {
            role: [ids[i] for i in rng.integers(0, len(ids), size=len(ids))]
            for role, ids in roles.items()
        }
        try:
            values = procedure(resampled)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            values = {}
        for key in set(collected) | set(values):
            collected.setdefault(key, []).append(float(values.get(key, np.nan)))

    point = procedure({role: list(ids) for role, ids in roles.items()})
    out: dict[str, BootstrapResult] = {}
    for key, values in collected.items():
        arr = np.asarray(values, dtype=np.float64)
        n_invalid = int(np.count_nonzero(~np.isfinite(arr)))
        notes = ["complete refit bootstrap: includes fitting and calibration uncertainty"]
        if n_invalid / max(len(arr), 1) > 0.05:
            notes.append(f"{n_invalid}/{len(arr)} refits were invalid; the interval is unreliable")
        out[key] = BootstrapResult(
            name=key,
            estimate=float(point.get(key, np.nan)),
            samples=arr,
            n_resamples=len(arr),
            n_invalid=n_invalid,
            alpha=alpha,
            notes=notes,
        )
    return out


def monte_carlo_stability(result: BootstrapResult, n_splits: int = 4) -> dict[str, float]:
    """Monte Carlo stability of the interval endpoints (§10.2 logging duty).

    Splits the resamples into ``n_splits`` blocks and reports the spread of
    each bound across blocks. A large spread means more resamples are needed,
    not that the estimate moved.
    """
    s = result.valid_samples
    if len(s) < n_splits * 50:
        return {"lower_spread": float("nan"), "upper_spread": float("nan"), "n_used": len(s)}
    blocks = np.array_split(s, n_splits)
    lows = [np.percentile(b, 100 * result.alpha) for b in blocks]
    highs = [np.percentile(b, 100 * (1 - result.alpha)) for b in blocks]
    return {
        "lower_spread": float(np.max(lows) - np.min(lows)),
        "upper_spread": float(np.max(highs) - np.min(highs)),
        "n_used": int(len(s)),
    }
