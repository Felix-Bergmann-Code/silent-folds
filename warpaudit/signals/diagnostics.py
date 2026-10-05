"""Development-only informativeness gates for C and E1 (specification §7.1a-§7.2a)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

__all__ = ["DiagnosticDecision", "classify_cycle", "classify_e1"]


@dataclass(frozen=True)
class DiagnosticDecision:
    family: str
    classification: str  # informative | degenerate | unresolved
    criteria: dict[str, float | bool]
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def classify_cycle(case_magnitudes: np.ndarray, *, tolerance: float) -> DiagnosticDecision:
    """Apply the frozen median-and-IQR cycle degeneracy rule."""
    values = _finite(case_magnitudes)
    if tolerance <= 0:
        raise ValueError("cycle tolerance must be positive")
    if len(values) < 3:
        return DiagnosticDecision(
            "C",
            "unresolved",
            {"n_cases": float(len(values)), "tolerance": tolerance},
            ("fewer than three finite development cases",),
        )
    median = float(np.median(values))
    iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
    degenerate = median <= tolerance and iqr <= tolerance
    return DiagnosticDecision(
        "C",
        "degenerate" if degenerate else "informative",
        {"n_cases": float(len(values)), "median": median, "iqr": iqr, "tolerance": tolerance},
        ("median and across-case IQR are both at or below tolerance",) if degenerate else (),
    )


def classify_e1(
    correspondence_spread: np.ndarray,
    seed_only_spread: np.ndarray,
    geometric_error: np.ndarray,
    *,
    repeatability_tolerance: float,
    seed_dominance_max_ratio: float,
    min_error_spearman: float,
) -> DiagnosticDecision:
    """Classify E1 using only development cases and prespecified thresholds."""
    corr = np.asarray(correspondence_spread, dtype=np.float64)
    seed = np.asarray(seed_only_spread, dtype=np.float64)
    error = np.asarray(geometric_error, dtype=np.float64)
    if not (corr.shape == seed.shape == error.shape and corr.ndim == 1):
        raise ValueError("E1 diagnostic arrays must be matching one-dimensional vectors")
    valid = np.isfinite(corr) & np.isfinite(seed) & np.isfinite(error)
    corr, seed, error = corr[valid], seed[valid], error[valid]
    if len(corr) < 5 or len(np.unique(corr)) < 2 or len(np.unique(error)) < 2:
        return DiagnosticDecision(
            "E1",
            "unresolved",
            {"n_cases": float(len(corr))},
            ("insufficient finite or varying development evidence",),
        )

    median_corr = float(np.median(corr))
    median_seed = float(np.median(seed))
    seed_ratio = median_seed / median_corr if median_corr > 0 else float("inf")
    rho_result = spearmanr(corr, error)
    rho = float(rho_result.statistic)
    repeatable = median_seed <= repeatability_tolerance
    seed_controlled = seed_ratio <= seed_dominance_max_ratio
    associated = rho >= min_error_spearman
    reasons = []
    if not repeatable:
        reasons.append("seed-only repeatability exceeds tolerance")
    if not seed_controlled:
        reasons.append("seed-only spread dominates correspondence-resampling spread")
    if not associated:
        reasons.append("development error association is below the frozen minimum")
    criteria: dict[str, float | bool] = {
        "n_cases": float(len(corr)),
        "median_correspondence_spread": median_corr,
        "median_seed_only_spread": median_seed,
        "seed_dominance_ratio": seed_ratio,
        "error_spearman": rho,
        "repeatable": repeatable,
        "seed_controlled": seed_controlled,
        "error_associated": associated,
    }
    return DiagnosticDecision(
        "E1", "informative" if not reasons else "degenerate", criteria, tuple(reasons)
    )
