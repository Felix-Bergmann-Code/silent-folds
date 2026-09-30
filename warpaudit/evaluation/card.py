"""Standalone evaluation card for externally produced registration scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import (
    auroc,
    average_precision,
    brier_score,
    calibration_in_the_large,
    prevalence,
)
from .riskcoverage import CasePopulation, risk_coverage_curve
from .weights import inverse_group_size_weights

REQUIRED_COLUMNS = ("score", "failure", "group")


@dataclass(frozen=True)
class EvaluationCard:
    summary: dict[str, object]
    curve: pd.DataFrame


def generalized_risk_area(failure_prevalence: float, failure_auroc: float) -> float:
    """Closed-form binary-loss AUGRC reference.

    This helper is retained for analytic checks. Evaluation cards report the
    empirical case-level area returned by ``RiskCoverageCurve.generalized_area``.
    """
    p, a = float(failure_prevalence), float(failure_auroc)
    if not np.isfinite([p, a]).all() or not 0 <= p <= 1 or not 0 <= a <= 1:
        return float("nan")
    return float((1.0 - a) * p * (1.0 - p) + 0.5 * p * p)


def evaluate_scores(
    frame: pd.DataFrame,
    *,
    requested_coverages: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9),
) -> EvaluationCard:
    """Evaluate an external score table without fitting or changing orientation."""
    missing = set(REQUIRED_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("score table is empty")
    if any(not 0 < float(c) <= 1 for c in requested_coverages):
        raise ValueError("requested coverages must lie in (0, 1]")

    data = frame.copy()
    score = pd.to_numeric(data["score"], errors="coerce").to_numpy(float)
    failure = pd.to_numeric(data["failure"], errors="coerce").to_numpy(float)
    if not np.isfinite(failure).all() or not np.isin(failure, [0.0, 1.0]).all():
        raise ValueError("failure must contain only finite 0/1 values")
    if "explicit_failure" in data:
        raw_explicit = data["explicit_failure"]
        if raw_explicit.dtype == bool:
            explicit = raw_explicit.to_numpy()
        else:
            normalized = raw_explicit.astype(str).str.strip().str.lower()
            allowed = {"true", "false", "1", "0"}
            if not set(normalized).issubset(allowed):
                raise ValueError("explicit_failure must contain only true/false or 1/0")
            explicit = normalized.isin(["true", "1"]).to_numpy()
    else:
        explicit = np.zeros(len(data), dtype=bool)
    if np.any(explicit & (failure != 1)):
        raise ValueError("every explicit failure must have failure=1")
    if np.any((~explicit) & (~np.isfinite(score))):
        raise ValueError("score may be missing only for explicit failures")
    group = data["group"].astype(str).to_numpy()
    if np.any(pd.isna(data["group"])) or np.any(data["group"].astype(str).str.len() == 0):
        raise ValueError("group identifiers must be non-empty")
    weights = inverse_group_size_weights(group)
    population = CasePopulation(score, failure, weights, group, explicit)
    rc = risk_coverage_curve(population)
    area, lo, hi = rc.area()
    generalized_area, generalized_lo, generalized_hi = rc.generalized_area()
    returned = (~explicit) & np.isfinite(score)
    roc = auroc(score[returned], failure[returned], weights[returned])
    prev = prevalence(failure[returned], weights[returned])

    summary: dict[str, object] = {
        "contract": {
            "score_orientation": "higher means greater failure risk",
            "weighting": "equal total weight per declared group",
            "explicit_failures": "included in attempted denominator and automatically rejected",
            "tie_rule": rc.convention,
        },
        "support": {
            "cases": int(len(data)),
            "groups": int(len(set(group))),
            "failures": int(np.sum(failure == 1)),
            "successes": int(np.sum(failure == 0)),
            "explicit_failures": int(explicit.sum()),
        },
        "ranking": {
            "auroc": roc,
            "average_precision": average_precision(
                score[returned], failure[returned], weights[returned]
            ),
            "scope": "returned transforms with finite scores",
        },
        "probability": None,
        "acceptance": {
            "aurc": area,
            "aurc_coverage_interval": [lo, hi],
            "augrc": generalized_area,
            "augrc_coverage_interval": [generalized_lo, generalized_hi],
            "returned_transform_failure_prevalence": prev,
            "maximum_attainable_coverage": rc.max_attainable_coverage,
            "risk_at_requested_coverage": {
                format(float(c), ".6g"): rc.risk_at(float(c)) for c in requested_coverages
            },
        },
    }
    if "probability" in data:
        probability = pd.to_numeric(data["probability"], errors="coerce").to_numpy(float)
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("probability must be finite and lie in [0, 1]")
        valid = ~explicit
        summary["probability"] = {
            "brier_all_attempts": brier_score(probability, failure, weights),
            "calibration_bias_all_attempts": calibration_in_the_large(
                probability, failure, weights
            ),
            "brier_returned_transforms_only": brier_score(
                probability[valid], failure[valid], weights[valid]
            )
            if valid.any()
            else float("nan"),
            "returned_transform_cases": int(valid.sum()),
        }

    curve = pd.DataFrame(
        {
            "coverage": rc.coverage,
            "selective_risk": rc.risk,
            "generalized_risk": rc.coverage * rc.risk,
            "threshold": rc.threshold,
        }
    )
    return EvaluationCard(summary=summary, curve=curve)
