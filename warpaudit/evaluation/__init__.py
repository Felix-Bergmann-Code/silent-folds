"""Weighted metrics, bootstrap, risk-coverage, policy estimands, report tables."""

from .bootstrap import BootstrapResult, group_bootstrap, monte_carlo_stability, refit_bootstrap
from .card import EvaluationCard, evaluate_scores, generalized_risk_area
from .inference import ComponentTest, JointClaim, decide
from .metrics import (
    BinaryCounts,
    auroc,
    average_precision,
    brier_score,
    calibration_in_the_large,
    class_bearing_groups,
    expected_calibration_error,
    prevalence,
    reliability_bins,
)
from .policy_eval import (
    PolicyComparison,
    delta_domain,
    delta_policy,
    delta_threshold,
    shared_coverage_risk_difference,
)
from .prevalence import PrevalenceDecomposition, reference_prevalence, standardise
from .riskcoverage import (
    AcceptanceOutcome,
    CasePopulation,
    RiskCoverageCurve,
    accept_by_threshold,
    risk_coverage_curve,
)
from .weights import group_first_weights, inverse_group_size_weights, row_weights

__all__ = [
    "AcceptanceOutcome",
    "BinaryCounts",
    "BootstrapResult",
    "CasePopulation",
    "ComponentTest",
    "EvaluationCard",
    "JointClaim",
    "PolicyComparison",
    "PrevalenceDecomposition",
    "RiskCoverageCurve",
    "accept_by_threshold",
    "auroc",
    "average_precision",
    "brier_score",
    "calibration_in_the_large",
    "class_bearing_groups",
    "decide",
    "delta_domain",
    "delta_policy",
    "delta_threshold",
    "expected_calibration_error",
    "evaluate_scores",
    "generalized_risk_area",
    "group_bootstrap",
    "group_first_weights",
    "inverse_group_size_weights",
    "monte_carlo_stability",
    "prevalence",
    "reference_prevalence",
    "refit_bootstrap",
    "reliability_bins",
    "risk_coverage_curve",
    "row_weights",
    "shared_coverage_risk_difference",
    "standardise",
]
