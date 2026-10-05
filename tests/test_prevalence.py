from __future__ import annotations

import numpy as np

from warpaudit.evaluation.prevalence import standardise
from warpaudit.evaluation.riskcoverage import CasePopulation
from warpaudit.predictors.policy import FrozenPolicy


def _pop(scores, failures, prefix):
    n = len(scores)
    return CasePopulation(
        score=np.asarray(scores, float),
        failure=np.asarray(failures, float),
        weight=np.ones(n),
        group=np.asarray([f"{prefix}{i}" for i in range(n)]),
        explicit_failure=np.zeros(n, bool),
    )


def test_prevalence_standardisation_uses_source_calibration_reference() -> None:
    policy = FrozenPolicy("p", threshold=0.5, nominal_acceptance=0.7)
    calibration = _pop([0, 0, 1, 1], [0, 0, 1, 1], "c")
    source = _pop([0.1, 0.8, 0.2, 0.9], [0, 0, 1, 1], "s")
    target = _pop([0.1, 0.8, 0.2, 0.9, 0.3], [0, 0, 1, 1, 1], "t")
    result = standardise(policy, target, source, calibration)
    assert result.pi_star == 0.5 and result.available
    assert result.target.coverage_star == 0.5 * result.target.a1 + 0.5 * result.target.a0
    assert result.target.risk_star == 0.5 * result.target.a1 / result.target.coverage_star
    # Standardisation changes composition; it deliberately preserves each
    # population's class-conditional acceptance rates.
    assert result.target.a1 != result.source.a1
