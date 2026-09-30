from __future__ import annotations

import numpy as np
import pytest

from warpaudit.evaluation.riskcoverage import CasePopulation, accept_by_threshold
from warpaudit.predictors.policy import FrozenPolicy, select_threshold


def _population(scores=(0.1, 0.2, 0.3, 0.4)) -> CasePopulation:
    return CasePopulation(
        score=np.asarray(scores),
        failure=np.array([0, 0, 1, 1], float),
        weight=np.ones(4),
        group=np.array(["a", "b", "c", "d"]),
        explicit_failure=np.zeros(4, bool),
    )


def test_zero_acceptance_risk_is_undefined() -> None:
    outcome = accept_by_threshold(_population(), -np.inf)
    assert outcome.coverage == 0 and not outcome.risk_defined and np.isnan(outcome.risk)


def test_threshold_is_frozen_after_calibration() -> None:
    policy = FrozenPolicy.from_calibration("p", _population(), nominal_acceptance=0.5)
    before = policy.threshold
    policy.apply(_population((0.8, 0.7, 0.6, 0.5)))
    assert policy.threshold == before
    policy.threshold += 0.01
    with pytest.raises(RuntimeError):
        policy.assert_unchanged()


def test_constant_scores_are_reported() -> None:
    selection = select_threshold(_population((1, 1, 1, 1)))
    assert selection.constant_scores and "identical" in selection.note
