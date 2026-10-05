from __future__ import annotations

import numpy as np

from warpaudit.evaluation.metrics import auroc, average_precision
from warpaudit.evaluation.riskcoverage import CasePopulation, risk_coverage_curve
from warpaudit.evaluation.weights import group_first_weights


def test_weighted_auroc_ties_and_one_class() -> None:
    assert auroc([0, 1, 1, 2], [0, 0, 1, 1]) == 0.875
    assert auroc([0, 1], [1, 1]) != auroc([0, 1], [1, 1])  # nan
    assert average_precision([0.1, 0.9], [0, 1]) == 1.0


def test_group_first_weights_do_not_overweight_large_groups() -> None:
    weights = group_first_weights(["a", "a", "b"], ["p1", "p2", "p3"])
    np.testing.assert_allclose(weights, [0.25, 0.25, 0.5])


def test_risk_coverage_keeps_explicit_failures_in_denominator() -> None:
    pop = CasePopulation(
        score=np.array([0.1, 0.2, np.nan]),
        failure=np.array([0, 1, 1]),
        weight=np.ones(3),
        group=np.array(["a", "b", "c"]),
        explicit_failure=np.array([False, False, True]),
    )
    curve = risk_coverage_curve(pop)
    assert curve.max_attainable_coverage == 2 / 3
    assert np.isnan(curve.risk_at(0.9))
