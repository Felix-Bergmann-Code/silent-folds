from __future__ import annotations

import numpy as np

from warpaudit.evaluation.bootstrap import BootstrapResult, group_bootstrap
from warpaudit.evaluation.inference import decide


def _result(name: str, estimate: float, samples: np.ndarray) -> BootstrapResult:
    return BootstrapResult(name, estimate, samples, len(samples), 0)


def test_intersection_union_requires_all_components_without_bonferroni() -> None:
    n = 200
    claim = decide(
        _result("auc", 0.8, np.full(n, 0.8)),
        _result("gap", 0.01, np.full(n, 0.01)),
        _result("policy", 0.1, np.full(n, 0.1)),
    )
    assert claim.joint_supported
    assert claim.alpha_one_sided == 0.05
    assert "no Bonferroni" in claim.summary()


def test_group_bootstrap_keeps_correlated_rows_together() -> None:
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    values = np.array([1, 1, 10, 10, 100, 100], float)

    def statistic(rows: np.ndarray) -> float:
        # Every sampled group contributes complete two-row blocks.
        selected = groups[rows]
        assert all(np.count_nonzero(selected == g) % 2 == 0 for g in set(selected))
        return float(values[rows].mean())

    result = group_bootstrap(groups, {"mean": statistic}, n_resamples=200)["mean"]
    assert result.n_invalid == 0 and result.reliable
