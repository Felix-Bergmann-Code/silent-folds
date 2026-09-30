"""Risk, coverage, and the risk-coverage curve (specification §6.3, §9.2, §10.1).

Definitions fixed here and used by every policy comparison:

* **Coverage** denominator is *all eligible attempted cases*, explicit
  operational failures included. Those failures are unavailable for
  acceptance, so the curve carries a **maximum attainable coverage** marker
  instead of silently dropping them.
* **Risk** is the group-weighted binary failure rate among accepted cases.
  At zero acceptance risk is **undefined**, not zero.
* Ties are handled deterministically and independently of labels. Under the
  default ``conservative_reject`` rule a threshold falling inside a block of
  equal scores rejects the whole block, so no label information can leak
  through tie ordering.
* Area under the risk-coverage curve is integrated with an explicit step
  convention over the *attainable* coverage interval, and the interval is
  reported alongside the area so runs with different explicit-failure rates
  are never compared through an unexplained number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "AcceptanceOutcome",
    "CasePopulation",
    "RiskCoverageCurve",
    "accept_by_threshold",
    "risk_coverage_curve",
]


@dataclass(frozen=True)
class CasePopulation:
    """Every attempted case in one evaluation cell.

    ``score`` is a risk score (higher = more likely failure); a case is
    accepted when ``score <= threshold``. Explicit failures may carry ``nan``
    scores: they are rejected regardless.
    """

    score: np.ndarray
    failure: np.ndarray  # 1.0 failure, 0.0 success; nan only where undefined
    weight: np.ndarray
    group: np.ndarray
    explicit_failure: np.ndarray  # bool
    loss: np.ndarray | None = None  # bounded geometric loss (§6.3)

    def __post_init__(self) -> None:
        score = np.asarray(self.score, dtype=np.float64)
        if score.ndim != 1:
            raise ValueError("score must be a 1-D array")
        n = len(score)
        object.__setattr__(self, "score", score)
        for name in ("failure", "weight", "group", "explicit_failure"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or len(value) != n:
                raise ValueError(f"{name} must have length {n}")
        object.__setattr__(self, "failure", np.asarray(self.failure, dtype=np.float64))
        object.__setattr__(self, "weight", np.asarray(self.weight, dtype=np.float64))
        object.__setattr__(self, "group", np.asarray(self.group))
        object.__setattr__(self, "explicit_failure", np.asarray(self.explicit_failure, dtype=bool))
        if self.loss is not None:
            loss = np.asarray(self.loss, dtype=np.float64)
            if loss.ndim != 1 or len(loss) != n:
                raise ValueError(f"loss must have length {n}")
            object.__setattr__(self, "loss", loss)
        if np.any(self.weight < 0) or not np.isfinite(self.weight).all():
            raise ValueError("weights must be finite and non-negative")
        if self.total_weight <= 0:
            raise ValueError("weights must have a positive sum")
        defined_labels = self.failure[np.isfinite(self.failure)]
        if not np.isin(defined_labels, (0.0, 1.0)).all():
            raise ValueError("failure labels must be 0, 1, or nan")
        # An explicit failure is by definition an operational failure (§6.2).
        bad = self.explicit_failure & (self.failure != 1.0)
        if bad.any():
            raise ValueError(
                "explicit failures must carry failure=1.0; " f"{int(bad.sum())} row(s) violate this"
            )

    def __len__(self) -> int:
        return int(len(self.score))

    @property
    def acceptable(self) -> np.ndarray:
        """Cases a policy is permitted to accept: valid outputs with a score."""
        return (~self.explicit_failure) & np.isfinite(self.score)

    @property
    def total_weight(self) -> float:
        return float(self.weight.sum())

    @property
    def max_attainable_coverage(self) -> float:
        """Weight share of cases that could ever be accepted (§6.3)."""
        total = self.total_weight
        return float(self.weight[self.acceptable].sum() / total) if total > 0 else float("nan")

    @property
    def explicit_failure_fraction(self) -> float:
        total = self.total_weight
        return (
            float(self.weight[self.explicit_failure].sum() / total) if total > 0 else float("nan")
        )

    @property
    def operational_prevalence(self) -> float:
        """Failure rate over ALL attempted cases, explicit failures included."""
        total = self.total_weight
        if np.any(~np.isfinite(self.failure)):
            return float("nan")
        return float((self.weight * self.failure).sum() / total) if total > 0 else float("nan")

    def subset(self, mask: np.ndarray) -> CasePopulation:
        mask = np.asarray(mask, dtype=bool)
        return CasePopulation(
            score=self.score[mask],
            failure=self.failure[mask],
            weight=self.weight[mask],
            group=self.group[mask],
            explicit_failure=self.explicit_failure[mask],
            loss=None if self.loss is None else self.loss[mask],
        )


@dataclass(frozen=True)
class AcceptanceOutcome:
    """What one frozen threshold did to one population."""

    threshold: float
    coverage: float
    risk: float
    accepted_weight: float
    n_accepted: int
    n_accepted_groups: int
    accepted_loss: float
    accept_given_failure: float  # a1 = P(accept | failure)
    accept_given_success: float  # a0 = P(accept | success)
    max_attainable_coverage: float
    explicit_failure_fraction: float
    risk_defined: bool
    note: str = ""

    def as_row(self) -> dict[str, float | int | bool | str]:
        return {
            "threshold": self.threshold,
            "realised_coverage": self.coverage,
            "accepted_failure_risk": self.risk,
            "accepted_weight": self.accepted_weight,
            "n_accepted": self.n_accepted,
            "n_accepted_groups": self.n_accepted_groups,
            "accepted_bounded_loss": self.accepted_loss,
            "accept_given_failure_a1": self.accept_given_failure,
            "accept_given_success_a0": self.accept_given_success,
            "max_attainable_coverage": self.max_attainable_coverage,
            "explicit_failure_fraction": self.explicit_failure_fraction,
            "risk_defined": self.risk_defined,
            "note": self.note,
        }


def accept_by_threshold(
    population: CasePopulation,
    threshold: float,
    *,
    tie_rule: str = "conservative_reject",
    tie_seed: int = 0,
    tie_target_coverage: float | None = None,
) -> AcceptanceOutcome:
    """Apply a frozen threshold and report the full acceptance outcome.

    ``tie_rule``
        ``conservative_reject`` (default) accepts ``score <= threshold``;
        a block of equal scores is therefore accepted or rejected together.
        ``random_reference`` breaks a fully tied population with a fixed
        seeded permutation to reach ``tie_target_coverage``. That arm exists
        only to report the random reference required by §9.2 for constant
        scores; it is never the deployed rule.
    """
    acceptable = population.acceptable
    w = population.weight
    total = population.total_weight

    if tie_rule == "random_reference":
        if tie_target_coverage is None:
            raise ValueError("random_reference tie rule needs tie_target_coverage")
        rng = np.random.default_rng(tie_seed)
        order = rng.permutation(np.flatnonzero(acceptable))
        cumulative = np.cumsum(w[order]) / total if total > 0 else np.zeros(len(order))
        keep = order[cumulative <= tie_target_coverage]
        accepted = np.zeros(len(population), dtype=bool)
        accepted[keep] = True
        note = f"random tie reference at target coverage {tie_target_coverage:.3f}"
    elif tie_rule == "conservative_reject":
        accepted = acceptable & (population.score <= threshold)
        note = ""
    else:
        raise ValueError(f"unknown tie rule {tie_rule!r}")

    acc_w = float(w[accepted].sum())
    coverage = acc_w / total if total > 0 else float("nan")
    failure = population.failure

    if acc_w > 0:
        risk = float((w[accepted] * failure[accepted]).sum() / acc_w)
        loss = (
            float((w[accepted] * population.loss[accepted]).sum() / acc_w)
            if population.loss is not None
            else float("nan")
        )
        risk_defined = True
    else:
        # At zero acceptance risk is undefined rather than zero (§9.2).
        risk, loss, risk_defined = float("nan"), float("nan"), False
        note = (note + "; " if note else "") + "zero acceptance: risk undefined"

    fail_mask = failure == 1.0
    succ_mask = failure == 0.0
    w_fail, w_succ = float(w[fail_mask].sum()), float(w[succ_mask].sum())
    a1 = float(w[accepted & fail_mask].sum() / w_fail) if w_fail > 0 else float("nan")
    a0 = float(w[accepted & succ_mask].sum() / w_succ) if w_succ > 0 else float("nan")

    return AcceptanceOutcome(
        threshold=float(threshold),
        coverage=coverage,
        risk=risk,
        accepted_weight=acc_w,
        n_accepted=int(accepted.sum()),
        n_accepted_groups=int(len(set(population.group[accepted].tolist()))),
        accepted_loss=loss,
        accept_given_failure=a1,
        accept_given_success=a0,
        max_attainable_coverage=population.max_attainable_coverage,
        explicit_failure_fraction=population.explicit_failure_fraction,
        risk_defined=risk_defined,
        note=note,
    )


@dataclass
class RiskCoverageCurve:
    coverage: np.ndarray
    risk: np.ndarray
    threshold: np.ndarray
    max_attainable_coverage: float
    min_attainable_coverage: float
    explicit_failure_fraction: float
    convention: str = "right-continuous step over attainable coverage"
    notes: list[str] = field(default_factory=list)

    def risk_at(self, requested_coverage: float) -> float:
        """Risk at the largest achieved coverage not exceeding the request.

        Returns ``nan`` when the request exceeds the attainable maximum, which
        is the honest answer: that coverage cannot be delivered at all.
        """
        if requested_coverage > self.max_attainable_coverage + 1e-12:
            return float("nan")
        eligible = self.coverage <= requested_coverage + 1e-12
        if not eligible.any():
            return float("nan")
        return float(self.risk[np.flatnonzero(eligible)[-1]])

    def area(self) -> tuple[float, float, float]:
        """``(area, interval_lower, interval_upper)`` over attainable coverage.

        The interval is returned with the area so two runs with different
        explicit-failure rates cannot be compared through the number alone
        (§10.1).
        """
        lo, hi = self.min_attainable_coverage, self.max_attainable_coverage
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return float("nan"), lo, hi
        keep = np.isfinite(self.risk)
        cov, rk = self.coverage[keep], self.risk[keep]
        if len(cov) < 2:
            return float("nan"), lo, hi
        widths = np.diff(cov)
        area = float(np.sum(widths * rk[1:]))  # right-continuous step
        return area / (cov[-1] - cov[0]), float(cov[0]), float(cov[-1])

    def generalized_area(self) -> tuple[float, float, float]:
        """Empirical area under generalized risk over attainable coverage.

        Generalized risk at an operating point is ``coverage * risk``.  The
        area is enumerated from the same case-level score thresholds and uses
        the same right-continuous convention as :meth:`area`; it is not
        reconstructed from prevalence and AUROC.  Unlike AURC, AUGRC is not
        normalized by the width of the attainable coverage interval.
        """

        lo, hi = self.min_attainable_coverage, self.max_attainable_coverage
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return float("nan"), lo, hi
        generalized = self.coverage * self.risk
        keep = np.isfinite(generalized)
        cov, gr = self.coverage[keep], generalized[keep]
        if len(cov) < 2:
            return float("nan"), lo, hi
        area = float(np.sum(np.diff(cov) * gr[1:]))
        return area, float(cov[0]), float(cov[-1])


def risk_coverage_curve(
    population: CasePopulation, *, tie_rule: str = "conservative_reject"
) -> RiskCoverageCurve:
    """Retrospective risk-coverage curve over eligible attempted cases.

    All eligible valid outputs are ranked by score; explicit failures stay
    rejected and remain in the denominator.
    """
    acceptable = population.acceptable
    idx = np.flatnonzero(acceptable)
    total = population.total_weight
    notes: list[str] = []

    if len(idx) == 0 or total <= 0:
        return RiskCoverageCurve(
            coverage=np.zeros(1),
            risk=np.array([np.nan]),
            threshold=np.array([np.nan]),
            max_attainable_coverage=0.0,
            min_attainable_coverage=0.0,
            explicit_failure_fraction=population.explicit_failure_fraction,
            notes=["no acceptable case: the curve is empty and risk is undefined"],
        )

    order = idx[np.argsort(population.score[idx], kind="mergesort")]
    s = population.score[order]
    w = population.weight[order]
    f = population.failure[order]

    # Collapse tied score blocks: a threshold can never split them (§9.2).
    boundaries = np.flatnonzero(np.diff(s)) + 1
    ends = np.concatenate((boundaries, [len(s)]))
    if len(ends) == 1:
        notes.append(
            "all acceptable scores are identical; only trivial coverage levels exist. "
            "Report the seeded random reference of §9.2 alongside this curve."
        )

    cum_w = np.cumsum(w)
    cum_fw = np.cumsum(w * f)

    coverage = np.concatenate(([0.0], cum_w[ends - 1] / total))
    with np.errstate(invalid="ignore", divide="ignore"):
        risk = np.concatenate(([np.nan], cum_fw[ends - 1] / cum_w[ends - 1]))
    threshold = np.concatenate(([-np.inf], s[ends - 1]))

    return RiskCoverageCurve(
        coverage=coverage,
        risk=risk,
        threshold=threshold,
        max_attainable_coverage=float(coverage[-1]),
        min_attainable_coverage=float(coverage[1]) if len(coverage) > 1 else 0.0,
        explicit_failure_fraction=population.explicit_failure_fraction,
        notes=notes,
    )
