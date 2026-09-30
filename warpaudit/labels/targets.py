"""Binary targets, operational status, and bounded loss (specification §6.2-6.3).

Two populations are kept apart throughout:

**Silent geometric failures** -- finite, evaluable returned transforms whose
error exceeds the threshold. Detector AUROC and transfer gaps concern these
valid-output cases.

**Operational failures** -- silent failures *plus* no matches, degenerate
fits, non-finite transforms, and exhausted runtime/memory limits. These enter
operational rejection and risk statistics; explicit no-output cases are
automatically rejected by every policy and are never assigned a TRE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import RegistrationStatus
from .errors import PointErrors

__all__ = [
    "BOUNDED_LOSS_CAP",
    "BOUNDED_LOSS_SENSITIVITY_CAPS",
    "CaseOutcome",
    "PRIMARY_TAU",
    "SECONDARY_TAUS",
    "bounded_loss",
    "case_outcome",
    "pixel_equivalent",
]

#: Primary benchmark failure threshold on ``tre_norm`` (spec §6.2). A
#: benchmark definition, not a clinical safety standard.
PRIMARY_TAU = 0.005
SECONDARY_TAUS: tuple[float, ...] = (0.0025, 0.01)

#: Declared benchmark penalty for the bounded continuous endpoint (§6.3).
BOUNDED_LOSS_CAP = 10.0
BOUNDED_LOSS_SENSITIVITY_CAPS: tuple[float, ...] = (5.0, 20.0)


def pixel_equivalent(tau: float, height: int, width: int) -> float:
    """Pixel equivalent of a normalised threshold for one image size.

    At FIRE's original 2912x2912, ``tau = 0.005`` is about 20.6 px.
    """
    return float(tau * np.hypot(height, width))


def bounded_loss(
    tre_norm: float | None,
    status: RegistrationStatus,
    *,
    tau: float = PRIMARY_TAU,
    cap: float = BOUNDED_LOSS_CAP,
) -> float:
    """``min(tre_norm / tau, cap)`` for valid outputs; ``cap`` for explicit failures.

    This is a declared benchmark penalty, not an imputed physical error
    (spec §6.3).
    """
    if status.is_explicit_failure:
        return float(cap)
    if tre_norm is None or not np.isfinite(tre_norm):
        return float("nan")
    return float(min(tre_norm / tau, cap))


@dataclass(frozen=True)
class CaseOutcome:
    """Everything the evaluation process knows about one attempted case."""

    status: RegistrationStatus
    tre_norm: float
    tre_px: float
    #: Silent geometric failure among valid outputs; ``nan`` when undefined.
    silent_failure: float
    #: Silent failure OR explicit no-output. Always defined for an attempt.
    operational_failure: bool
    #: Explicit failures are unavailable for acceptance (spec §6.3).
    eligible_for_acceptance: bool
    bounded_loss: float
    tau: float
    reason: str = ""

    @property
    def is_valid_output(self) -> bool:
        return self.status is RegistrationStatus.OK and np.isfinite(self.tre_norm)


def case_outcome(
    status: RegistrationStatus,
    errors: PointErrors | None,
    *,
    tau: float = PRIMARY_TAU,
    cap: float = BOUNDED_LOSS_CAP,
) -> CaseOutcome:
    """Assemble the label row for one attempted case.

    An ``ok`` status with undefined landmark error (missing or invalid ground
    truth) is *not* an operational failure -- it is a separate predeclared
    dataset exclusion, flagged through ``reason`` and an undefined
    ``silent_failure`` so that it can be removed by annotation support rather
    than counted as a model failure.
    """
    if status.is_explicit_failure:
        return CaseOutcome(
            status=status,
            tre_norm=float("nan"),
            tre_px=float("nan"),
            silent_failure=float("nan"),
            operational_failure=True,
            eligible_for_acceptance=False,
            bounded_loss=bounded_loss(None, status, tau=tau, cap=cap),
            tau=tau,
            reason=f"explicit failure: {status.value}",
        )

    if errors is None or not errors.defined:
        reason = "missing or invalid ground truth" if errors is None else errors.reason
        return CaseOutcome(
            status=status,
            tre_norm=float("nan"),
            tre_px=float("nan"),
            silent_failure=float("nan"),
            operational_failure=False,
            eligible_for_acceptance=True,
            bounded_loss=float("nan"),
            tau=tau,
            reason=reason or "undefined geometric error",
        )

    failed = bool(errors.tre_norm > tau)
    return CaseOutcome(
        status=status,
        tre_norm=float(errors.tre_norm),
        tre_px=float(errors.tre_px),
        silent_failure=float(failed),
        operational_failure=failed,
        eligible_for_acceptance=True,
        bounded_loss=bounded_loss(errors.tre_norm, status, tau=tau, cap=cap),
        tau=tau,
        reason=errors.reason,
    )
