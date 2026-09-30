"""The G1 freeze record and the access rule it enforces (spec §3.3, §14).

G1 is the point after which confirmatory outcomes may be read. The
specification is unambiguous that "all design changes precede test access" and
that one must not "select the primary endpoint or direction from completed test
results". A document alone cannot enforce that, so the freeze is a machine
artefact: it names the direction, folds, feature families, learner, policy, and
margins, and hashes them. Confirmatory label access is refused unless a freeze
exists whose recorded configuration still matches the one being run.

The record deliberately stores a review sign-off. A freeze written by a script
with nobody's name on it is not the reviewed gate §14 describes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..cache.hashing import short_hash

__all__ = [
    "FREEZE_FILENAME",
    "FreezeError",
    "FreezeRecord",
    "load_freeze",
    "require_confirmatory_access",
]

FREEZE_FILENAME = "g1_freeze.json"


class FreezeError(RuntimeError):
    """Raised when confirmatory access is attempted without a matching freeze."""


@dataclass
class FreezeRecord:
    """Everything fixed at G1, plus the evidence it was fixed deliberately."""

    generated_at: str
    config_hash: str
    git_commit: str
    registration_code_identity: str
    feature_code_identity: str
    primary_direction: str
    source_dataset: str
    target_dataset: str
    n_folds: int
    n_train_groups: int
    n_calibration_groups: int
    fold_hashes: tuple[str, ...]
    frozen_feature_families: tuple[str, ...]
    common_block_pipelines: tuple[str, ...]
    learner: str
    nominal_acceptance: float
    auroc_floor: float
    gap_noninferiority_margin: float
    delta_policy_margin: float
    alpha_one_sided: float
    signed_off_by: str
    review_note: str
    #: Where the frozen direction came from: an argument, the configuration, or
    #: the development-only M2 recommendation.
    direction_source: str = "argument"
    feasibility: dict[str, Any] = field(default_factory=dict)
    acknowledged_infeasible: bool = False

    @property
    def hash(self) -> str:
        payload = asdict(self)
        payload.pop("generated_at", None)
        return short_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["freeze_hash"] = self.hash
        return payload


def load_freeze(manifests: Path) -> FreezeRecord | None:
    """Read the freeze record, or ``None`` when G1 has not been passed."""
    path = Path(manifests) / FREEZE_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("freeze_hash", "")
    known = {f for f in FreezeRecord.__dataclass_fields__}
    record = FreezeRecord(
        **{
            key: (tuple(value) if key.endswith(("_hashes", "_families", "_pipelines")) else value)
            for key, value in payload.items()
            if key in known
        }
    )
    if recorded and recorded != record.hash:
        raise FreezeError(
            f"{path} has been edited since it was written: recorded freeze hash "
            f"{recorded} does not match its contents ({record.hash}). A freeze that "
            "can be quietly amended is not a freeze; rewrite it through the command "
            "and record the change in PROTOCOL_DEVIATIONS.md."
        )
    return record


def require_confirmatory_access(manifests: Path, *, config_hash: str, what: str) -> FreezeRecord:
    """Return the freeze permitting ``what``, or explain why access is refused."""
    record = load_freeze(manifests)
    if record is None:
        # The full external study has a distinct, stricter freeze. Import here
        # to avoid a module cycle: full_freeze reuses FreezeError.
        from .full_freeze import FULL_FREEZE_FILENAME, require_full_freeze

        if (Path(manifests) / FULL_FREEZE_FILENAME).is_file():
            return require_full_freeze(  # type: ignore[return-value]
                Path(manifests), config_hash=config_hash
            )
        raise FreezeError(
            f"{what} reads confirmatory outcomes, which G1 gates. No freeze record at "
            f"{Path(manifests) / FREEZE_FILENAME}. Run `warpaudit freeze` once the "
            "protocol, splits, and feature families are settled and reviewed."
        )
    if record.config_hash != config_hash:
        raise FreezeError(
            f"{what} is refused: the frozen configuration ({record.config_hash}) is not "
            f"the one being run ({config_hash}). Changing the protocol after G1 and "
            "before reading outcomes is exactly what the freeze exists to prevent. "
            "Restore the frozen configuration, or re-freeze deliberately and record "
            "the change in PROTOCOL_DEVIATIONS.md."
        )
    return record
