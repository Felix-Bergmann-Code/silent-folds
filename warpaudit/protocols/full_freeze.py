"""Machine-enforced freeze for the external factorized-stability study."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..cache.hashing import short_hash
from .freeze import FreezeError

FULL_FREEZE_FILENAME = "g2_full_study_freeze.json"


@dataclass(frozen=True)
class FullStudyFreeze:
    generated_at: str
    config_hash: str
    git_commit: str
    code_identity: str
    development_datasets: tuple[str, ...]
    external_confirmatory_datasets: tuple[str, ...]
    external_descriptive_datasets: tuple[str, ...]
    pipelines: tuple[str, ...]
    baseline_families: tuple[str, ...]
    augmented_families: tuple[str, ...]
    primary_metric: str
    min_brier_improvement: float
    min_passing_cell_fraction: float
    min_class_bearing_groups: int
    information_plan_hash: str
    bootstrap_resamples: int
    high_confidence_cutoff: float
    signed_off_by: str
    review_note: str

    @property
    def hash(self) -> str:
        payload = asdict(self)
        payload.pop("generated_at")
        return short_hash(payload)

    def to_dict(self) -> dict:
        return {**asdict(self), "freeze_hash": self.hash}


def load_full_freeze(manifests: Path) -> FullStudyFreeze | None:
    path = Path(manifests) / FULL_FREEZE_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("freeze_hash", "")
    for name in (
        "development_datasets", "external_confirmatory_datasets",
        "external_descriptive_datasets", "pipelines", "baseline_families",
        "augmented_families",
    ):
        payload[name] = tuple(payload[name])
    record = FullStudyFreeze(**payload)
    if recorded != record.hash:
        raise FreezeError(f"{path} was edited after freezing")
    return record


def require_full_freeze(manifests: Path, *, config_hash: str) -> FullStudyFreeze:
    record = load_full_freeze(manifests)
    if record is None:
        raise FreezeError("full-study outcomes are locked; run `warpaudit freeze-full-study` first")
    if record.config_hash != config_hash:
        raise FreezeError("full-study configuration differs from the reviewed freeze")
    return record
