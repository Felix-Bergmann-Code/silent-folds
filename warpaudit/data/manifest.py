"""Dataset, provenance, and pretraining-exposure manifests (spec §4.1-4.3).

Three artefacts are produced by ``warpaudit audit-data``:

``manifests/pairs.parquet``
    One row per registration pair with namespaced ids, group identity and
    basis, original dimensions, and annotation provenance.
``manifests/development_groups.json``
    The seeded development reserve (see :mod:`warpaudit.data.development`).
``manifests/provenance.yaml``
    Dataset versions, archive checksums, download dates, access terms, and the
    checkpoint exposure register.

The exposure register records benchmark familiarity in four separate
categories, because "XFeat reports HPatches results" is evidence of evaluation
exposure and nothing more. ``training_overlap`` stays ``unknown`` until
someone actually establishes it (spec §4.3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from ..cache.store import atomic_write_text

__all__ = [
    "CheckpointExposure",
    "DatasetProvenance",
    "PairRecord",
    "ProvenanceManifest",
]

ExposureState = Literal["confirmed", "reported", "unknown", "none"]


@dataclass
class PairRecord:
    """One row of ``manifests/pairs.parquet``."""

    dataset_id: str
    dataset_version: str
    pair_id: str
    moving_image_id: str
    fixed_image_id: str
    moving_path: str
    fixed_path: str
    moving_sha256: str
    fixed_sha256: str
    moving_hw: tuple[int, int]
    fixed_hw: tuple[int, int]
    group_id: str
    group_basis: str
    component_id: str
    annotation_kind: str
    annotation_provenance: str
    annotation_paths: tuple[str, ...]
    annotation_sha256: tuple[str, ...]
    n_landmarks: int
    dataset_category: str = ""
    direction: str = "moving_to_fixed"
    eye_id: str = ""
    patient_id: str = ""
    subject_evidence: str = ""
    is_development: bool = False
    fold: int = -1
    sampling_probability: float = float("nan")

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["moving_h"], row["moving_w"] = self.moving_hw
        row["fixed_h"], row["fixed_w"] = self.fixed_hw
        row.pop("moving_hw")
        row.pop("fixed_hw")
        # Store tuples as canonical JSON strings. This remains portable across
        # Parquet engines and does not turn a multi-file annotation into an
        # ambiguous delimiter-separated path.
        import json

        row["annotation_paths"] = json.dumps(row["annotation_paths"], separators=(",", ":"))
        row["annotation_sha256"] = json.dumps(
            row["annotation_sha256"], separators=(",", ":")
        )
        return row


@dataclass
class DatasetProvenance:
    dataset_id: str
    version: str
    download_date: str = ""
    citation: str = ""
    archive_checksum: str = ""
    source_url: str = ""
    source_archives: list[dict[str, str | int]] = field(default_factory=list)
    licence: str = "[VERIFY]"
    redistribute_images: bool = False
    redistribute_derived: str = "[VERIFY]"
    n_images: int = 0
    n_image_ids: int = 0
    n_pairs: int = 0
    n_groups: int = 0
    n_exact_duplicate_sets: int = 0
    group_basis: str = "image_component"
    grouping_claim: str = ""
    unresolved_identities: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class CheckpointExposure:
    """Benchmark-exposure record for one pretrained component (§4.3)."""

    component: str
    checkpoint: str
    benchmark: str
    training_overlap: ExposureState = "unknown"
    validation_or_tuning: ExposureState = "unknown"
    reported_evaluation: ExposureState = "unknown"
    evidence: str = ""

    def summary(self) -> str:
        return (
            f"{self.component} on {self.benchmark}: training_overlap="
            f"{self.training_overlap}, validation/tuning={self.validation_or_tuning}, "
            f"reported_evaluation={self.reported_evaluation}"
        )


@dataclass
class ProvenanceManifest:
    datasets: list[DatasetProvenance] = field(default_factory=list)
    checkpoints: list[CheckpointExposure] = field(default_factory=list)
    generated_at: str = ""
    config_hash: str = ""
    code_hash: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            {
                "generated_at": self.generated_at,
                "config_hash": self.config_hash,
                "code_hash": self.code_hash,
                "datasets": [asdict(d) for d in self.datasets],
                "checkpoint_exposure": [asdict(c) for c in self.checkpoints],
                "warnings": self.warnings,
            },
            sort_keys=False,
            allow_unicode=True,
        )

    def write(self, path: str | Path) -> Path:
        return atomic_write_text(path, self.to_yaml())


def default_exposure_register() -> list[CheckpointExposure]:
    """The register as the literature currently supports it (§4.3, matrix §3).

    Every ``reported`` below is "this paper evaluates on HPatches"; none of it
    licenses a claim of training leakage.
    """
    return [
        CheckpointExposure(
            component="SuperPoint",
            checkpoint=(
                "superpoint_v1.pth sha256:"
                "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e"
            ),
            benchmark="HPatches",
            training_overlap="unknown",
            validation_or_tuning="unknown",
            reported_evaluation="reported",
            evidence="[R28] §7.2 separates MS-COCO training from HPatches evaluation",
        ),
        CheckpointExposure(
            component="LightGlue",
            checkpoint=(
                "superpoint_lightglue_v0-1_arxiv.pth sha256:"
                "6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a"
            ),
            benchmark="HPatches",
            training_overlap="unknown",
            validation_or_tuning="unknown",
            reported_evaluation="reported",
            evidence="[R30] official tooling exposes HPatches evaluation",
        ),
        CheckpointExposure(
            component="XFeat",
            checkpoint=(
                "xfeat.pt sha256:"
                "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b"
            ),
            benchmark="HPatches",
            training_overlap="unknown",
            validation_or_tuning="unknown",
            reported_evaluation="reported",
            evidence="[R29] §4.2 reports HPatches homography estimation",
        ),
    ]
