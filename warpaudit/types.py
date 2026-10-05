"""Core typed interfaces (specification §12.3).

These types are the contract between data loading, registration adapters,
signal computation, and evaluation. Two invariants are load-bearing and are
enforced by tests in ``tests/test_label_access.py``:

1. ``SignalContext`` has no annotation field. A feature can never reach
   landmark arrays, ground-truth homographies, evaluation masks derived from
   ground truth, errors, or failure labels (spec §7.3, §12.3).
2. ``PairInput.acquisition_meta`` is restricted by an allowlist so that
   clinical labels and evaluation-derived difficulty cannot enter a predictor
   through metadata. Patient IDs are grouping metadata, not features.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .geometry.transforms import Transform

# --------------------------------------------------------------------------
# Status vocabulary (spec §6.2, §12.3)
# --------------------------------------------------------------------------


class RegistrationStatus(str, Enum):
    """Explicit outcome of a registration attempt.

    Every attempted case keeps a row. ``OK`` means a finite, evaluable forward
    transform was returned -- it does *not* mean the geometry is correct.
    Everything else is an *explicit* operational failure: automatically
    rejected by any policy, never assigned a fabricated TRE, and always
    counted in operational denominators (spec §6.2, §6.3).
    """

    OK = "ok"
    NO_MATCHES = "no_matches"
    DEGENERATE_FIT = "degenerate_fit"
    INVALID_TRANSFORM = "invalid_transform"
    TIMEOUT = "timeout"
    OOM = "oom"
    INFRASTRUCTURE_ERROR = "infrastructure_error"

    @property
    def is_explicit_failure(self) -> bool:
        return self is not RegistrationStatus.OK

    @property
    def is_infrastructure(self) -> bool:
        """Infrastructure failures are retried under a fixed policy and are
        reported separately from scientific no-output cases (spec §6.2)."""
        return self in (
            RegistrationStatus.TIMEOUT,
            RegistrationStatus.OOM,
            RegistrationStatus.INFRASTRUCTURE_ERROR,
        )


GroupBasis = Literal["patient", "eye", "specimen", "sequence", "image_component"]

#: Ordering from strongest to weakest independence evidence. Patient grouping
#: overrides smaller eye or component groupings when known (spec §4.1).
GROUP_BASIS_PRIORITY: tuple[GroupBasis, ...] = (
    "patient",
    "eye",
    "specimen",
    "sequence",
    "image_component",
)


# --------------------------------------------------------------------------
# Coordinates (spec §5.2)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageFrame:
    """Original and working geometry of one image plus the map between them.

    ``to_working`` is the homogeneous 3x3 affine ``A`` mapping *original*
    pixel-center coordinates ``(x, y)`` to *working* pixel-center coordinates.
    Padding and cropping are represented inside ``A`` together with the
    explicit ``pad_*`` fields so that a reader never has to infer them from a
    size ratio. Never assume division by a diagonal corrects anisotropic
    stretching (spec §5.2).
    """

    image_id: str
    original_height: int
    original_width: int
    working_height: int
    working_width: int
    to_working: np.ndarray  # (3, 3) float64, original -> working
    pad_left: int = 0
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0
    resample_note: str = ""

    def __post_init__(self) -> None:
        A = np.asarray(self.to_working, dtype=np.float64)
        sizes = (
            self.original_height,
            self.original_width,
            self.working_height,
            self.working_width,
        )
        if any(int(v) <= 0 for v in sizes):
            raise ValueError(f"image dimensions must be positive, got {sizes}")
        pads = (self.pad_left, self.pad_top, self.pad_right, self.pad_bottom)
        if any(int(v) < 0 for v in pads):
            raise ValueError(f"padding must be non-negative, got {pads}")
        if self.pad_left + self.pad_right >= self.working_width:
            raise ValueError("horizontal padding leaves no image content")
        if self.pad_top + self.pad_bottom >= self.working_height:
            raise ValueError("vertical padding leaves no image content")
        if A.shape != (3, 3):
            raise ValueError(f"to_working must be 3x3, got {A.shape}")
        if not np.isfinite(A).all():
            raise ValueError("to_working must be finite")
        if abs(float(np.linalg.det(A))) < 1e-12:
            raise ValueError("to_working must be invertible")
        object.__setattr__(self, "to_working", A)

    @property
    def original_diagonal(self) -> float:
        """sqrt(H^2 + W^2) in ORIGINAL pixels -- the normalisation denominator
        for ``tre_norm`` and for every normalised positional spread (§6.1)."""
        return float(np.hypot(self.original_height, self.original_width))

    @property
    def working_diagonal(self) -> float:
        return float(np.hypot(self.working_height, self.working_width))


@dataclass(frozen=True)
class CoordinateMetadata:
    """Canonical geometry for one registration pair (spec §5.2).

    The forward map is ``T_m2f``: moving pixel centres -> fixed pixel centres.
    Published pixel endpoints are evaluated in ORIGINAL fixed-image
    coordinates via ``T_original = inv(A_f) . T_working . A_m``.
    """

    moving: ImageFrame
    fixed: ImageFrame
    pixel_center_convention: str = "integer-centre"  # centre of pixel (0,0) is (0.0, 0.0)
    interpolation: str = "bilinear"
    padding_mode: str = "zeros"

    @property
    def A_m(self) -> np.ndarray:
        return self.moving.to_working

    @property
    def A_f(self) -> np.ndarray:
        return self.fixed.to_working


# --------------------------------------------------------------------------
# Pair, annotation, registration result (spec §12.3)
# --------------------------------------------------------------------------

#: Metadata keys a feature context may see. Anything not listed here is
#: dropped by ``PairInput`` construction. Clinical labels, evaluation-derived
#: difficulty, error values, and failure labels are never admissible.
ACQUISITION_META_ALLOWLIST: frozenset[str] = frozenset(
    {
        "modality",
        "device",
        "field_of_view_deg",
        "colour_space",
        "bit_depth",
        "anatomy",
        "acquisition_session",
        "dataset_category",  # e.g. FIRE S/P/A -- descriptive stratum, not a label
    }
)

#: Keys that must never appear anywhere in metadata reaching a signal.
FORBIDDEN_META_KEYS: frozenset[str] = frozenset(
    {
        "tre_px",
        "tre_norm",
        "failure",
        "is_failure",
        "label",
        "landmarks",
        "landmarks_moving",
        "landmarks_fixed",
        "homography_gt",
        "gt_homography",
        "error",
        "loss",
        "difficulty",
        "diagnosis",
        "patient_id",
        "group_id",
    }
)


class LabelAccessError(RuntimeError):
    """Raised when ground truth would leak into a feature-visible object."""


def sanitise_acquisition_meta(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    """Apply the metadata allowlist (spec §12.3).

    Raises on an explicitly forbidden key rather than silently dropping it, so
    that a leak is a loud failure during development instead of a quiet one
    during confirmation.
    """
    if not meta:
        return {}
    offending = sorted(set(meta) & FORBIDDEN_META_KEYS)
    if offending:
        raise LabelAccessError(
            "ground-truth or grouping keys are not permitted in acquisition_meta: "
            + ", ".join(offending)
        )
    return {k: v for k, v in meta.items() if k in ACQUISITION_META_ALLOWLIST}


@dataclass(frozen=True)
class PairInput:
    """One registration pair in one canonical direction (spec §12.3)."""

    dataset_id: str
    pair_id: str
    group_id: str
    group_basis: GroupBasis
    moving_image_id: str
    fixed_image_id: str
    moving_path: Path
    fixed_path: Path
    coordinates: CoordinateMetadata
    acquisition_meta: dict[str, Any] = field(default_factory=dict)
    direction: Literal["canonical", "reverse"] = "canonical"
    condition: str = "clean"
    severity: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "moving_path", Path(self.moving_path))
        object.__setattr__(self, "fixed_path", Path(self.fixed_path))
        object.__setattr__(
            self, "acquisition_meta", sanitise_acquisition_meta(self.acquisition_meta)
        )
        if self.group_basis not in GROUP_BASIS_PRIORITY:
            raise ValueError(f"unknown group_basis {self.group_basis!r}")

    @property
    def key(self) -> str:
        """Immutable join key used by the labels and prediction tables."""
        return (
            f"{self.dataset_id}/{self.pair_id}/{self.direction}"
            f"/{self.condition}/{self.severity}"
        )


AnnotationKind = Literal["landmarks", "homography"]


@dataclass(frozen=True)
class AnnotationPayload:
    """Ground truth in ORIGINAL image coordinates.

    ``points_moving``/``points_fixed`` are (N, 2) arrays of corresponding
    landmark pixel centres. ``homography`` is the 3x3 original-frame
    moving->fixed ground-truth map (HPatches). Exactly one is populated.
    """

    points_moving: np.ndarray | None = None
    points_fixed: np.ndarray | None = None
    homography: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.homography is not None:
            H = np.asarray(self.homography, dtype=np.float64)
            if H.shape != (3, 3):
                raise ValueError("homography must be 3x3")
            object.__setattr__(self, "homography", H)
        if (self.points_moving is None) != (self.points_fixed is None):
            raise ValueError("landmark arrays must be supplied together")
        if self.points_moving is not None:
            pm = np.asarray(self.points_moving, dtype=np.float64)
            pf = np.asarray(self.points_fixed, dtype=np.float64)
            if pm.ndim != 2 or pm.shape[1] != 2 or pm.shape != pf.shape:
                raise ValueError("landmarks must be matching (N, 2) arrays")
            object.__setattr__(self, "points_moving", pm)
            object.__setattr__(self, "points_fixed", pf)
        if self.homography is None and self.points_moving is None:
            raise ValueError("an annotation payload must carry landmarks or a homography")


@dataclass(frozen=True)
class EvaluationAnnotation:
    """Restricted object. Lives only in the evaluation process (spec §12.3)."""

    dataset_id: str
    pair_id: str
    kind: AnnotationKind
    payload: AnnotationPayload
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistrationResult:
    """Output of one registration attempt (spec §12.3).

    ``forward_moving_to_fixed`` is expressed in WORKING coordinates; the
    canonical original-frame map is recovered with
    ``warpaudit.geometry.coordinates.to_original_frame``.
    """

    pipeline_id: str
    status: RegistrationStatus
    forward_moving_to_fixed: Transform | None
    matches_moving: np.ndarray | None = None
    matches_fixed: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None
    match_scores: np.ndarray | None = None
    runtime_s: float = float("nan")
    peak_vram_bytes: int = 0
    cpu_time_s: float = float("nan")
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = RegistrationStatus(self.status)
        if self.status is RegistrationStatus.OK and self.forward_moving_to_fixed is None:
            raise ValueError("status 'ok' requires a forward transform")
        if self.status.is_explicit_failure and self.forward_moving_to_fixed is not None:
            raise ValueError("explicit failures must not carry a transform")
        arrays = {
            "matches_moving": self.matches_moving,
            "matches_fixed": self.matches_fixed,
        }
        lengths: set[int] = set()
        for name, value in arrays.items():
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"{name} must be an (N, 2) array")
            setattr(self, name, arr)
            lengths.add(len(arr))
        if len(lengths) > 1:
            raise ValueError("moving and fixed correspondence arrays must have equal length")
        n = next(iter(lengths), 0)
        for name, dtype in (("inlier_mask", bool), ("match_scores", np.float64)):
            value = getattr(self, name)
            if value is None:
                continue
            arr = np.asarray(value, dtype=dtype)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
            setattr(self, name, arr)
        if np.isfinite(self.runtime_s) and self.runtime_s < 0:
            raise ValueError("runtime_s must be non-negative when defined")
        if self.peak_vram_bytes < 0:
            raise ValueError("peak_vram_bytes must be non-negative")

    @property
    def is_explicit_failure(self) -> bool:
        return self.status.is_explicit_failure

    @property
    def n_matches(self) -> int:
        return 0 if self.matches_moving is None else int(len(self.matches_moving))

    @property
    def n_inliers(self) -> int:
        return 0 if self.inlier_mask is None else int(np.count_nonzero(self.inlier_mask))


@runtime_checkable
class Registrar(Protocol):
    pipeline_id: str

    def register(self, pair: PairInput, seed: int) -> RegistrationResult: ...


# --------------------------------------------------------------------------
# Signals (spec §7, §12.3)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationGridWithoutGroundTruth:
    """A prespecified grid of fixed-image points, free of ground truth.

    Two summaries are always available (spec §5.3): one over the *fixed
    prespecified* grid and one restricted to valid support, so a predicted
    warp cannot improve its score by shrinking its own evaluation region.
    """

    points_fixed: np.ndarray  # (M, 2) working fixed-image coordinates
    valid_mask: np.ndarray  # (M,) bool -- inside documented valid support
    frame: Literal["working", "original"] = "working"
    spec: str = ""

    def __post_init__(self) -> None:
        pts = np.asarray(self.points_fixed, dtype=np.float64)
        mask = np.asarray(self.valid_mask, dtype=bool)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("grid points must be (M, 2)")
        if mask.shape != (pts.shape[0],):
            raise ValueError("valid_mask must be (M,)")
        object.__setattr__(self, "points_fixed", pts)
        object.__setattr__(self, "valid_mask", mask)

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid_mask))


@dataclass(frozen=True)
class SignalConfig:
    """Frozen numerical settings for signal computation."""

    bootstrap_B: int = 32
    perturbation_B: int = 8
    grid_size: int = 32
    seed: int = 0
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalContext:
    """Everything a feature may see. Deliberately has no annotation field."""

    pair: PairInput
    result: RegistrationResult
    reverse_estimate: RegistrationResult | None
    auxiliary_results: Mapping[str, RegistrationResult]
    grid: EvaluationGridWithoutGroundTruth
    config: SignalConfig
    images: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureValue:
    """One scalar feature with its availability reason, unit, and cost."""

    value: float
    available: bool = True
    reason: str = ""
    unit: str = "dimensionless"
    definition_version: str = "1"


@dataclass
class FeatureBundle:
    """Features produced by one family for one case, plus measured cost."""

    family: str
    values: dict[str, FeatureValue] = field(default_factory=dict)
    wall_time_s: float = 0.0
    cpu_time_s: float = 0.0
    peak_vram_bytes: int = 0

    def add(
        self,
        name: str,
        value: float | None,
        *,
        unit: str = "dimensionless",
        reason: str = "",
        definition_version: str = "1",
    ) -> None:
        """Record a value, or an explicit missing entry when ``value`` is None
        or non-finite. Missingness is never silently imputed here; a
        source-fitted imputer plus availability indicators handles it later
        (spec §7.3)."""
        ok = value is not None and np.isfinite(value)
        self.values[name] = FeatureValue(
            value=float(value) if ok else float("nan"),
            available=bool(ok),
            reason=reason if not ok else "",
            unit=unit,
            definition_version=definition_version,
        )

    def as_row(self, prefix: str = "") -> dict[str, float]:
        row: dict[str, float] = {}
        for name, fv in self.values.items():
            key = f"{prefix}{name}"
            row[key] = fv.value
            row[f"{key}__available"] = float(fv.available)
        return row


@runtime_checkable
class Signal(Protocol):
    family: str

    def compute(self, ctx: SignalContext) -> FeatureBundle: ...


__all__ = [
    "ACQUISITION_META_ALLOWLIST",
    "AnnotationKind",
    "AnnotationPayload",
    "CoordinateMetadata",
    "EvaluationAnnotation",
    "EvaluationGridWithoutGroundTruth",
    "FORBIDDEN_META_KEYS",
    "FeatureBundle",
    "FeatureValue",
    "GROUP_BASIS_PRIORITY",
    "GroupBasis",
    "ImageFrame",
    "LabelAccessError",
    "PairInput",
    "RegistrationResult",
    "RegistrationStatus",
    "Registrar",
    "Signal",
    "SignalConfig",
    "SignalContext",
    "sanitise_acquisition_meta",
]
