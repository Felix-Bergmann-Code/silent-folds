"""Configuration schema and validation (specification §3.3, §12.1).

Configuration is part of the scientific contract, not a convenience layer.
Anything that would change a confirmatory result -- the failure threshold, the
nominal acceptance coverage, the noninferiority margin, the development
fraction, the feature families in the common block -- lives here so it can be
frozen, hashed, and cited in ``PREREGISTRATION.md``.

Validation is deliberately strict:

* unknown keys are errors, so a silently ignored typo cannot change a run;
* values are checked against the ranges the specification fixes;
* every error is collected and reported together, because a half-validated
  config wastes a GPU day.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

from .cache.hashing import short_hash
from .types import GROUP_BASIS_PRIORITY

__all__ = [
    "Config",
    "ConfigError",
    "DatasetConfig",
    "FeatureConfig",
    "GeometryConfig",
    "InferenceConfig",
    "FullStudyConfig",
    "LabelConfig",
    "LearnerConfig",
    "PathsConfig",
    "PipelineConfig",
    "PolicyConfig",
    "SplitConfig",
    "load_config",
    "validate_config",
]


class ConfigError(ValueError):
    """Raised with every validation problem found, not just the first."""


def _require(errors: list[str], cond: bool, message: str) -> None:
    if not cond:
        errors.append(message)


def _unknown_keys(cls, data: Mapping[str, Any], where: str, errors: list[str]) -> None:
    known = {f.name for f in fields(cls)}
    for key in data:
        if key not in known:
            errors.append(f"{where}: unknown key {key!r} (known: {sorted(known)})")


def _build(cls, data: Mapping[str, Any] | None, where: str, errors: list[str]):
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        errors.append(f"{where}: expected a mapping, got {type(data).__name__}")
        return cls()
    data = dict(data)
    _unknown_keys(cls, data, where, errors)
    known = {f.name for f in fields(cls)}
    defaults = cls()
    clean: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            continue
        default = getattr(defaults, key)
        if isinstance(default, bool) and not isinstance(value, bool):
            errors.append(f"{where}.{key}: expected bool, got {type(value).__name__}")
            continue
        if isinstance(default, int) and not isinstance(default, bool):
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(f"{where}.{key}: expected int, got {type(value).__name__}")
                continue
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, int | float):
                errors.append(f"{where}.{key}: expected number, got {type(value).__name__}")
                continue
            value = float(value)
        elif isinstance(default, str) and not isinstance(value, str):
            errors.append(f"{where}.{key}: expected string, got {type(value).__name__}")
            continue
        clean[key] = value
    return cls(**clean)


# --------------------------------------------------------------------------


@dataclass
class PathsConfig:
    data_root: str = "data"
    cache_root: str = "cache"
    manifests: str = "manifests"
    reports: str = "reports"
    figures: str = "paper/figures"
    tables: str = "paper/tables"

    def resolve(self, base: Path) -> dict[str, Path]:
        return {f.name: (base / getattr(self, f.name)) for f in fields(self)}


@dataclass
class DatasetConfig:
    """One dataset and the *evidence* behind its grouping (spec §4.1)."""

    id: str = ""
    version: str = ""
    root: str = ""
    role: str = "pilot"  # pilot | expansion | exploratory | secondary
    group_basis: str = "image_component"
    #: True only when recovered subject identity actually supports the claim.
    #: An engineering assertion verifies recorded groups, not inferred identity.
    patient_ids_available: bool = False
    annotation_kind: str = "landmarks"
    source_url: str = ""
    archive_path: str = ""
    archive_sha256: str = ""
    archive_size_bytes: int = 0
    parent_source_url: str = ""
    parent_archive_path: str = ""
    parent_archive_sha256: str = ""
    parent_archive_md5: str = ""
    parent_archive_size_bytes: int = 0
    parent_licence: str = ""
    download_date: str = ""
    citation: str = ""
    licence: str = "[VERIFY]"
    access_note: str = "[VERIFY]"
    redistribute_images: bool = False
    redistribute_derived: str = "[VERIFY]"
    expected_images: int = 0
    expected_pairs: int = 0
    #: Excluded from headline claims and primary pooled-source training (§4.3).
    secondary_only: bool = False


@dataclass
class PipelineConfig:
    id: str = ""
    version: str = ""
    matcher: str = ""
    checkpoint: str = "[VERIFY]"
    transform_family: str = "homography"
    fit: str = "warpaudit_ransac_dlt"
    ransac_threshold_px: float = 3.0
    max_iters: int = 10000
    confidence: float = 0.9999
    min_matches: int = 4
    in_common_block: bool = True
    environment: str = "match"
    # ``entrypoint`` is intended for an adapter installed in the current
    # interpreter. ``subprocess`` keeps the matcher stack in an isolated
    # environment and exchanges only the versioned JSON protocol.
    adapter: str = "entrypoint"
    command: tuple[str, ...] = ()
    timeout_s: float = 600.0
    options: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass
class GeometryConfig:
    working_long_edge: int = 1024
    pad_to_square: bool = False
    pixel_center_convention: str = "integer-centre"
    interpolation: str = "bilinear"
    padding_mode: str = "zeros"
    grid_size: int = 32


@dataclass
class LabelConfig:
    tau_primary: float = 0.005
    tau_secondary: tuple[float, ...] = (0.0025, 0.01)
    bounded_loss_cap: float = 10.0
    bounded_loss_sensitivity_caps: tuple[float, ...] = (5.0, 20.0)
    hpatches_min_valid_points: int = 20
    r8_landmark_success_px: float = 12.5


@dataclass
class FeatureConfig:
    families: tuple[str, ...] = ("A", "B", "E1")
    bootstrap_B: int = 32
    perturbation_B: int = 8
    perturbation_B_sensitivity: tuple[int, ...] = (4, 16)
    #: Development-frozen gates. Values are prospective defaults confirmed or
    #: revised on development data before freeze (spec §7.1a, §7.2a).
    e1_repeatability_tolerance: float = 1e-4
    e1_seed_dominance_max_ratio: float = 0.5
    e1_min_error_spearman: float = 0.15
    cycle_tolerance: float = 1e-4
    fallback_families_if_e1_degenerate: tuple[str, ...] = ("A", "B")


@dataclass
class SplitConfig:
    development_fraction: float = 0.20
    seed: int = 20260907
    n_outer_folds: int = 5
    min_outer_folds: int = 3
    #: Screening rule from §10.3, not a power guarantee.
    low_information_group_threshold: int = 10
    conformal_min_calibration_groups: int = 19


@dataclass
class PolicyConfig:
    nominal_acceptance: float = 0.70
    #: Deterministic, label-independent tie handling (spec §9.2).
    tie_rule: str = "conservative_reject"
    requested_coverages: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9)
    calibration_method: str = "platt"
    reliability_bins: int = 5
    safety_accepted_failure_rate: float = 0.05


@dataclass
class LearnerConfig:
    primary: str = "logistic"
    secondary: str = "lightgbm"
    logistic_C: tuple[float, ...] = (0.1, 1.0, 10.0)
    lightgbm_leaves: tuple[int, ...] = (7, 15)
    lightgbm_min_child_samples: tuple[int, ...] = (20, 50)
    forest_n_estimators: int = 200
    forest_max_depth: int = 6
    sample_weighting: str = "inverse_group_size"


@dataclass
class InferenceConfig:
    """Margins and multiplicity rule for the H4 joint claim (§3.2, §10.2-10.3)."""

    auroc_floor: float = 0.70
    gap_noninferiority_margin: float = 0.05
    delta_policy_margin: float = 0.05
    alpha_one_sided: float = 0.05
    conditional_bootstrap: int = 10_000
    refit_bootstrap: int = 1_000
    max_invalid_resample_fraction: float = 0.05
    #: Intersection-union: all three component nulls must be rejected, so no
    #: Bonferroni division is applied (spec §10.2).
    multiplicity_rule: str = "intersection_union"


@dataclass
class PlanningConfig:
    """Feasibility-screening parameters for the M2 direction decision (§10.3).

    Every field here affects *reports only*: the direction preflight, its
    projections, and its recommendation. None of them enters registration,
    label, or feature computation, which is why :attr:`Config.hash` -- the
    cache reuse contract -- deliberately excludes this section. A parameter
    that would change any cached scientific value must never be added here;
    put it in the section that owns that computation so the caches invalidate.
    """

    #: Class-bearing and accepted-group minima a fold must clear to be usable.
    min_class_bearing_groups: int = 5
    min_accepted_groups: int = 5
    #: Posterior probability required of each minimum before a direction is
    #: called feasible. This is a screening rule, not a power guarantee.
    feasibility_probability: float = 0.90
    #: Design scenarios for the H4 joint bound, as
    #: ``name: [target_auroc, gap_auc, delta_policy]``. They are assumptions,
    #: never estimates of unseen confirmatory effects.
    scenarios: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "optimistic": (0.85, 0.02, 0.10),
            "central": (0.78, 0.03, 0.07),
            "pessimistic": (0.72, 0.05, 0.04),
        }
    )
    #: Paired correlation assumed between the two target scores and between
    #: the two accepted-policy risks.
    paired_correlation: float = 0.5
    #: Accepted-group failure risk the policy contrast is centred on.
    accepted_failure_risk: float = 0.15
    n_simulations: int = 20_000
    seed: int = 20260907


@dataclass
class FullStudyConfig:
    """Frozen design choices for the external factorized-stability study."""

    analysis_mode: str = "confirmatory"
    development_datasets: tuple[str, ...] = ("FIRE", "COph100")
    external_confirmatory_datasets: tuple[str, ...] = ()
    external_descriptive_datasets: tuple[str, ...] = ()
    baseline_families: tuple[str, ...] = ("A", "B")
    augmented_families: tuple[str, ...] = ("A", "B", "E1", "E2")
    primary_metric: str = "brier_improvement"
    min_brier_improvement: float = 0.01
    min_passing_cell_fraction: float = 0.50
    min_class_bearing_groups: int = 10
    min_information_probability: float = 0.80
    bootstrap_resamples: int = 2000
    high_confidence_cutoff: float = 0.20


@dataclass
class Config:
    project: str = "warpaudit"
    spec_revision: str = "2.2"
    tier: str = "pilot"  # pilot | focused | expanded | full
    #: Selected at M2 by information feasibility, never assumed (§3.3).
    primary_direction: str | None = None
    direction_priority: tuple[str, ...] = ("COph100->FIRE", "FIRE->COph100")
    paths: PathsConfig = field(default_factory=PathsConfig)
    datasets: tuple[DatasetConfig, ...] = ()
    pipelines: tuple[PipelineConfig, ...] = ()
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    learners: LearnerConfig = field(default_factory=LearnerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    #: Report-only; excluded from :attr:`hash`. See :class:`PlanningConfig`.
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    full_study: FullStudyConfig = field(default_factory=FullStudyConfig)
    source_path: str = ""

    # -- accessors -------------------------------------------------------

    def dataset(self, dataset_id: str) -> DatasetConfig:
        for ds in self.datasets:
            if ds.id == dataset_id:
                return ds
        raise KeyError(f"no dataset {dataset_id!r} in configuration")

    def pipeline(self, pipeline_id: str) -> PipelineConfig:
        for pl in self.pipelines:
            if pl.id == pipeline_id:
                return pl
        raise KeyError(f"no pipeline {pipeline_id!r} in configuration")

    @property
    def common_block(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.pipelines if p.in_common_block)

    @property
    def primary_source_pool(self) -> tuple[str, ...]:
        """Datasets eligible as a primary source (HPatches excluded, §4.3)."""
        return tuple(d.id for d in self.datasets if not d.secondary_only)

    def as_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    #: Sections excluded from :attr:`hash`. ``source_path`` is machine-specific;
    #: ``planning`` is report-only by construction (see :class:`PlanningConfig`).
    #: Adding to this set invalidates nothing and hides nothing only while every
    #: excluded section stays outside cached scientific computation.
    HASH_EXCLUDED_SECTIONS: ClassVar[frozenset[str]] = frozenset({"source_path", "planning"})

    @property
    def hash(self) -> str:
        """The cache reuse contract: every value that can change a cached result."""
        payload = self.as_dict()
        for section in self.HASH_EXCLUDED_SECTIONS:
            payload.pop(section, None)
        # The additive full-study schema must not invalidate completed pilot
        # caches. It enters the identity only when that execution mode is used.
        if self.tier != "full":
            payload.pop("full_study", None)
        return short_hash(payload)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list | tuple):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------


def _coerce(data: Mapping[str, Any], errors: list[str]) -> Config:
    data = dict(data)
    _unknown_keys(Config, data, "root", errors)

    project = data.get("project", "warpaudit")
    if not isinstance(project, str):
        errors.append(f"root.project: expected string, got {type(project).__name__}")
        project = "warpaudit"
    tier = data.get("tier", "pilot")
    if not isinstance(tier, str):
        errors.append(f"root.tier: expected string, got {type(tier).__name__}")
        tier = "pilot"
    primary_direction = data.get("primary_direction")
    if primary_direction is not None and not isinstance(primary_direction, str):
        errors.append(
            f"root.primary_direction: expected string or null, got {type(primary_direction).__name__}"
        )
        primary_direction = None
    direction_raw = data.get("direction_priority")
    if direction_raw is None:
        direction_priority = ("COph100->FIRE", "FIRE->COph100")
    elif isinstance(direction_raw, list | tuple) and all(
        isinstance(value, str) for value in direction_raw
    ):
        direction_priority = tuple(direction_raw)
    else:
        errors.append("root.direction_priority: expected a sequence of strings")
        direction_priority = ("COph100->FIRE", "FIRE->COph100")

    datasets = tuple(
        _build(DatasetConfig, d, f"datasets[{i}]", errors)
        for i, d in enumerate(data.get("datasets") or ())
    )
    pipelines = tuple(
        _build(PipelineConfig, p, f"pipelines[{i}]", errors)
        for i, p in enumerate(data.get("pipelines") or ())
    )

    def coerce_tuple(value: Any, cast, where: str, default: tuple) -> tuple:
        if isinstance(value, str) or not isinstance(value, list | tuple):
            errors.append(f"{where}: expected a sequence")
            return default
        try:
            return tuple(cast(item) for item in value)
        except (TypeError, ValueError):
            errors.append(f"{where}: contains a value of the wrong type")
            return default

    for i, pipeline in enumerate(pipelines):
        pipeline.command = coerce_tuple(
            pipeline.command,
            str,
            f"pipelines[{i}].command",
            PipelineConfig().command,
        )
        if not isinstance(pipeline.options, dict):
            errors.append(f"pipelines[{i}].options: expected a mapping")
            pipeline.options = {}
        if not isinstance(pipeline.provenance, dict) or not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in pipeline.provenance.items()
        ):
            errors.append(f"pipelines[{i}].provenance: expected a string-to-string mapping")
            pipeline.provenance = {}

    labels = _build(LabelConfig, data.get("labels"), "labels", errors)
    labels.tau_secondary = coerce_tuple(
        labels.tau_secondary, float, "labels.tau_secondary", LabelConfig().tau_secondary
    )
    labels.bounded_loss_sensitivity_caps = coerce_tuple(
        labels.bounded_loss_sensitivity_caps,
        float,
        "labels.bounded_loss_sensitivity_caps",
        LabelConfig().bounded_loss_sensitivity_caps,
    )
    features = _build(FeatureConfig, data.get("features"), "features", errors)
    features.families = coerce_tuple(
        features.families, str, "features.families", FeatureConfig().families
    )
    features.perturbation_B_sensitivity = coerce_tuple(
        features.perturbation_B_sensitivity,
        int,
        "features.perturbation_B_sensitivity",
        FeatureConfig().perturbation_B_sensitivity,
    )
    features.fallback_families_if_e1_degenerate = coerce_tuple(
        features.fallback_families_if_e1_degenerate,
        str,
        "features.fallback_families_if_e1_degenerate",
        FeatureConfig().fallback_families_if_e1_degenerate,
    )
    policy = _build(PolicyConfig, data.get("policy"), "policy", errors)
    policy.requested_coverages = coerce_tuple(
        policy.requested_coverages,
        float,
        "policy.requested_coverages",
        PolicyConfig().requested_coverages,
    )
    learners = _build(LearnerConfig, data.get("learners"), "learners", errors)
    learners.logistic_C = coerce_tuple(
        learners.logistic_C, float, "learners.logistic_C", LearnerConfig().logistic_C
    )
    learners.lightgbm_leaves = coerce_tuple(
        learners.lightgbm_leaves,
        int,
        "learners.lightgbm_leaves",
        LearnerConfig().lightgbm_leaves,
    )
    learners.lightgbm_min_child_samples = coerce_tuple(
        learners.lightgbm_min_child_samples,
        int,
        "learners.lightgbm_min_child_samples",
        LearnerConfig().lightgbm_min_child_samples,
    )

    planning = _build(PlanningConfig, data.get("planning"), "planning", errors)
    raw_scenarios = planning.scenarios
    if not isinstance(raw_scenarios, Mapping):
        errors.append("planning.scenarios: expected a mapping of name to three numbers")
        planning.scenarios = PlanningConfig().scenarios
    else:
        scenarios: dict[str, tuple[float, float, float]] = {}
        for name, values in raw_scenarios.items():
            triple = coerce_tuple(values, float, f"planning.scenarios.{name}", ())
            if len(triple) != 3:
                errors.append(
                    f"planning.scenarios.{name}: expected [target_auroc, gap_auc, "
                    f"delta_policy], got {len(triple)} value(s)"
                )
                continue
            scenarios[str(name)] = (triple[0], triple[1], triple[2])
        planning.scenarios = scenarios or PlanningConfig().scenarios

    full_study = _build(FullStudyConfig, data.get("full_study"), "full_study", errors)
    for name in (
        "development_datasets",
        "external_confirmatory_datasets",
        "external_descriptive_datasets",
        "baseline_families",
        "augmented_families",
    ):
        setattr(
            full_study,
            name,
            coerce_tuple(
                getattr(full_study, name),
                str,
                f"full_study.{name}",
                getattr(FullStudyConfig(), name),
            ),
        )

    return Config(
        project=project,
        spec_revision=str(data.get("spec_revision", "2.2")),
        tier=tier,
        primary_direction=primary_direction,
        direction_priority=direction_priority,
        paths=_build(PathsConfig, data.get("paths"), "paths", errors),
        datasets=datasets,
        pipelines=pipelines,
        geometry=_build(GeometryConfig, data.get("geometry"), "geometry", errors),
        labels=labels,
        features=features,
        splits=_build(SplitConfig, data.get("splits"), "splits", errors),
        policy=policy,
        learners=learners,
        inference=_build(InferenceConfig, data.get("inference"), "inference", errors),
        planning=planning,
        full_study=full_study,
        source_path=str(data.get("source_path", "")),
    )


VALID_FAMILIES = frozenset("A B C D E1 E2 F G".split())
VALID_TIERS = frozenset({"pilot", "focused", "expanded", "full"})
VALID_DATASET_ROLES = frozenset({
    "pilot", "expansion", "exploratory", "secondary",
    "development", "external_confirmatory", "external_descriptive",
})
VALID_ANNOTATIONS = frozenset({"landmarks", "homography"})


def validate_config(cfg: Config) -> list[str]:
    """Return every scientific and structural problem found."""
    e: list[str] = []

    _require(e, cfg.tier in VALID_TIERS, f"tier must be one of {sorted(VALID_TIERS)}")
    _require(e, len(cfg.datasets) >= 1, "at least one dataset must be configured")
    _require(e, len(cfg.pipelines) >= 1, "at least one pipeline must be configured")

    ids = [d.id for d in cfg.datasets]
    _require(e, len(ids) == len(set(ids)), f"duplicate dataset ids: {ids}")
    pids = [p.id for p in cfg.pipelines]
    _require(e, len(pids) == len(set(pids)), f"duplicate pipeline ids: {pids}")
    for d in cfg.datasets:
        _require(e, bool(d.id and d.version), f"dataset {d.id!r} needs id and version")
        _require(
            e,
            d.role in VALID_DATASET_ROLES,
            f"dataset {d.id}: role must be one of {sorted(VALID_DATASET_ROLES)}",
        )
        _require(
            e,
            d.group_basis in GROUP_BASIS_PRIORITY,
            f"dataset {d.id}: unknown group_basis {d.group_basis!r}",
        )
        _require(
            e,
            d.annotation_kind in VALID_ANNOTATIONS,
            f"dataset {d.id}: annotation_kind must be one of {sorted(VALID_ANNOTATIONS)}",
        )
        _require(e, d.archive_size_bytes >= 0, f"dataset {d.id}: archive_size_bytes is negative")
        _require(e, d.expected_images >= 0, f"dataset {d.id}: expected_images is negative")
        _require(e, d.expected_pairs >= 0, f"dataset {d.id}: expected_pairs is negative")
        if d.archive_sha256:
            _require(
                e,
                len(d.archive_sha256) == 64
                and all(char in "0123456789abcdefABCDEF" for char in d.archive_sha256),
                f"dataset {d.id}: archive_sha256 must be 64 hexadecimal characters",
            )
        _require(
            e,
            d.parent_archive_size_bytes >= 0,
            f"dataset {d.id}: parent_archive_size_bytes is negative",
        )
        if d.parent_archive_sha256:
            _require(
                e,
                len(d.parent_archive_sha256) == 64
                and all(
                    char in "0123456789abcdefABCDEF" for char in d.parent_archive_sha256
                ),
                f"dataset {d.id}: parent_archive_sha256 must be 64 hexadecimal characters",
            )
        has_parent = any(
            (
                d.parent_source_url,
                d.parent_archive_path,
                d.parent_archive_sha256,
                d.parent_archive_md5,
                d.parent_archive_size_bytes,
                d.parent_licence,
            )
        )
        if has_parent:
            _require(
                e,
                bool(
                    d.parent_source_url
                    and d.parent_archive_path
                    and d.parent_archive_sha256
                    and d.parent_archive_size_bytes > 0
                    and d.parent_licence
                ),
                f"dataset {d.id}: parent archive contract is incomplete",
            )
        if d.parent_archive_md5:
            _require(
                e,
                len(d.parent_archive_md5) == 32
                and all(char in "0123456789abcdefABCDEF" for char in d.parent_archive_md5),
                f"dataset {d.id}: parent_archive_md5 must be 32 hexadecimal characters",
            )
    for p in cfg.pipelines:
        _require(e, bool(p.id and p.version), f"pipeline {p.id!r} needs id and version")
        _require(e, p.min_matches >= 4, f"pipeline {p.id}: robust fitting needs >= 4 matches")
        _require(e, p.ransac_threshold_px > 0, f"pipeline {p.id}: threshold must be positive")
        _require(e, p.max_iters > 0, f"pipeline {p.id}: max_iters must be positive")
        _require(
            e,
            0.0 < p.confidence < 1.0,
            f"pipeline {p.id}: confidence must lie strictly in (0, 1)",
        )
        _require(
            e,
            p.transform_family in {"homography", "affine", "tps", "dense"},
            f"pipeline {p.id}: unsupported transform_family {p.transform_family!r}",
        )
        _require(
            e,
            p.fit
            in {"warpaudit_ransac_dlt", "warpaudit_ransac_consensus_tps"},
            f"pipeline {p.id}: configured fitter {p.fit!r} has no addressable local implementation",
        )
        _require(
            e,
            p.adapter in {"entrypoint", "subprocess"},
            f"pipeline {p.id}: adapter must be 'entrypoint' or 'subprocess'",
        )
        _require(e, p.timeout_s > 0, f"pipeline {p.id}: timeout_s must be positive")
        if p.adapter == "subprocess":
            _require(e, bool(p.command), f"pipeline {p.id}: subprocess adapter needs command")
            _require(
                e,
                all(bool(part) for part in p.command),
                f"pipeline {p.id}: command entries must be non-empty",
            )
            _require(
                e,
                isinstance(p.options.get("environment_lock"), str)
                and bool(p.options.get("environment_lock")),
                f"pipeline {p.id}: subprocess adapter needs options.environment_lock",
            )

    g = cfg.geometry
    _require(e, g.working_long_edge > 0, "geometry.working_long_edge must be positive")
    _require(e, g.grid_size >= 2, "geometry.grid_size must be >= 2")
    _require(
        e,
        g.pixel_center_convention == "integer-centre",
        "geometry.pixel_center_convention is fixed to 'integer-centre' by §5.2",
    )
    _require(e, g.interpolation == "bilinear", "geometry.interpolation must be 'bilinear'")
    _require(e, g.padding_mode == "zeros", "geometry.padding_mode must be 'zeros'")

    lb = cfg.labels
    _require(e, lb.tau_primary > 0, "labels.tau_primary must be positive")
    _require(
        e,
        all(t > 0 for t in lb.tau_secondary),
        "labels.tau_secondary values must be positive",
    )
    _require(e, lb.bounded_loss_cap > 1, "labels.bounded_loss_cap must exceed 1")
    _require(
        e,
        lb.hpatches_min_valid_points >= 20,
        "labels.hpatches_min_valid_points below the predeclared exclusion of 20 (§6.1)",
    )

    ft = cfg.features
    unknown = sorted(set(ft.families) - VALID_FAMILIES)
    _require(e, not unknown, f"features.families contains unknown families {unknown}")
    _require(
        e,
        {"A", "B", "E1"} <= set(ft.families),
        "pilot feature configuration must include the A/B/E1 core",
    )
    _require(e, ft.bootstrap_B >= 2, "features.bootstrap_B must be >= 2 for a spread")
    _require(e, ft.perturbation_B >= 2, "features.perturbation_B must be >= 2")
    _require(
        e,
        ft.e1_repeatability_tolerance > 0,
        "features.e1_repeatability_tolerance must be positive",
    )
    _require(
        e,
        0 <= ft.e1_seed_dominance_max_ratio <= 1,
        "features.e1_seed_dominance_max_ratio must lie in [0, 1]",
    )
    _require(
        e,
        -1 <= ft.e1_min_error_spearman <= 1,
        "features.e1_min_error_spearman must lie in [-1, 1]",
    )
    _require(e, ft.cycle_tolerance > 0, "features.cycle_tolerance must be positive")
    _require(
        e,
        set(ft.fallback_families_if_e1_degenerate) <= VALID_FAMILIES,
        "features.fallback_families_if_e1_degenerate contains unknown families",
    )
    _require(
        e,
        "E1" not in ft.fallback_families_if_e1_degenerate,
        "the E1-degenerate fallback composite must not itself contain E1 (§7.2a)",
    )

    sp = cfg.splits
    _require(
        e,
        0.0 < sp.development_fraction < 1.0,
        "splits.development_fraction must lie strictly in (0, 1)",
    )
    _require(e, sp.n_outer_folds >= sp.min_outer_folds >= 2, "splits: need >= 2 folds")
    _require(
        e, sp.low_information_group_threshold > 0, "low-information threshold must be positive"
    )
    _require(
        e,
        sp.conformal_min_calibration_groups >= 19,
        "conformal_min_calibration_groups must be at least 19 for alpha=0.05",
    )

    # The registration job id is keyed by pipeline *version*, not id (§12.4), so
    # two pipelines sharing a version string would silently share cache rows and
    # the second would read the first's results as its own.
    versions: dict[str, list[str]] = {}
    for pipeline in cfg.pipelines:
        versions.setdefault(pipeline.version, []).append(pipeline.id)
    for version, pipeline_ids in sorted(versions.items()):
        _require(
            e,
            len(pipeline_ids) == 1,
            f"pipelines {sorted(pipeline_ids)} share version {version!r}; job identity is keyed by "
            "pipeline version, so their cached registrations would collide",
        )

    pl = cfg.planning
    _require(
        e,
        pl.min_class_bearing_groups >= 1 and pl.min_accepted_groups >= 1,
        "planning minima must be at least one group",
    )
    _require(
        e,
        0.0 < pl.feasibility_probability < 1.0,
        "planning.feasibility_probability must lie strictly in (0, 1)",
    )
    _require(
        e,
        -1.0 < pl.paired_correlation < 1.0,
        "planning.paired_correlation must lie strictly in (-1, 1)",
    )
    _require(
        e,
        0.0 < pl.accepted_failure_risk < 1.0,
        "planning.accepted_failure_risk must lie strictly in (0, 1)",
    )
    _require(e, pl.n_simulations >= 100, "planning.n_simulations must be >= 100")
    _require(e, bool(pl.scenarios), "planning.scenarios must declare at least one scenario")
    for name, (auroc, gap, delta) in pl.scenarios.items():
        _require(
            e,
            0.0 < auroc < 1.0 and 0.0 <= gap < 1.0 and -1.0 < delta < 1.0,
            f"planning.scenarios.{name}: assumed values are outside their admissible ranges",
        )

    po = cfg.policy
    _require(
        e,
        0.0 < po.nominal_acceptance < 1.0,
        "policy.nominal_acceptance must lie strictly in (0, 1)",
    )
    _require(
        e,
        all(0.0 < c <= 1.0 for c in po.requested_coverages),
        "policy.requested_coverages must lie in (0, 1]",
    )
    _require(
        e,
        po.calibration_method in {"platt", "isotonic"},
        "policy.calibration_method must be 'platt' or 'isotonic'",
    )
    _require(e, po.reliability_bins >= 2, "policy.reliability_bins must be >= 2")
    _require(
        e,
        po.tie_rule == "conservative_reject",
        "policy.tie_rule is fixed to 'conservative_reject' for deployed policies",
    )
    _require(
        e,
        0.0 < po.safety_accepted_failure_rate < 1.0,
        "policy.safety_accepted_failure_rate must lie strictly in (0, 1)",
    )

    learner = cfg.learners
    _require(e, learner.primary == "logistic", "learners.primary is fixed to 'logistic'")
    _require(e, all(C > 0 for C in learner.logistic_C), "all logistic C values must be positive")
    _require(
        e,
        learner.sample_weighting == "inverse_group_size",
        "learners.sample_weighting must be 'inverse_group_size'",
    )

    inf = cfg.inference
    _require(e, 0.5 < inf.auroc_floor < 1.0, "inference.auroc_floor must lie in (0.5, 1)")
    _require(
        e,
        inf.gap_noninferiority_margin > 0,
        "inference.gap_noninferiority_margin must be positive",
    )
    _require(e, inf.delta_policy_margin > 0, "inference.delta_policy_margin must be positive")
    _require(
        e,
        0.0 < inf.alpha_one_sided < 0.5,
        "inference.alpha_one_sided must lie in (0, 0.5)",
    )
    _require(
        e,
        inf.multiplicity_rule == "intersection_union",
        "inference.multiplicity_rule is fixed to 'intersection_union' by §10.2",
    )
    _require(
        e,
        inf.conditional_bootstrap >= 1000 and inf.refit_bootstrap >= 100,
        "bootstrap counts are too small for the declared interval precision",
    )

    if cfg.primary_direction is not None:
        _require(
            e,
            cfg.primary_direction in cfg.direction_priority,
            "primary_direction must appear in the pre-recorded direction_priority order",
        )
        src = cfg.primary_direction.split("->")[0]
        direction_parts = cfg.primary_direction.split("->")
        _require(
            e,
            len(direction_parts) == 2
            and direction_parts[0] in ids
            and direction_parts[1] in ids
            and direction_parts[0] != direction_parts[1],
            "primary_direction must name two distinct configured datasets",
        )
        secondary = {d.id for d in cfg.datasets if d.secondary_only}
        _require(
            e,
            src not in secondary,
            f"dataset {src!r} is marked secondary_only and cannot be a primary source (§4.3)",
        )

    if len(cfg.common_block) < 2:
        e.append("at least two pipelines must be marked in_common_block (§5.1)")

    if cfg.tier == "full":
        fs = cfg.full_study
        configured = set(ids)
        declared = (
            set(fs.development_datasets)
            | set(fs.external_confirmatory_datasets)
            | set(fs.external_descriptive_datasets)
        )
        _require(e, bool(fs.development_datasets), "full_study needs development datasets")
        _require(
            e,
            bool(fs.external_confirmatory_datasets) or fs.analysis_mode == "descriptive",
            "full_study needs at least one untouched external confirmatory dataset",
        )
        _require(e, fs.analysis_mode in {"confirmatory", "descriptive"},
                 "full_study.analysis_mode must be confirmatory or descriptive")
        _require(e, fs.analysis_mode != "descriptive" or not fs.external_confirmatory_datasets,
                 "descriptive mode cannot declare confirmatory datasets")
        _require(e, bool(fs.external_confirmatory_datasets or fs.external_descriptive_datasets),
                 "full_study needs external datasets")
        _require(e, declared <= configured, "full_study names an unconfigured dataset")
        _require(
            e,
            len(declared)
            == len(fs.development_datasets)
            + len(fs.external_confirmatory_datasets)
            + len(fs.external_descriptive_datasets),
            "full_study dataset roles must be disjoint",
        )
        _require(
            e,
            set(fs.baseline_families) < set(fs.augmented_families) <= VALID_FAMILIES,
            "full_study augmented_families must be a strict valid superset of baseline_families",
        )
        _require(
            e,
            {"E1", "E2"} <= set(fs.augmented_families),
            "full_study augmented model must include factorized E1 and input-perturbation E2",
        )
        _require(
            e,
            fs.primary_metric == "brier_improvement",
            "full_study.primary_metric is frozen to brier_improvement",
        )
        _require(e, fs.min_brier_improvement >= 0, "minimum Brier improvement is negative")
        _require(
            e,
            0 < fs.min_passing_cell_fraction <= 1,
            "full-study passing-cell fraction must lie in (0, 1]",
        )
        _require(e, fs.min_class_bearing_groups >= 10, "full-study class gate must be >= 10")
        _require(
            e,
            0.5 < fs.min_information_probability < 1,
            "full-study information probability must lie in (0.5, 1)",
        )
        _require(e, fs.bootstrap_resamples >= 1000, "full-study bootstrap needs >= 1000 draws")
        _require(
            e,
            0 < fs.high_confidence_cutoff < 0.5,
            "full-study high-confidence cutoff must lie in (0, 0.5)",
        )

    return e


def load_config(path: str | Path, *, strict: bool = True) -> Config:
    """Load and validate a YAML configuration."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    raw.setdefault("source_path", str(path))

    errors: list[str] = []
    cfg = _coerce(raw, errors)
    errors.extend(validate_config(cfg))
    if errors and strict:
        joined = "\n  - ".join(errors)
        raise ConfigError(f"{path}: {len(errors)} configuration problem(s):\n  - {joined}")
    return cfg
