"""Command-line boundary for the Sprint-One protocol.

Commands fail closed: missing datasets, manifests, matcher adapters, or
measured timings are reported as unavailable evidence.  They never turn a
zero-row run or a literature timing into a completed experimental result.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import itertools
import json
import multiprocessing
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

from . import __version__
from .cache.hashing import file_digest, short_hash
from .cache.ledger import JobRecord, StatusLedger
from .cache.store import (
    PARQUET_UNAVAILABLE_REASON,
    ShardedTable,
    atomic_write_bytes,
    atomic_write_text,
)
from .config import Config, ConfigError, load_config
from .config_edit import replace_pipeline_provenance
from .data.annotations import load_coph100_control_points, load_fire_control_points
from .data.development import DevelopmentManifest, select_development_groups
from .data.grouping import assign_groups, grouping_claim
from .data.loaders import DatasetUnavailable, PairListing, get_loader
from .data.manifest import (
    DatasetProvenance,
    PairRecord,
    ProvenanceManifest,
    default_exposure_register,
)
from .data.prepare import (
    ArchiveSpec,
    PreparationError,
    download_archive,
    prepare_coph100,
    prepare_fire,
    verify_archive,
)
from .evaluation.bootstrap import BootstrapResult, monte_carlo_stability, refit_bootstrap
from .evaluation.card import evaluate_scores
from .evaluation.experiment import ROLES as EXPERIMENT_ROLES
from .evaluation.experiment import FoldOutcome, evaluate_fold
from .evaluation.full_study import evaluate_full_study
from .evaluation.inference import decide
from .evaluation.overlays import contact_sheet, render_registration_panel
from .evaluation.perturbations import (
    translation_perturbations,
    warp_by_input_map,
)
from .evaluation.report import (
    FIGURE_SPECS,
    capability_table,
    markdown_table,
    matplotlib_available,
    per_pipeline_table,
    policy_table,
    primary_table,
    provenance_table,
    reliability_frame,
    render_figures,
    signal_family_table,
)
from .geometry.coordinates import identity_frame, make_frame, to_original_frame
from .geometry.grids import prespecified_grid
from .geometry.resample import resample_to_frame
from .labels.errors import landmark_success_fraction, point_errors
from .labels.targets import case_outcome
from .parallel import ordered_map
from .protocols.freeze import (
    FREEZE_FILENAME,
    FreezeError,
    FreezeRecord,
    require_confirmatory_access,
)
from .protocols.full_freeze import FullStudyFreeze, require_full_freeze
from .protocols.leakage import check_split, enforce
from .protocols.planning import (
    DevelopmentEvidence,
    DirectionPlan,
    development_evidence,
    make_direction_plan,
    project_accepted_groups,
    project_class_support,
    simulate_iut_scenario,
)
from .protocols.splits import TransferSplit, make_outer_folds
from .registration.adapters import AdapterUnavailable, load_registrar
from .registration.runner import (
    RegistrationJob,
    RegistrationRunner,
    deserialise_transform,
)
from .registration.subprocess_adapter import (
    AdapterProtocolError,
    _venv_interpreter,
    subprocess_factory,
)
from .signals import available_families, compute_families
from .signals.context import ContextSources, build_signal_context
from .signals.diagnostics import classify_cycle, classify_e1
from .study import StagePlan, build_plan
from .types import (
    CoordinateMetadata,
    PairInput,
    RegistrationResult,
    RegistrationStatus,
)

EXIT_USAGE = 2
EXIT_PREREQUISITE = 3
#: At least one stage of an unattended run failed; its summary names which.
EXIT_STAGE_FAILED = 4


class CommandError(RuntimeError):
    """A concise, expected command failure suitable for terminal output."""

    def __init__(self, message: str, exit_code: int = EXIT_PREREQUISITE) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _project_root(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    return path.parent.parent if path.parent.name == "configs" else Path.cwd().resolve()


def _resolve(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _code_identity(root: Path) -> str:
    """Content identity of executable project code, independent of Git metadata."""
    files = (
        sorted(root.glob("warpaudit/**/*.py"))
        + sorted(root.glob("scripts/matchers/*.py"))
        + sorted(root.glob("requirements-*.lock"))
        + [root / "pyproject.toml"]
    )
    payload = [(str(path.relative_to(root)), file_digest(path)) for path in files if path.is_file()]
    return f"tree:{short_hash(payload, length=64)}"


def _scoped_code_identity(root: Path, files: Iterable[Path]) -> str:
    """Hash a computational stage without coupling it to unrelated reporting code."""
    unique = sorted({path for path in files if path.is_file()})
    payload = [(str(path.relative_to(root)), file_digest(path)) for path in unique]
    return f"tree:{short_hash(payload, length=64)}"


def _definition_code_identity(path: Path, names: Iterable[str]) -> str:
    """Hash selected Python definitions without coupling them to their module.

    Some computational stages live at the CLI boundary, but hashing all of
    ``cli.py`` makes unrelated commands invalidate expensive cached results.
    An AST identity keeps changes to the selected implementations visible while
    ignoring additions elsewhere in the module.
    """
    requested = tuple(names)
    if not path.is_file():
        return f"definitions:{short_hash([], length=64)}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    missing = sorted(set(requested) - set(definitions))
    if missing:
        raise ValueError(f"{path}: missing identity definition(s) {missing}")
    payload = [
        (name, ast.dump(definitions[name], annotate_fields=True, include_attributes=False))
        for name in requested
    ]
    return f"definitions:{short_hash(payload, length=64)}"


def _registration_code_identity(root: Path) -> str:
    # Projective-pole diagnostics are a downstream reporting primitive.  They
    # neither construct matcher inputs nor fit registration transforms, and
    # evidence_followup hashes that module separately.  Keeping it out of the
    # registration/feature identities prevents an analysis-only addition from
    # invalidating the expensive registration and feature caches.
    geometry_files = [
        path
        for path in root.glob("warpaudit/geometry/**/*.py")
        if path.name != "projective.py"
    ]
    files = (
        list(root.glob("warpaudit/registration/**/*.py"))
        + geometry_files
        + list(root.glob("scripts/matchers/*.py"))
        + [
            root / "warpaudit/types.py",
            root / "warpaudit/cache/hashing.py",
            root / "warpaudit/cache/ledger.py",
            root / "warpaudit/parallel.py",
            root / "warpaudit/cache/store.py",
            root / "requirements-eval.lock",
            root / "pyproject.toml",
        ]
    )
    return _scoped_code_identity(root, files)


def _feature_code_identity(root: Path) -> str:
    geometry_files = [
        path
        for path in root.glob("warpaudit/geometry/**/*.py")
        if path.name != "projective.py"
    ]
    files = (
        list(root.glob("warpaudit/signals/**/*.py"))
        + geometry_files
        + [
            root / "warpaudit/types.py",
            root / "warpaudit/registration/fitting.py",
            root / "requirements-eval.lock",
            root / "pyproject.toml",
        ]
    )
    # Features depend on correspondence extraction and both registrations too
    # (notably C and F). A registration repair must invalidate derived features.
    return "features:" + short_hash(
        {"implementation": _scoped_code_identity(root, files),
         "registration": _registration_code_identity(root),
         "e2": _e2_code_identity(root)},
        length=64,
    )


def _e2_code_identity(root: Path) -> str:
    """Identity of registration plus deterministic E2 input construction.

    Only the CLI definitions that construct an E2 job are relevant here.  In
    particular, reporting and evaluation commands in the same module cannot
    alter an E2 registration and must not invalidate all derived features.
    """
    perturbation_hash = short_hash(
        {
            "implementation": _scoped_code_identity(
                root,
                [
                    root / "warpaudit/evaluation/perturbations.py",
                    root / "warpaudit/signals/family_e2_perturbation.py",
                ],
            ),
            "construction": _definition_code_identity(
                root / "warpaudit/cli.py", ("_pair_from_row", "_prepare_e2_pair")
            ),
        },
        length=64,
    )
    return f"e2:{short_hash({'registration': _registration_code_identity(root), 'perturbation': perturbation_hash}, length=64)}"


def _git_commit(root: Path) -> str:
    """Best-effort repository revision, recorded separately from code bytes."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""


def _pipeline_contract_hashes(cfg: Config, root: Path, pipeline_id: str) -> tuple[str, str]:
    """Return the environment and checkpoint identities used by registration jobs."""
    pipeline = cfg.pipeline(pipeline_id)
    environment_lock = _resolve(root, str(pipeline.options.get("environment_lock", "")))
    if pipeline.adapter == "subprocess" and not environment_lock.is_file():
        raise CommandError(f"pipeline environment lock is absent: {environment_lock}")
    environment_hash = short_hash(
        {
            "environment": pipeline.environment,
            "adapter": pipeline.adapter,
            "command": pipeline.command,
            "expected_provenance": pipeline.provenance,
            "environment_lock_hash": (
                file_digest(environment_lock) if pipeline.adapter == "subprocess" else ""
            ),
        }
    )
    checkpoint_hash = short_hash(
        {"checkpoint": pipeline.checkpoint, "provenance": pipeline.provenance}
    )
    return environment_hash, checkpoint_hash


def _current_registration_rows(
    registrations: pd.DataFrame,
    pairs: pd.DataFrame,
    cfg: Config,
    root: Path,
) -> pd.DataFrame:
    """Reject cached rows whose complete execution contract is no longer current."""
    if registrations.empty:
        return registrations.copy()
    required = {
        "pair_id",
        "dataset_version",
        "pipeline_id",
        "pipeline_version",
        "code_hash",
        "env_hash",
        "checkpoint_hash",
        "config_hash",
    }
    missing = required - set(registrations)
    if missing:
        raise CommandError(
            "registration cache predates the strict provenance contract; missing columns: "
            f"{sorted(missing)}"
        )
    dataset_versions = {
        str(row["pair_id"]): str(row["dataset_version"]) for _, row in pairs.iterrows()
    }
    pipeline_contracts = {
        pipeline.id: (*_pipeline_contract_hashes(cfg, root, pipeline.id), pipeline.version)
        for pipeline in cfg.pipelines
    }
    code_hash = _registration_code_identity(root)

    def is_current(row: pd.Series) -> bool:
        pair_id = str(row["pair_id"])
        contract = pipeline_contracts.get(str(row["pipeline_id"]))
        if pair_id not in dataset_versions or contract is None:
            return False
        env_hash, checkpoint_hash, pipeline_version = contract
        condition = str(row.get("condition", "clean"))
        expected = {
            "dataset_version": dataset_versions[pair_id],
            "pipeline_version": pipeline_version,
            "code_hash": _e2_code_identity(root) if condition.startswith("e2:") else code_hash,
            "env_hash": env_hash,
            "checkpoint_hash": checkpoint_hash,
            "config_hash": cfg.hash,
        }
        return all(str(row.get(key, "")) == str(value) for key, value in expected.items())

    mask = registrations.apply(is_current, axis=1)
    return registrations.loc[mask].copy()


def _load_subject_maps(items: Iterable[str]) -> dict[str, dict[str, tuple[str, str]]]:
    """Read ``DATASET=csv`` maps with pair_id, subject_id, and optional basis."""
    out: dict[str, dict[str, tuple[str, str]]] = {}
    for item in items:
        if "=" not in item:
            raise CommandError(f"invalid --subject-map {item!r}; expected DATASET=path", EXIT_USAGE)
        dataset_id, raw_path = item.split("=", 1)
        rows: dict[str, tuple[str, str]] = {}
        with open(raw_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            required = {"pair_id", "subject_id"}
            if not required <= set(reader.fieldnames or []):
                raise CommandError(f"{raw_path}: subject map needs columns {sorted(required)}")
            for row in reader:
                pair_id = str(row["pair_id"]).strip()
                if not pair_id.startswith(f"{dataset_id}/"):
                    pair_id = f"{dataset_id}/{pair_id}"
                subject = str(row["subject_id"]).strip()
                basis = str(row.get("basis") or "patient").strip()
                if subject:
                    rows[pair_id] = (subject, basis)
        out[dataset_id] = rows
    return out


def _annotation_count(pair: PairListing) -> int:
    paths = pair.annotation_paths
    if not paths or any(not path.exists() for path in paths):
        return 0
    try:
        if len(paths) == 1:
            moving, _ = load_fire_control_points(paths[0])
        elif len(paths) == 2:
            moving, _ = load_coph100_control_points(paths[0], paths[1])
        else:
            raise ValueError(f"unsupported annotation file count: {len(paths)}")
    except (OSError, ValueError) as exc:
        raise CommandError(f"{pair.pair_id}: invalid annotation: {exc}") from exc
    return int(len(moving))


def _archive_spec(ds) -> ArchiveSpec:
    if not ds.source_url or not ds.archive_sha256:
        raise CommandError(
            f"{ds.id}: source_url and archive_sha256 must be configured before preparation"
        )
    return ArchiveSpec(ds.id, ds.source_url, ds.archive_sha256, ds.archive_size_bytes)


def _parent_archive_spec(ds) -> ArchiveSpec:
    if not ds.parent_source_url or not ds.parent_archive_sha256:
        raise CommandError(
            f"{ds.id}: parent_source_url and parent_archive_sha256 must be configured"
        )
    return ArchiveSpec(
        f"{ds.id}/parent",
        ds.parent_source_url,
        ds.parent_archive_sha256,
        ds.parent_archive_size_bytes,
    )


def _download_progress(label: str) -> Callable[[int, int], None]:
    """Report download progress sparsely enough to keep a log readable."""
    state = {"next": 0}

    def report(received: int, total: int) -> None:
        step = 64 << 20  # every 64 MiB
        if received < state["next"] and received < total:
            return
        state["next"] = received + step
        if total:
            print(
                f"  {label}: {received / 1e6:,.0f} / {total / 1e6:,.0f} MB "
                f"({received / total:.0%})",
                flush=True,
            )
        else:
            print(f"  {label}: {received / 1e6:,.0f} MB", flush=True)

    return report


def command_prepare_data(args: argparse.Namespace) -> int:
    """Prepare local data from official, checksum-pinned archives."""
    cfg = load_config(args.config)
    root = _project_root(args.config)
    selected = cfg.datasets if args.dataset == "all" else (cfg.dataset(args.dataset),)
    for ds in selected:
        archive = _resolve(root, ds.archive_path)
        dataset_root = _dataset_root(root, ds.root)
        spec = _archive_spec(ds)
        try:
            if not archive.is_file():
                if not args.download:
                    raise PreparationError(
                        f"{ds.id}: archive absent at {archive}; pass --download to retrieve "
                        f"the pinned official file from {ds.source_url}"
                    )
                if ds.id == "FIRE" and not args.acknowledge_fire_terms_unresolved:
                    raise PreparationError(
                        "FIRE has no explicit redistribution licence on its official page. "
                        "Review DATA_LICENCES.md and pass --acknowledge-fire-terms-unresolved "
                        "to download for local research use."
                    )
                download_archive(archive, spec, progress=_download_progress(ds.id))
            else:
                verify_archive(archive, spec)
            if ds.id == "FIRE":
                if not args.acknowledge_fire_terms_unresolved:
                    raise PreparationError(
                        "Review the unresolved FIRE terms in DATA_LICENCES.md, then pass "
                        "--acknowledge-fire-terms-unresolved before local extraction."
                    )
                summary = prepare_fire(
                    archive, dataset_root, spec, reuse_existing=args.reuse_existing
                )
            elif ds.id == "COph100":
                parent_archive = _resolve(root, ds.parent_archive_path)
                parent_spec = _parent_archive_spec(ds)
                if not parent_archive.is_file():
                    if not args.download:
                        raise PreparationError(
                            f"COph100 parent archive absent at {parent_archive}; pass --download"
                        )
                    download_archive(
                        parent_archive,
                        parent_spec,
                        progress=_download_progress(f"{ds.id} parent"),
                    )
                else:
                    verify_archive(parent_archive, parent_spec)
                summary = prepare_coph100(
                    archive,
                    parent_archive,
                    dataset_root,
                    spec,
                    parent_spec,
                    reuse_existing=args.reuse_existing,
                )
            else:
                raise PreparationError(f"no archive preparer implemented for {ds.id!r}")
        except (OSError, PreparationError) as exc:
            raise CommandError(str(exc)) from exc
        print(f"{ds.id}: {json.dumps(summary, sort_keys=True)}")
    return 0


def _write_frame(path: Path, frame: pd.DataFrame) -> Path:
    """Atomic Parquet write; JSONL fallback is explicit in the returned path."""
    try:
        data = frame.to_parquet(index=False)
        if data is None:
            raise RuntimeError("pandas did not return Parquet bytes")
        return atomic_write_bytes(path, data)
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        fallback = path.with_suffix(".jsonl")
        atomic_write_text(fallback, frame.to_json(orient="records", lines=True))
        atomic_write_text(
            path.with_suffix(".format-warning.txt"),
            f"Parquet unavailable; wrote {fallback.name}. Reason: {type(exc).__name__}: {exc}\n",
        )
        return fallback


def _dataset_root(project_root: Path, configured_root: str) -> Path:
    return _resolve(project_root, configured_root)


def command_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(f"valid: {args.config}")
    print(f"spec_revision: {cfg.spec_revision}")
    print(f"config_hash: {cfg.hash}")
    print(f"datasets: {', '.join(d.id for d in cfg.datasets)}")
    print(f"common_block: {', '.join(cfg.common_block)}")
    return 0


def command_environment(args: argparse.Namespace) -> int:
    root = _project_root(args.config)
    cfg = load_config(args.config)
    import pandas
    import scipy
    import sklearn

    payload = {
        "generated_at": _utc_now(),
        "warpaudit": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "config_hash": cfg.hash,
        "code_identity": _code_identity(root),
        "registration_code_identity": _registration_code_identity(root),
        "feature_code_identity": _feature_code_identity(root),
        "git_commit": _git_commit(root),
        "parquet_available": not bool(PARQUET_UNAVAILABLE_REASON()),
        "parquet_unavailable_reason": PARQUET_UNAVAILABLE_REASON(),
    }
    output = _resolve(root, args.output) if args.output else None
    if output:
        atomic_write_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(output)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_audit_data(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    root = _project_root(args.config)
    paths = cfg.paths.resolve(root)
    manifests = paths["manifests"]
    subject_maps = _load_subject_maps(args.subject_map)
    generated_at = _utc_now()
    code_id = _code_identity(root)

    all_records: list[PairRecord] = []
    provenance: list[DatasetProvenance] = []
    dataset_groups: dict[str, list[str]] = {}
    warnings: list[str] = []
    image_inventory: dict[str, dict[str, object]] = {}
    annotation_inventory: dict[str, dict[str, object]] = {}
    digest_to_global_ids: dict[str, list[str]] = defaultdict(list)
    duplicate_content_by_dataset: dict[str, dict[str, list[str]]] = {}

    for ds in cfg.datasets:
        archive_digest = ""
        source_archives: list[dict[str, str | int]] = []
        archive = _resolve(root, ds.archive_path) if ds.archive_path else None
        if archive is not None:
            if not archive.is_file():
                raise CommandError(f"{ds.id}: configured archive is absent: {archive}")
            archive_digest = file_digest(archive)
            if ds.archive_sha256 and archive_digest.lower() != ds.archive_sha256.lower():
                raise CommandError(
                    f"{ds.id}: archive SHA-256 mismatch: expected {ds.archive_sha256}, "
                    f"got {archive_digest}"
                )
            source_archives.append(
                {
                    "role": "annotations" if ds.parent_archive_path else "dataset_archive",
                    "url": ds.source_url,
                    "path": str(archive.relative_to(root)),
                    "size_bytes": archive.stat().st_size,
                    "sha256": archive_digest,
                }
            )
        if ds.parent_archive_path:
            parent_archive = _resolve(root, ds.parent_archive_path)
            if not parent_archive.is_file():
                raise CommandError(f"{ds.id}: configured parent archive is absent: {parent_archive}")
            parent_digest = file_digest(parent_archive)
            if (
                ds.parent_archive_sha256
                and parent_digest.lower() != ds.parent_archive_sha256.lower()
            ):
                raise CommandError(
                    f"{ds.id}: parent archive SHA-256 mismatch: "
                    f"expected {ds.parent_archive_sha256}, got {parent_digest}"
                )
            source_archives.append(
                {
                    "role": "source_images",
                    "url": ds.parent_source_url,
                    "path": str(parent_archive.relative_to(root)),
                    "size_bytes": parent_archive.stat().st_size,
                    "sha256": parent_digest,
                    "supplied_md5": ds.parent_archive_md5,
                    "licence": ds.parent_licence,
                }
            )
        try:
            listings = get_loader(ds.id)(_dataset_root(root, ds.root), ds.id)
        except DatasetUnavailable as exc:
            raise CommandError(str(exc)) from exc

        external = subject_maps.get(ds.id, {})
        subject_ids: dict[str, str] = {}
        subject_bases: dict[str, str] = {}
        for pair in listings:
            if pair.pair_id in external:
                subject_ids[pair.pair_id], subject_bases[pair.pair_id] = external[pair.pair_id]
            elif pair.subject_id:
                subject_ids[pair.pair_id] = pair.subject_id
                subject_bases[pair.pair_id] = pair.subject_basis

        images = {p.moving_image_id: p.moving_path for p in listings}
        images.update({p.fixed_image_id: p.fixed_path for p in listings})
        digest_to_ids: dict[str, list[str]] = defaultdict(list)
        image_items = sorted(images.items())
        digests = ordered_map(file_digest, (path for _, path in image_items),
                              workers=getattr(args, "workers", 1), threads=0, io=True)
        for (image_id, path), digest in zip(image_items, digests, strict=True):
            if not path.exists():
                raise CommandError(f"{ds.id}: image listed but absent: {path}")
            digest_to_ids[digest].append(image_id)
            digest_to_global_ids[digest].append(image_id)
            height, width = next(
                pair.moving_hw if pair.moving_image_id == image_id else pair.fixed_hw
                for pair in listings
                if image_id in (pair.moving_image_id, pair.fixed_image_id)
            )
            image_inventory[image_id] = {
                "dataset_id": ds.id,
                "image_id": image_id,
                "path": str(path.relative_to(root)),
                "sha256": digest,
                "height": height,
                "width": width,
            }
        duplicates = {d: ids for d, ids in digest_to_ids.items() if len(ids) > 1}
        duplicate_content_by_dataset[ds.id] = duplicates

        assignments, report = assign_groups(
            ds.id,
            [p.as_graph_edge() for p in listings],
            subject_ids=subject_ids,
            subject_bases=subject_bases,  # type: ignore[arg-type]
            subject_basis="patient",
            duplicate_groups=duplicates,
            subject_evidence="dataset index or explicit audit subject map",
        )
        if ds.patient_ids_available and not report.patient_disjoint_claimable:
            raise CommandError(
                f"{ds.id}: config asserts complete patient IDs, but the realized audit "
                f"has {report.n_pairs_without_subject_id} unresolved pairs and basis {report.basis}"
            )
        if not ds.patient_ids_available and report.patient_disjoint_claimable:
            warnings.append(
                f"{ds.id}: complete patient identity was observed but "
                "patient_ids_available is false in config"
            )
        annotation_files = sorted({path for pair in listings for path in pair.annotation_paths})
        annotation_hashes = dict(zip(annotation_files, ordered_map(
            file_digest, annotation_files, workers=getattr(args, "workers", 1), threads=0, io=True
        ), strict=True))
        by_pair = {a.pair_id: a for a in assignments}
        if ds.expected_pairs and len(listings) != ds.expected_pairs:
            raise CommandError(
                f"{ds.id}: expected {ds.expected_pairs} pairs, found {len(listings)}"
            )
        if ds.expected_images and len(digest_to_ids) != ds.expected_images:
            raise CommandError(
                f"{ds.id}: expected {ds.expected_images} unique image contents, "
                f"found {len(digest_to_ids)} across {report.n_images} image IDs"
            )
        dataset_groups[ds.id] = sorted({a.group_id for a in assignments})
        for pair in listings:
            assignment = by_pair[pair.pair_id]
            annotation_paths = tuple(str(path.relative_to(root)) for path in pair.annotation_paths)
            annotation_digests = tuple(annotation_hashes[path] for path in pair.annotation_paths)
            for path, digest in zip(pair.annotation_paths, annotation_digests, strict=True):
                key = str(path.relative_to(root))
                annotation_inventory[key] = {
                    "dataset_id": ds.id,
                    "path": key,
                    "sha256": digest,
                    "kind": pair.annotation_kind,
                }
            all_records.append(
                PairRecord(
                    dataset_id=ds.id,
                    dataset_version=ds.version,
                    pair_id=pair.pair_id,
                    moving_image_id=pair.moving_image_id,
                    fixed_image_id=pair.fixed_image_id,
                    moving_path=str(pair.moving_path.relative_to(root)),
                    fixed_path=str(pair.fixed_path.relative_to(root)),
                    moving_sha256=str(image_inventory[pair.moving_image_id]["sha256"]),
                    fixed_sha256=str(image_inventory[pair.fixed_image_id]["sha256"]),
                    moving_hw=pair.moving_hw,
                    fixed_hw=pair.fixed_hw,
                    group_id=assignment.group_id,
                    group_basis=assignment.group_basis,
                    component_id=assignment.component_id,
                    annotation_kind=pair.annotation_kind,
                    annotation_provenance=pair.annotation_provenance,
                    annotation_paths=annotation_paths,
                    annotation_sha256=annotation_digests,
                    n_landmarks=_annotation_count(pair),
                    dataset_category=pair.dataset_category,
                    eye_id=str(pair.extra.get("eye_id", "")),
                    patient_id=str(pair.extra.get("patient_id", "")),
                    subject_evidence=pair.subject_evidence or assignment.subject_evidence,
                )
            )
        claim = grouping_claim(report)
        print(claim)
        provenance.append(
            DatasetProvenance(
                dataset_id=ds.id,
                version=ds.version,
                download_date=ds.download_date,
                citation=ds.citation,
                archive_checksum=f"sha256:{archive_digest or ds.archive_sha256}",
                source_url=ds.source_url,
                source_archives=source_archives,
                licence=ds.licence,
                redistribute_images=ds.redistribute_images,
                redistribute_derived=ds.redistribute_derived,
                n_images=len(digest_to_ids),
                n_image_ids=report.n_images,
                n_pairs=report.n_pairs,
                n_groups=report.n_groups,
                n_exact_duplicate_sets=len(duplicates),
                group_basis=report.basis,
                grouping_claim=claim,
                unresolved_identities=report.n_pairs_without_subject_id,
                notes=[ds.access_note, *report.notes],
            )
        )
        warnings.extend(f"{ds.id}: {note}" for note in report.notes)

    # Full-study roles are dataset-level: all previously inspected datasets are
    # development and every external dataset remains untouched. The pilot keeps
    # its historical within-dataset development fraction.
    development: list[str] = []
    if cfg.tier == "full":
        for dataset_id in cfg.full_study.development_datasets:
            development.extend(dataset_groups[dataset_id])
        development_note = (
            "Full-study dataset roles applied: development datasets are fully inspectable; "
            "external datasets are entirely confirmatory."
        )
    else:
        for ds in cfg.datasets:
            development.extend(
                select_development_groups(
                    dataset_groups[ds.id],
                    fraction=cfg.splits.development_fraction,
                    seed=cfg.splits.seed,
                )
            )
        development_note = "Development selection was stratified by dataset and rounded up."
    all_groups = sorted({r.group_id for r in all_records})
    dev_manifest = DevelopmentManifest(
        seed=cfg.splits.seed,
        fraction=cfg.splits.development_fraction,
        development_groups=sorted(development),
        confirmatory_groups=sorted(set(all_groups) - set(development)),
        notes=[development_note],
    )

    fold_by_group: dict[str, int] = {}
    for ds in cfg.datasets:
        candidates = [g for g in dataset_groups[ds.id] if g not in set(development)]
        n_folds = min(cfg.splits.n_outer_folds, len(candidates))
        if n_folds >= cfg.splits.min_outer_folds:
            fold_by_group.update(
                make_outer_folds(candidates, n_folds=n_folds, seed=cfg.splits.seed)
            )
        else:
            warnings.append(
                f"{ds.id}: {len(candidates)} non-development groups cannot support the "
                f"minimum {cfg.splits.min_outer_folds} outer folds; fold left unassigned"
            )

    group_pair_counts = Counter(r.group_id for r in all_records)
    for record in all_records:
        record.is_development = record.group_id in set(development)
        record.fold = fold_by_group.get(record.group_id, -1)
        record.sampling_probability = (
            1.0 / len(dataset_groups[record.dataset_id]) / group_pair_counts[record.group_id]
        )

    # The group assignment must keep every reused image on exactly one side of
    # the development boundary. This checks the realized split, not only the
    # grouping implementation.
    roles_by_image: dict[str, set[str]] = defaultdict(set)
    for record in all_records:
        role = "development" if record.is_development else "confirmatory"
        roles_by_image[record.moving_image_id].add(role)
        roles_by_image[record.fixed_image_id].add(role)
    crossing = {image: roles for image, roles in roles_by_image.items() if len(roles) > 1}
    if crossing:
        first = next(iter(sorted(crossing)))
        raise CommandError(f"shared image crosses recorded partitions: {first} -> {crossing[first]}")

    cross_dataset_duplicates = {
        digest: ids
        for digest, ids in digest_to_global_ids.items()
        if len({image_id.split("/", 1)[0] for image_id in ids}) > 1
    }
    if cross_dataset_duplicates:
        warnings.append(
            f"{len(cross_dataset_duplicates)} exact image-content duplicate set(s) span datasets; "
            "do not assign those datasets to opposing transfer roles until reviewed"
        )

    pair_frame = pd.DataFrame([r.to_row() for r in all_records])
    pair_path = _write_frame(manifests / "pairs.parquet", pair_frame)
    dev_path = dev_manifest.write(manifests / "development_groups.json")
    prov = ProvenanceManifest(
        datasets=provenance,
        checkpoints=default_exposure_register(),
        generated_at=generated_at,
        config_hash=cfg.hash,
        code_hash=code_id,
        warnings=warnings,
    )
    prov_path = prov.write(manifests / "provenance.yaml")
    data_json = {
        "generated_at": generated_at,
        "config_hash": cfg.hash,
        "code_identity": code_id,
        "n_pairs": len(all_records),
        "n_images": len(digest_to_global_ids),
        "n_image_ids": len(
            {i for r in all_records for i in (r.moving_image_id, r.fixed_image_id)}
        ),
        "n_groups": len(all_groups),
        "datasets": [p.__dict__ for p in provenance],
        "images": [image_inventory[key] for key in sorted(image_inventory)],
        "annotations": [annotation_inventory[key] for key in sorted(annotation_inventory)],
        "duplicate_content_by_dataset": duplicate_content_by_dataset,
        "cross_dataset_duplicate_content": cross_dataset_duplicates,
        "partition_audit": {"shared_images_crossing_development_boundary": 0},
        "warnings": warnings,
    }
    data_path = atomic_write_text(
        manifests / "data.json", json.dumps(data_json, indent=2, sort_keys=True) + "\n"
    )
    print(f"pairs: {pair_path} ({len(all_records)} rows)")
    print(f"development manifest: {dev_path}")
    print(f"provenance: {prov_path}")
    print(f"data summary: {data_path}")
    return 0


def _pairs_manifest(cfg: Config, root: Path) -> tuple[pd.DataFrame, Path]:
    base = cfg.paths.resolve(root)["manifests"]
    parquet = base / "pairs.parquet"
    jsonl = base / "pairs.jsonl"
    if parquet.exists():
        return pd.read_parquet(parquet), parquet
    if jsonl.exists():
        return pd.read_json(jsonl, lines=True), jsonl
    raise CommandError("pair manifest absent; run `python -m warpaudit audit-data ...` first")


def _select_pairs(
    pairs: pd.DataFrame,
    *,
    split: str,
    pair_ids: Iterable[str] = (),
    dataset_ids: Iterable[str] = (),
    limit: int | None = None,
) -> pd.DataFrame:
    """Deterministically select manifest rows without changing split membership."""
    selected = pairs[pairs["is_development"].astype(bool) == (split == "development")].copy()
    requested_datasets = set(dataset_ids)
    if requested_datasets:
        unknown = requested_datasets - set(pairs["dataset_id"].astype(str))
        if unknown:
            raise CommandError(f"unknown dataset selector(s): {sorted(unknown)}", EXIT_USAGE)
        selected = selected[selected["dataset_id"].astype(str).isin(requested_datasets)]
    requested_pairs = set(pair_ids)
    if requested_pairs:
        unknown = requested_pairs - set(pairs["pair_id"].astype(str))
        if unknown:
            raise CommandError(f"unknown pair selector(s): {sorted(unknown)}", EXIT_USAGE)
        selected = selected[selected["pair_id"].astype(str).isin(requested_pairs)]
    selected = selected.sort_values(["dataset_id", "pair_id"], kind="stable")
    if limit is not None:
        if limit <= 0:
            raise CommandError("--limit must be positive", EXIT_USAGE)
        selected = selected.head(limit)
    return selected


def _pair_from_row(row: pd.Series, cfg: Config, root: Path, *, direction: str) -> PairInput:
    moving_frame = make_frame(
        str(row["moving_image_id"]),
        (int(row["moving_h"]), int(row["moving_w"])),
        long_edge=cfg.geometry.working_long_edge,
        pad_to_square=cfg.geometry.pad_to_square,
    )
    fixed_frame = make_frame(
        str(row["fixed_image_id"]),
        (int(row["fixed_h"]), int(row["fixed_w"])),
        long_edge=cfg.geometry.working_long_edge,
        pad_to_square=cfg.geometry.pad_to_square,
    )
    moving_path = _resolve(root, str(row["moving_path"]))
    fixed_path = _resolve(root, str(row["fixed_path"]))
    moving_id, fixed_id = str(row["moving_image_id"]), str(row["fixed_image_id"])
    if direction == "reverse":
        moving_frame, fixed_frame = fixed_frame, moving_frame
        moving_path, fixed_path = fixed_path, moving_path
        moving_id, fixed_id = fixed_id, moving_id
    return PairInput(
        dataset_id=str(row["dataset_id"]),
        pair_id=str(row["pair_id"]),
        group_id=str(row["group_id"]),
        group_basis=str(row["group_basis"]),  # type: ignore[arg-type]
        moving_image_id=moving_id,
        fixed_image_id=fixed_id,
        moving_path=moving_path,
        fixed_path=fixed_path,
        coordinates=CoordinateMetadata(
            moving=moving_frame,
            fixed=fixed_frame,
            pixel_center_convention=cfg.geometry.pixel_center_convention,
            interpolation=cfg.geometry.interpolation,
            padding_mode=cfg.geometry.padding_mode,
        ),
        acquisition_meta={"dataset_category": str(row.get("dataset_category") or "")},
        direction=direction,  # type: ignore[arg-type]
    )


def _verify_pair_images(row: pd.Series, root: Path) -> None:
    for side in ("moving", "fixed"):
        path = _resolve(root, str(row[f"{side}_path"]))
        expected = str(row.get(f"{side}_sha256") or "")
        if not path.is_file():
            raise CommandError(f"{row['pair_id']}: {side} image is absent: {path}")
        if expected:
            actual = file_digest(path)
            if actual != expected:
                raise CommandError(
                    f"{row['pair_id']}: {side} image changed since audit-data; "
                    f"expected {expected}, got {actual}"
                )


def _optional_array(
    value: object, dtype: object = np.float64, *, points: bool = False
) -> np.ndarray | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    array = np.asarray(value, dtype=dtype)
    # JSON [] loses the second dimension of an empty (0, 2) point array.
    # Restore only that unambiguous representation; malformed nonempty arrays
    # still reach RegistrationResult's strict shape validation.
    return array.reshape(0, 2) if points and array.shape == (0,) else array


def _result_from_row(row: pd.Series) -> RegistrationResult:
    diagnostics = row.get("diagnostics")
    if isinstance(diagnostics, str):
        diagnostics = json.loads(diagnostics)
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    return RegistrationResult(
        pipeline_id=str(row["pipeline_id"]),
        status=RegistrationStatus(str(row["status"])),
        forward_moving_to_fixed=deserialise_transform(row.get("transform_params")),
        matches_moving=_optional_array(row.get("matches_moving"), points=True),
        matches_fixed=_optional_array(row.get("matches_fixed"), points=True),
        inlier_mask=_optional_array(row.get("inlier_mask"), bool),
        match_scores=_optional_array(row.get("match_scores")),
        runtime_s=float(row.get("runtime_s", np.nan)),
        cpu_time_s=float(row.get("cpu_time_s", np.nan)),
        peak_vram_bytes=int(row.get("peak_vram_bytes", 0)),
        diagnostics=diagnostics,
    )


def command_register(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    root = _project_root(args.config)
    pairs, _ = _pairs_manifest(cfg, root)
    pipeline = cfg.pipeline(args.pipeline)
    pipeline_contract = json.dumps(
        {
            "version": pipeline.version,
            "checkpoint": pipeline.checkpoint,
            "options": pipeline.options,
            "provenance": pipeline.provenance,
        },
        sort_keys=True,
    )
    if "[VERIFY]" in pipeline_contract:
        raise CommandError(
            f"pipeline {pipeline.id!r} still has an unverified version/checkpoint; "
            "record exact upstream commit and weight hash before running"
        )
    try:
        registrar = load_registrar(pipeline, project_root=root)
    except AdapterUnavailable as exc:
        raise CommandError(str(exc)) from exc

    selected = _select_pairs(
        pairs,
        split=args.split,
        pair_ids=args.pair,
        dataset_ids=args.dataset,
        limit=args.limit,
    )
    directions = ("canonical", "reverse") if args.direction == "both" else (args.direction,)
    code_id = _registration_code_identity(root)
    env_id, checkpoint_id = _pipeline_contract_hashes(cfg, root, pipeline.id)
    jobs = []
    for _, row in selected.iterrows():
        _verify_pair_images(row, root)
        for direction in directions:
            pair = _pair_from_row(row, cfg, root, direction=direction)
            jobs.append(
                RegistrationJob(
                    pair=pair,
                    dataset_version=str(row["dataset_version"]),
                    pipeline_version=pipeline.version,
                    seed=cfg.splits.seed,
                    fold=int(row["fold"]),
                    is_development=bool(row["is_development"]),
                    sampling_probability=float(row["sampling_probability"]),
                    code_hash=code_id,
                    env_hash=env_id,
                    checkpoint_hash=checkpoint_id,
                    config_hash=cfg.hash,
                )
            )
    if not jobs:
        raise CommandError(f"no {args.split} pairs are available in the pair manifest")
    cache_root = cfg.paths.resolve(root)["cache_root"]
    runner = RegistrationRunner(
        registrar,
        ShardedTable(cache_root, "registrations"),
        StatusLedger(cache_root / "status.jsonl"),
        max_infrastructure_attempts=args.max_attempts,
    )
    summary = _run_with_progress(runner, jobs, label=pipeline.id,
                                 workers=getattr(args, "workers", 1),
                                 worker_threads=getattr(args, "worker_threads", 1))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_with_progress(
    runner: RegistrationRunner, jobs, *, label: str, workers: int = 1,
    worker_threads: int = 1, total: int | None = None, scratch_root: Path | None = None,
) -> dict[str, int]:
    """Drive the runner one job at a time so a long sweep reports progress.

    ``RegistrationRunner.run`` is silent by design; an unattended overnight
    sweep needs a heartbeat to distinguish slow work from a hung matcher. This
    calls the same per-job entry point, so job identity, retries, and cache
    contracts are unchanged.
    """
    if workers < 1 or worker_threads < 0:
        raise CommandError("workers must be positive and worker threads nonnegative", EXIT_USAGE)
    if total is None:
        total = len(jobs)
    if workers > 1 or not isinstance(jobs, list):
        requested = []
        def tracked():
            for job in jobs:
                requested.append(job)
                yield job
        started = time.perf_counter()
        computed = 0
        for job, result in runner.run_parallel(tracked(), workers=workers,
                                               worker_threads=worker_threads):
            if scratch_root is not None:
                for path in (job.pair.moving_path, job.pair.fixed_path):
                    if not path.resolve().is_relative_to(scratch_root.resolve()):
                        raise CommandError("E2 scratch path escaped its temporary directory")
                    path.unlink(missing_ok=True)
            computed += 1
            elapsed = time.perf_counter() - started
            print(f"[{computed:>5}/{total}] {label} {job.pair.pair_id} "
                  f"{job.pair.condition} {result.status.value} "
                  f"elapsed {elapsed / 60:.1f}m workers={workers}", flush=True)
        incomplete = runner.incomplete_jobs(requested)
        if incomplete:
            raise CommandError(f"{len(incomplete)} requested registration job(s) remain unresolved "
                               "after the fixed attempt policy; downstream stages blocked. "
                               f"Job ids: {[job.job_id for job in incomplete]}")
        print(f"computed {computed} of {total} job(s); the rest were already cached", flush=True)
        return runner.ledger.summary()
    started = time.perf_counter()
    computed = 0
    for index, job in enumerate(jobs, start=1):
        job_started = time.perf_counter()
        result = runner.run_one(job)
        executed = result is not None
        # Retry only technical failures, with the original job and seed and the
        # existing fixed attempt cap. Never retry a scientific no-output case.
        while result is not None and result.status.is_infrastructure:
            print(f"retryable infrastructure failure: {job.job_id} {result.status.value}",
                  flush=True)
            result = runner.run_one(job)
        elapsed = time.perf_counter() - job_started
        if result is None:
            state = "unresolved" if runner.incomplete_jobs([job]) else "cached"
        else:
            state = result.status.value
        computed += int(executed)
        done = time.perf_counter() - started
        remaining = (done / index) * (total - index)
        print(
            f"[{index:>5}/{total}] {label} {job.pair.pair_id} {job.pair.direction} "
            f"{state} {elapsed:6.2f}s  elapsed {done / 60:6.1f}m  eta {remaining / 60:6.1f}m",
            flush=True,
        )
    print(f"computed {computed} of {total} job(s); the rest were already cached", flush=True)
    incomplete = runner.incomplete_jobs(jobs)
    if incomplete:
        raise CommandError(
            f"{len(incomplete)} requested registration job(s) remain unresolved after "
            f"the fixed {runner.max_infrastructure_attempts}-attempt policy; "
            f"downstream stages blocked. Job ids: {[job.job_id for job in incomplete]}"
        )
    return runner.ledger.summary()


def _group_balanced_sample(pairs: pd.DataFrame, *, cap: int, seed: int) -> pd.DataFrame:
    """A seeded, group-balanced subset of at most ``cap`` pairs per dataset.

    §11 requires a fixed seeded sample of eligible groups and pairs, balanced by
    group and independent of outcomes. Groups are visited round-robin in a
    seeded order, so a subject with many pairs cannot dominate the sample and
    the selection cannot depend on how any case scored.
    """
    if cap <= 0:
        raise CommandError("--sample-cap must be positive", EXIT_USAGE)
    blocks: list[pd.DataFrame] = []
    for dataset_id, frame in pairs.groupby("dataset_id", sort=True):
        by_group: dict[str, list[str]] = {}
        for _, row in frame.sort_values("pair_id", kind="stable").iterrows():
            by_group.setdefault(str(row["group_id"]), []).append(str(row["pair_id"]))
        order = sorted(
            by_group, key=lambda g: short_hash({"dataset": str(dataset_id), "group": g, "seed": seed})
        )
        chosen: list[str] = []
        depth = 0
        while len(chosen) < cap and any(len(by_group[g]) > depth for g in order):
            for group in order:
                if len(chosen) >= cap:
                    break
                if len(by_group[group]) > depth:
                    chosen.append(by_group[group][depth])
            depth += 1
        blocks.append(frame[frame["pair_id"].astype(str).isin(set(chosen))])
    if not blocks:
        return pairs.iloc[0:0]
    return pd.concat(blocks).sort_values(["dataset_id", "pair_id"], kind="stable")


def _prepare_e2_pair(task):
    (row, cfg, root, temporary_root, reruns, translation_fraction,
     condition_prefix, pipeline, code_id, env_id, checkpoint_id) = task
    jobs = []
    _verify_pair_images(row, root)
    baseline = _pair_from_row(row, cfg, root, direction="canonical")
    with Image.open(baseline.moving_path) as image:
        moving = resample_to_frame(
            np.asarray(image.convert("RGB")), baseline.coordinates.moving
        )
    with Image.open(baseline.fixed_path) as image:
        fixed = resample_to_frame(
            np.asarray(image.convert("RGB")), baseline.coordinates.fixed
        )
    pair_seed = cfg.splits.seed ^ int(short_hash(baseline.pair_id, length=8), 16)
    maps = translation_perturbations(
        moving.shape[0],
        moving.shape[1],
        count=reruns,
        seed=pair_seed,
        max_fraction=translation_fraction,
    )
    fixed_map = np.eye(3, dtype=np.float64)
    for index, moving_map in enumerate(maps):
        condition = f"{condition_prefix}:{index:03d}"
        pair_dir = temporary_root / short_hash(
            {"pair": baseline.pair_id, "condition": condition}
        )
        pair_dir.mkdir(parents=True, exist_ok=True)
        moving_path = pair_dir / "moving.png"
        fixed_path = pair_dir / "fixed.png"
        Image.fromarray(warp_by_input_map(moving, moving_map)).save(moving_path)
        Image.fromarray(np.asarray(fixed, dtype=np.uint8)).save(fixed_path)
        perturbed = PairInput(
            dataset_id=baseline.dataset_id,
            pair_id=baseline.pair_id,
            group_id=baseline.group_id,
            group_basis=baseline.group_basis,
            moving_image_id=f"{baseline.moving_image_id}/{condition}",
            fixed_image_id=baseline.fixed_image_id,
            moving_path=moving_path,
            fixed_path=fixed_path,
            coordinates=CoordinateMetadata(
                moving=identity_frame(
                    f"{baseline.moving_image_id}/{condition}", moving.shape[:2]
                ),
                fixed=identity_frame(baseline.fixed_image_id, fixed.shape[:2]),
            ),
            acquisition_meta=baseline.acquisition_meta,
            direction="canonical",
            condition=condition,
        )
        jobs.append(
            RegistrationJob(
                pair=perturbed,
                dataset_version=str(row["dataset_version"]),
                pipeline_version=pipeline.version,
                seed=cfg.splits.seed,
                fold=int(row["fold"]),
                # Taken from the manifest, never assumed: E2 may now run
                # on confirmatory pairs, and a mislabelled split flag
                # would put a reserve row into a development selection.
                is_development=bool(row["is_development"]),
                sampling_probability=float(row["sampling_probability"]),
                code_hash=code_id,
                env_hash=env_id,
                checkpoint_hash=checkpoint_id,
                config_hash=cfg.hash,
                input_corrections=(moving_map, fixed_map),
            )
        )
    return jobs


def command_e2(args: argparse.Namespace) -> int:
    """Run B full registrations under deterministic moving-image translations.

    The perturbation draws are prefix-stable, so the B = 4 sensitivity variant
    of §7.2a reuses the first four reruns and only a larger B costs more
    registrations.
    """
    cfg = load_config(args.config)
    root = _project_root(args.config)
    pairs, _ = _pairs_manifest(cfg, root)
    if args.split == "confirmatory":
        try:
            require_confirmatory_access(
                cfg.paths.resolve(root)["manifests"],
                config_hash=cfg.hash,
                what="perturbing confirmatory pairs",
            )
        except FreezeError as exc:
            raise CommandError(str(exc)) from exc
    pipeline = cfg.pipeline(args.pipeline)
    try:
        base_registrar = load_registrar(pipeline, project_root=root)
    except AdapterUnavailable as exc:
        raise CommandError(str(exc)) from exc
    selected = _select_pairs(
        pairs,
        split=args.split,
        pair_ids=args.pair,
        dataset_ids=args.dataset,
        limit=args.limit,
    )
    if not args.pair and args.sample_cap:
        selected = _group_balanced_sample(
            selected, cap=args.sample_cap, seed=cfg.splits.seed
        )
    if selected.empty:
        raise CommandError(f"no {args.split} pairs match the E2 selectors")
    reruns = cfg.features.perturbation_B if args.reruns is None else args.reruns
    if reruns < 2:
        raise CommandError("--reruns must be at least two", EXIT_USAGE)
    if not 0.0 < args.translation_fraction < 0.5:
        raise CommandError("--translation-fraction must lie in (0, 0.5)", EXIT_USAGE)
    env_id, checkpoint_id = _pipeline_contract_hashes(cfg, root, pipeline.id)
    condition_prefix = (
        f"e2:translation-v1-f{args.translation_fraction:.6f}".rstrip("0").rstrip(".")
    )
    with tempfile.TemporaryDirectory(prefix="warpaudit-e2-") as temporary:
        temporary_root = Path(temporary)
        code_id = _e2_code_identity(root)
        preparation = (
            (row, cfg, root, temporary_root, reruns, args.translation_fraction,
             condition_prefix, pipeline, code_id, env_id, checkpoint_id)
            for _, row in selected.iterrows()
        )
        workers = getattr(args, "workers", 1)
        worker_threads = getattr(args, "worker_threads", 1)
        def prepared_jobs():
            for batch in ordered_map(_prepare_e2_pair, preparation, workers=workers,
                                     threads=worker_threads):
                yield from batch
        cache_root = cfg.paths.resolve(root)["cache_root"]
        runner = RegistrationRunner(
            base_registrar,
            ShardedTable(cache_root, "registrations"),
            StatusLedger(cache_root / "status.jsonl"),
            max_infrastructure_attempts=args.max_attempts,
        )
        summary = _run_with_progress(runner, prepared_jobs(), label=f"{pipeline.id} e2",
                                     workers=workers, worker_threads=worker_threads,
                                     total=len(selected) * reruns, scratch_root=temporary_root)
    print(
        json.dumps(
            {
                "jobs": len(selected) * reruns,
                "pairs": len(selected),
                "pipeline": pipeline.id,
                "distribution": condition_prefix,
                "ledger": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _annotation_points(
    row: pd.Series, root: Path, *, direction: str
) -> tuple[np.ndarray, np.ndarray]:
    raw_paths = row["annotation_paths"]
    if isinstance(raw_paths, str):
        raw_paths = json.loads(raw_paths)
    if not isinstance(raw_paths, list | tuple):
        raise CommandError(f"{row['pair_id']}: annotation_paths is not a sequence")
    raw_digests = row["annotation_sha256"]
    if isinstance(raw_digests, str):
        raw_digests = json.loads(raw_digests)
    if not isinstance(raw_digests, list | tuple) or len(raw_digests) != len(raw_paths):
        raise CommandError(f"{row['pair_id']}: annotation hashes do not match annotation paths")
    paths = tuple(_resolve(root, str(path)) for path in raw_paths)
    for path, expected in zip(paths, raw_digests, strict=True):
        if not path.is_file():
            raise CommandError(f"{row['pair_id']}: annotation is absent: {path}")
        actual = file_digest(path)
        if actual != str(expected):
            raise CommandError(
                f"{row['pair_id']}: annotation changed since audit-data; "
                f"expected {expected}, got {actual}"
            )
    try:
        if len(paths) == 1:
            moving, fixed = load_fire_control_points(paths[0])
        elif len(paths) == 2:
            moving, fixed = load_coph100_control_points(paths[0], paths[1])
        else:
            raise ValueError(f"expected one or two annotation files, got {len(paths)}")
    except (OSError, ValueError) as exc:
        raise CommandError(f"{row['pair_id']}: invalid annotation: {exc}") from exc
    return (fixed, moving) if direction == "reverse" else (moving, fixed)


def _compute_label_task(task):
    registration_row, pair_row, cfg, root = task
    direction = str(registration_row["direction"])
    pair = _pair_from_row(pair_row, cfg, root, direction=direction)
    result = _result_from_row(registration_row)
    points_moving, points_fixed = _annotation_points(pair_row, root, direction=direction)
    original_transform = (
        None
        if result.forward_moving_to_fixed is None
        else to_original_frame(result.forward_moving_to_fixed, pair.coordinates)
    )
    errors = point_errors(
        original_transform,
        points_moving,
        points_fixed,
        pair.coordinates.fixed,
        normalised_thresholds=(cfg.labels.tau_primary, *cfg.labels.tau_secondary),
    )
    outcome = case_outcome(
        result.status,
        errors,
        tau=cfg.labels.tau_primary,
        cap=cfg.labels.bounded_loss_cap,
    )
    return (
        {
            "job_id": str(registration_row["job_id"]),
            "dataset_id": pair.dataset_id,
            "pair_id": pair.pair_id,
            "annotation_kind": str(pair_row["annotation_kind"]),
            "annotation_provenance": str(pair_row["annotation_provenance"]),
            **errors.as_row(),
            "tau": outcome.tau,
            "silent_failure": outcome.silent_failure,
            "operational_failure": outcome.operational_failure,
            "eligible_for_acceptance": outcome.eligible_for_acceptance,
            "bounded_loss": outcome.bounded_loss,
            "landmark_success_fraction_12p5px": landmark_success_fraction(
                errors, cfg.labels.r8_landmark_success_px
            ),
        }
    )


def command_labels(args: argparse.Namespace) -> int:
    """Join annotations to cached registrations in the restricted evaluator.

    Development outcomes are always readable. Confirmatory outcomes are gated
    by the G1 freeze (§3.3): the protocol must be fixed and reviewed before any
    of it is seen, so this refuses rather than warns.
    """
    cfg = load_config(args.config)
    root = _project_root(args.config)
    pair_rows, _ = _pairs_manifest(cfg, root)
    if args.split == "confirmatory":
        try:
            require_confirmatory_access(
                cfg.paths.resolve(root)["manifests"],
                config_hash=cfg.hash,
                what="scoring confirmatory registrations",
            )
        except FreezeError as exc:
            raise CommandError(str(exc)) from exc
    selected_pairs = _select_pairs(
        pair_rows,
        split=args.split,
        pair_ids=args.pair,
        dataset_ids=args.dataset,
        limit=None,
    )
    selected_ids = set(selected_pairs["pair_id"].astype(str))
    if not selected_ids:
        raise CommandError(f"no {args.split} pairs match the requested selectors")
    try:
        cfg.pipeline(args.pipeline)
    except KeyError as exc:
        raise CommandError(str(exc), EXIT_USAGE) from exc
    registrations = ShardedTable(cfg.paths.resolve(root)["cache_root"], "registrations").load()
    if registrations.empty:
        raise CommandError("registration cache is empty; no label rows were written")
    registrations = _current_registration_rows(registrations, pair_rows, cfg, root)
    registrations = registrations[
        registrations["pair_id"].astype(str).isin(selected_ids)
        & (registrations["pipeline_id"].astype(str) == args.pipeline)
        & (registrations["condition"].astype(str) == "clean")
        & registrations["direction"].astype(str).isin(
            args.direction or ("canonical", "reverse")
        )
        & (registrations["is_development"].astype(bool) == (args.split == "development"))
    ].sort_values("job_id", kind="stable")
    if registrations.empty:
        raise CommandError(f"no matching {args.split} registrations are cached")

    manifest_by_pair = {str(row["pair_id"]): row for _, row in selected_pairs.iterrows()}
    output: list[dict[str, object]] = []
    tasks = ((row, manifest_by_pair[str(row["pair_id"])], cfg, root)
             for _, row in registrations.iterrows())
    output = list(ordered_map(_compute_label_task, tasks,
                              workers=getattr(args, "workers", 1),
                              threads=getattr(args, "worker_threads", 1)))
    labels = ShardedTable(cfg.paths.resolve(root)["cache_root"], "labels")
    written = labels.append(output, shard_hint=args.pipeline)
    print(json.dumps({"rows": len(output), "path": str(written)}, indent=2, sort_keys=True))
    return 0


def _png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def command_render_overlays(args: argparse.Namespace) -> int:
    """Render source-pixel panels for a local development-only geometry review."""
    cfg = load_config(args.config)
    root = _project_root(args.config)
    pair_rows, _ = _pairs_manifest(cfg, root)
    selected = _select_pairs(
        pair_rows,
        split="development",
        pair_ids=args.pair,
        dataset_ids=args.dataset,
        limit=args.limit,
    )
    if selected.empty:
        raise CommandError("no development pairs match the requested overlay selectors")
    selected_ids = set(selected["pair_id"].astype(str))
    cache_root = cfg.paths.resolve(root)["cache_root"]
    registrations = ShardedTable(cache_root, "registrations").load()
    if registrations.empty:
        raise CommandError("registration cache is empty; no overlays were rendered")
    registrations = _current_registration_rows(registrations, pair_rows, cfg, root)
    registrations = registrations[
        registrations["pair_id"].astype(str).isin(selected_ids)
        & (registrations["pipeline_id"].astype(str) == args.pipeline)
        & (registrations["direction"].astype(str) == args.direction)
        & (registrations["condition"].astype(str) == "clean")
        & registrations["is_development"].astype(bool)
    ].sort_values(["dataset_id", "pair_id", "job_id"], kind="stable")
    registrations = registrations.drop_duplicates(["pair_id", "direction"], keep="last")
    if registrations.empty:
        raise CommandError("no matching clean development registrations are cached")

    labels = ShardedTable(cache_root, "labels").load()
    label_by_job = (
        {}
        if labels.empty
        else {str(row["job_id"]): row for _, row in labels.iterrows()}
    )
    manifest_by_pair = {str(row["pair_id"]): row for _, row in selected.iterrows()}
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panels: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for _, registration_row in registrations.iterrows():
        pair_row = manifest_by_pair[str(registration_row["pair_id"])]
        _verify_pair_images(pair_row, root)
        pair = _pair_from_row(pair_row, cfg, root, direction=args.direction)
        result = _result_from_row(registration_row)
        if result.forward_moving_to_fixed is None:
            records.append(
                {
                    "job_id": str(registration_row["job_id"]),
                    "pair_id": pair.pair_id,
                    "status": result.status.value,
                    "rendered": False,
                }
            )
            continue
        with Image.open(pair.moving_path) as image:
            moving = np.asarray(image.convert("RGB"))
        with Image.open(pair.fixed_path) as image:
            fixed = np.asarray(image.convert("RGB"))
        job_id = str(registration_row["job_id"])
        label = label_by_job.get(job_id)
        tre = ""
        if label is not None and bool(label.get("tre_defined", False)):
            tre = f"; TRE={float(label['tre_px']):.3f}px"
        panel, overlap = render_registration_panel(
            moving,
            fixed,
            result.forward_moving_to_fixed,
            pair.coordinates,
            title=f"{pair.pair_id}; {args.pipeline}; {args.direction}{tre}",
        )
        filename = f"{len(panels):03d}-{job_id}.png"
        atomic_write_bytes(output_dir / filename, _png_bytes(panel))
        panels.append(panel)
        records.append(
            {
                "job_id": job_id,
                "dataset_id": pair.dataset_id,
                "pair_id": pair.pair_id,
                "pipeline_id": args.pipeline,
                "direction": args.direction,
                "status": result.status.value,
                "overlap_fraction": overlap,
                "tre_px": None if not tre else float(label["tre_px"]),
                "file": filename,
                "rendered": True,
            }
        )
    if not panels:
        raise CommandError("matching registrations have no renderable transforms")
    sheet = contact_sheet(panels, columns=args.columns, thumbnail_width=args.thumbnail_width)
    sheet_path = atomic_write_bytes(output_dir / "contact_sheet.png", _png_bytes(sheet))
    atomic_write_text(
        output_dir / "index.json",
        json.dumps(
            {
                "generated_at": _utc_now(),
                "source_pixels": True,
                "redistribute": False,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(json.dumps({"rendered": len(panels), "contact_sheet": str(sheet_path)}, indent=2))
    return 0


def command_probe_adapter(args: argparse.Namespace) -> int:
    """Exercise one isolated matcher without datasets or cached outcomes."""
    cfg = load_config(args.config)
    root = _project_root(args.config)
    pipeline = cfg.pipeline(args.pipeline)
    try:
        registrar = load_registrar(pipeline, project_root=root)
    except (AdapterUnavailable, OSError, ValueError) as exc:
        raise CommandError(str(exc)) from exc

    rng = np.random.default_rng(20260907)
    height = width = 256
    moving = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    # Blend in larger structures so the probe exercises both detector context
    # and an exactly known translation. This is an engineering smoke fixture,
    # never registration-performance evidence.
    moving[30:70, 25:110] = (230, 40, 30)
    moving[120:190, 145:210] = (20, 210, 90)
    fixed = np.roll(moving, shift=(3, 5), axis=(0, 1))
    with tempfile.TemporaryDirectory(prefix="warpaudit-probe-") as tmp:
        tmp_path = Path(tmp)
        moving_path, fixed_path = tmp_path / "moving.png", tmp_path / "fixed.png"
        Image.fromarray(moving).save(moving_path)
        Image.fromarray(fixed).save(fixed_path)
        moving_frame = make_frame("probe-moving", (height, width), long_edge=width)
        fixed_frame = make_frame("probe-fixed", (height, width), long_edge=width)
        pair = PairInput(
            dataset_id="adapter-probe",
            pair_id="adapter-probe/translation",
            group_id="adapter-probe/group",
            group_basis="image_component",
            moving_image_id="probe-moving",
            fixed_image_id="probe-fixed",
            moving_path=moving_path,
            fixed_path=fixed_path,
            coordinates=CoordinateMetadata(moving=moving_frame, fixed=fixed_frame),
            acquisition_meta={"modality": "synthetic engineering fixture"},
        )
        try:
            result = registrar.register(pair, cfg.splits.seed)
            repeated = registrar.register(pair, cfg.splits.seed)
        except Exception as exc:
            raise CommandError(f"adapter probe failed: {exc}") from exc

    repeatability_px = float("inf")
    correspondence_repeatable = False
    if (
        result.status is RegistrationStatus.OK
        and repeated.status is RegistrationStatus.OK
        and result.matches_moving is not None
        and repeated.matches_moving is not None
        and result.matches_fixed is not None
        and repeated.matches_fixed is not None
        and result.matches_moving.shape == repeated.matches_moving.shape
        and result.matches_fixed.shape == repeated.matches_fixed.shape
    ):
        correspondence_repeatable = bool(
            np.array_equal(result.matches_moving, repeated.matches_moving)
            and np.array_equal(result.matches_fixed, repeated.matches_fixed)
        )
        grid = prespecified_grid(fixed_frame, size=8).points_fixed
        first_prediction = result.forward_moving_to_fixed.apply(grid)
        second_prediction = repeated.forward_moving_to_fixed.apply(grid)
        repeatability_px = float(
            np.max(np.linalg.norm(first_prediction - second_prediction, axis=1))
        )

    summary = {
        "generated_at": _utc_now(),
        "pipeline_id": pipeline.id,
        "status": result.status.value,
        "n_matches": result.n_matches,
        "n_inliers": result.n_inliers,
        "runtime_s": result.runtime_s,
        "provenance": result.diagnostics.get("provenance", {}),
        "matcher_diagnostics": result.diagnostics.get("matcher_diagnostics", {}),
        "repeatability": {
            "same_correspondence_bytes": correspondence_repeatable,
            "max_grid_disagreement_px": repeatability_px,
            "tolerance_px": args.repeatability_tolerance_px,
        },
        "fixture": "synthetic translation; engineering smoke test only",
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = _resolve(root, args.output)
        atomic_write_text(output, rendered)
        print(output)
    else:
        print(rendered, end="")
    if result.status is not RegistrationStatus.OK or repeated.status is not RegistrationStatus.OK:
        raise CommandError(
            f"adapter probe returned statuses {result.status.value}, {repeated.status.value}"
        )
    if repeatability_px > args.repeatability_tolerance_px:
        raise CommandError(
            f"adapter repeatability disagreement {repeatability_px:.6g}px exceeds "
            f"{args.repeatability_tolerance_px:.6g}px"
        )
    return 0


def _compute_feature_task(task):
    """Compute one case in isolation; workers never read labels or write caches."""
    index, row, pair, result, sources, needed, cfg = task
    started = time.perf_counter()
    pipeline = cfg.pipeline(result.pipeline_id)
    bundles = compute_families(
        build_signal_context(
            pair=pair,
            result=result,
            sources=sources,
            families=needed,
            grid_size=cfg.geometry.grid_size,
            bootstrap_B=cfg.features.bootstrap_B,
            perturbation_B=cfg.features.perturbation_B,
            perturbation_B_sensitivity=cfg.features.perturbation_B_sensitivity,
            seed=int(row["seed"]),
            fitting_policy={
                "threshold_px": pipeline.ransac_threshold_px,
                "max_iters": pipeline.max_iters,
                "confidence": pipeline.confidence,
                "min_matches": pipeline.min_matches,
                "transform_family": pipeline.transform_family,
                "tps_regularisation": pipeline.options.get("tps_regularisation", 1e-3),
            },
            load_image=lambda name: _working_frame_image(pair, name),
        ),
        needed,
    )
    return index, row, result, bundles, time.perf_counter() - started


_FEATURE_THREAD_LIMITER = None


def _initialise_feature_worker(threads):
    """Avoid nested native thread pools competing with the case workers."""
    global _FEATURE_THREAD_LIMITER
    if threads:
        _FEATURE_THREAD_LIMITER = threadpool_limits(limits=threads)


def _feature_results(tasks, workers, worker_threads=0):
    """Bound in-flight work and return cases in their original manifest order."""
    if workers == 1:
        with threadpool_limits(limits=worker_threads or None):
            yield from map(_compute_feature_task, tasks)
        return
    # Explicit spawn keeps Windows and POSIX on the same isolation model.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialise_feature_worker, initargs=(worker_threads,),
    ) as pool:
        tasks = iter(tasks)
        pending = deque(
            pool.submit(_compute_feature_task, task)
            for task in itertools.islice(tasks, 2 * workers)
        )
        while pending:
            yield pending.popleft().result()
            task = next(tasks, None)
            if task is not None:
                pending.append(pool.submit(_compute_feature_task, task))


def command_features(args: argparse.Namespace) -> int:
    workers = getattr(args, "workers", 1)
    worker_threads = getattr(args, "worker_threads", 0)
    if workers < 1:
        raise CommandError("--workers must be positive", EXIT_USAGE)
    if worker_threads < 0:
        raise CommandError("--worker-threads must be nonnegative", EXIT_USAGE)
    cfg = load_config(args.config)
    root = _project_root(args.config)
    pair_rows, _ = _pairs_manifest(cfg, root)
    requested = list(args.families)
    if getattr(args, "families_from_gate", False):
        # Keeps the reserve sweep aligned with whatever the development gate
        # actually froze, so a gate that retains E1 cannot leave the frozen
        # composite un-extractable over the confirmatory split.
        gate_path = cfg.paths.resolve(root)["manifests"] / "feature_gate.json"
        if not gate_path.is_file():
            raise CommandError(
                "--families-from-gate needs the development gate; run "
                "`warpaudit diagnose-development` first"
            )
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        frozen = list(gate.get("frozen_feature_families", ()))
        if not frozen:
            raise CommandError(f"{gate_path} names no frozen feature families")
        requested = sorted(set(requested) | set(frozen))
    if not requested:
        raise CommandError("no families requested; pass --families or --families-from-gate",
                           EXIT_USAGE)
    args.families = requested
    unknown = sorted(set(args.families) - set(available_families()))
    if unknown:
        raise CommandError(
            f"unimplemented signal families {unknown}; available: {available_families()}"
        )
    cache_root = cfg.paths.resolve(root)["cache_root"]
    registrations = ShardedTable(cache_root, "registrations").load()
    if registrations.empty:
        raise CommandError("registration cache is empty; no feature rows were written")
    registrations = _current_registration_rows(registrations, pair_rows, cfg, root)
    wanted_pairs = set(
        pair_rows.loc[
            pair_rows["is_development"].astype(bool) == (args.split == "development"), "pair_id"
        ].astype(str)
    )
    registrations = registrations[registrations["pair_id"].astype(str).isin(wanted_pairs)]
    if args.pair:
        unknown = set(args.pair) - set(pair_rows["pair_id"].astype(str))
        if unknown:
            raise CommandError(f"unknown pair selector(s): {sorted(unknown)}", EXIT_USAGE)
        registrations = registrations[
            registrations["pair_id"].astype(str).isin(set(args.pair))
        ]
    if args.dataset:
        unknown = set(args.dataset) - {dataset.id for dataset in cfg.datasets}
        if unknown:
            raise CommandError(f"unknown dataset selector(s): {sorted(unknown)}", EXIT_USAGE)
        registrations = registrations[
            registrations["dataset_id"].astype(str).isin(set(args.dataset))
        ]
    if args.pipeline:
        unknown = set(args.pipeline) - {pipeline.id for pipeline in cfg.pipelines}
        if unknown:
            raise CommandError(f"unknown pipeline selector(s): {sorted(unknown)}", EXIT_USAGE)
    # Selectors choose which jobs to compute. They must not shrink the context a
    # signal reads from: family C needs the reverse direction and family F needs
    # the other pipeline's row for the same pair, so narrowing the pool by
    # direction or pipeline would silently turn a computable feature into a
    # missing one.
    clean_registrations = registrations[registrations["condition"].astype(str) == "clean"]
    baseline = clean_registrations
    if args.pipeline:
        baseline = baseline[baseline["pipeline_id"].astype(str).isin(set(args.pipeline))]
    if args.direction:
        baseline = baseline[baseline["direction"].astype(str).isin(set(args.direction))]
    if baseline.empty:
        raise CommandError(f"no clean cached registrations for split {args.split!r}")
    manifest_by_pair = {str(row["pair_id"]): row for _, row in pair_rows.iterrows()}
    lookup = {
        (
            str(row["pair_id"]),
            str(row["pipeline_id"]),
            str(row["direction"]),
            int(row["seed"]),
        ): row
        for _, row in clean_registrations.iterrows()
    }
    feature_table = ShardedTable(cache_root, "features", key_column="feature_id")
    cached = feature_table.load()
    existing = (
        set() if cached.empty else set(cached["feature_id"].astype(str))
    )
    # A family's values are written together, so the presence of any current row
    # for (job, family) means that family is complete for that job. Checking it
    # up front lets a resumed sweep skip the computation instead of repeating it
    # and discarding the result at the write step.
    completed_families: set[tuple[str, str]] = set()
    if not cached.empty and {"job_id", "family", "code_hash", "config_hash"} <= set(cached):
        current = cached[
            (cached["code_hash"].astype(str) == _feature_code_identity(root))
            & (cached["config_hash"].astype(str) == cfg.hash)
        ]
        completed_families = {
            (str(job), str(family))
            for job, family in zip(current["job_id"], current["family"], strict=True)
        }
    output_rows: list[dict[str, object]] = []
    feature_code_hash = _feature_code_identity(root)

    total_jobs = len(baseline)
    sweep_started = time.perf_counter()
    flush_after = max(1, args.flush_every)
    written = 0

    def flush(rows: list[dict[str, object]]) -> None:
        """Publish a whole number of jobs' features, never part of one.

        Buffering the entire sweep keeps a batch logically atomic but loses
        hours of work to one interruption. Flushing on job boundaries keeps the
        same guarantee -- a job's features are complete or absent -- while
        letting a resumed run skip what already landed.
        """
        nonlocal written
        if rows:
            feature_table.append(list(rows), shard_hint=cfg.hash)
            written += len(rows)
            rows.clear()

    skipped = 0

    def tasks():
        nonlocal skipped
        for index, (_, row) in enumerate(baseline.iterrows(), start=1):
            needed = tuple(
                family
                for family in args.families
                if (str(row["job_id"]), family) not in completed_families
            )
            if not needed:
                skipped += 1
                continue
            pair_row = manifest_by_pair[str(row["pair_id"])]
            pair = _pair_from_row(pair_row, cfg, root, direction=str(row["direction"]))
            result = _result_from_row(row)
            reverse_direction = "reverse" if pair.direction == "canonical" else "canonical"
            reverse_row = lookup.get(
                (pair.pair_id, result.pipeline_id, reverse_direction, int(row["seed"]))
            )
            sources = ContextSources(
                reverse=None if reverse_row is None else _result_from_row(reverse_row),
                perturbations={
                    str(aux["condition"]): _result_from_row(aux)
                    for _, aux in registrations[
                        (registrations["pair_id"].astype(str) == pair.pair_id)
                        & (registrations["pipeline_id"].astype(str) == result.pipeline_id)
                        & registrations["condition"].astype(str).str.startswith("e2:")
                    ].iterrows()
                },
                same_pair_other_pipelines={
                    str(other["pipeline_id"]): _result_from_row(other)
                    for _, other in clean_registrations[
                        (clean_registrations["pair_id"].astype(str) == pair.pair_id)
                        & (clean_registrations["direction"].astype(str) == pair.direction)
                        & (clean_registrations["pipeline_id"].astype(str) != result.pipeline_id)
                        & (clean_registrations["seed"].astype(int) == int(row["seed"]))
                    ].iterrows()
                },
            )
            yield index, row, pair, result, sources, needed, cfg

    print(f"feature execution: {workers} worker(s), "
          f"native threads per worker: {worker_threads or 'library default'}, "
          "one cache writer", flush=True)
    completed_now = 0
    for index, row, result, bundles, elapsed in _feature_results(tasks(), workers, worker_threads):
        job_rows: list[dict[str, object]] = []
        for family, bundle in bundles.items():
            for name, value in bundle.values.items():
                feature_hash = short_hash(
                    {
                        "family": family,
                        "name": name,
                        "definition_version": value.definition_version,
                        "code_hash": feature_code_hash,
                    }
                )
                feature_id = short_hash(
                    {
                        "job_id": row["job_id"],
                        "feature_hash": feature_hash,
                        "config_hash": cfg.hash,
                    }
                )
                if feature_id in existing:
                    continue
                job_rows.append(
                    {
                        "feature_id": feature_id,
                        "job_id": row["job_id"],
                        "family": family,
                        "feature_name": name,
                        "value": value.value,
                        "available": value.available,
                        "reason": value.reason,
                        "unit": value.unit,
                        "definition_version": value.definition_version,
                        "incremental_wall_s": bundle.wall_time_s,
                        "incremental_cpu_s": bundle.cpu_time_s,
                        "feature_hash": feature_hash,
                        "code_hash": feature_code_hash,
                        "config_hash": cfg.hash,
                    }
                )
        output_rows.extend(job_rows)
        completed_now += 1
        done = time.perf_counter() - sweep_started
        remaining = (done / completed_now) * max(0, total_jobs - skipped - completed_now)
        print(
            f"[{index:>5}/{total_jobs}] {result.pipeline_id} {row['pair_id']} "
            f"{len(job_rows):>3} new  {elapsed:6.2f}s  elapsed {done / 60:6.1f}m  "
            f"eta {remaining / 60:6.1f}m",
            flush=True,
        )
        if completed_now % flush_after == 0:
            flush(output_rows)
    flush(output_rows)
    print(
        f"features: {written} new row(s); {skipped} of {total_jobs} job(s) were already "
        "complete for the requested families"
    )
    return 0


def _working_frame_image(pair: PairInput, which: str) -> np.ndarray:
    """Decode one side of a pair into its declared working frame."""
    path = pair.moving_path if which == "moving" else pair.fixed_path
    frame = pair.coordinates.moving if which == "moving" else pair.coordinates.fixed
    with Image.open(path) as image:
        original = np.asarray(image.convert("RGB"), dtype=np.float64)
    return resample_to_frame(original, frame)


def _feature_column(features: pd.DataFrame, family: str, name: str) -> pd.Series:
    """Return one current long-format feature as a job-indexed numeric series."""
    rows = features[
        (features["family"].astype(str) == family)
        & (features["feature_name"].astype(str) == name)
    ].copy()
    if rows.empty:
        return pd.Series(dtype=np.float64, name=name)
    rows = rows.drop_duplicates("job_id", keep="last").set_index("job_id")
    values = pd.to_numeric(rows["value"], errors="coerce")
    values[~rows["available"].astype(bool)] = np.nan
    values.name = name
    return values


def command_diagnose_development(args: argparse.Namespace) -> int:
    """Classify C and E1 using development rows only and freeze the feature gate."""
    cfg = load_config(args.config)
    root = _project_root(args.config)
    pair_rows, _ = _pairs_manifest(cfg, root)
    cache_root = cfg.paths.resolve(root)["cache_root"]
    all_registrations = ShardedTable(cache_root, "registrations").load()
    all_registrations = _current_registration_rows(all_registrations, pair_rows, cfg, root)
    registrations = all_registrations[
        all_registrations["is_development"].astype(bool)
        & (all_registrations["direction"].astype(str) == "canonical")
        & (all_registrations["condition"].astype(str) == "clean")
    ].drop_duplicates("job_id", keep="last")
    if registrations.empty:
        raise CommandError("no current clean canonical development registrations")

    features = ShardedTable(cache_root, "features", key_column="feature_id").load()
    required_feature_columns = {"job_id", "code_hash", "config_hash"}
    if features.empty or not required_feature_columns <= set(features):
        raise CommandError("feature cache is absent or predates schema 1.1.0")
    features = features[
        features["job_id"].astype(str).isin(set(registrations["job_id"].astype(str)))
        & (features["code_hash"].astype(str) == _feature_code_identity(root))
        & (features["config_hash"].astype(str) == cfg.hash)
    ]
    missing_families = {"C", "E1"} - set(features["family"].astype(str))
    if missing_families:
        raise CommandError(
            f"current development feature cache lacks {sorted(missing_families)}; "
            "run `warpaudit features --split development --families C E1`"
        )
    labels = ShardedTable(cache_root, "labels").load()
    if labels.empty:
        raise CommandError("development label cache is absent")
    labels = labels[
        labels["job_id"].astype(str).isin(set(registrations["job_id"].astype(str)))
    ].drop_duplicates("job_id", keep="last")

    cycle_values = _feature_column(features, "C", "cycle_median")
    correspondence = _feature_column(features, "E1", "bootstrap_spread_valid_mean")
    seed_only = _feature_column(features, "E1", "seed_only_spread_valid_mean")
    error = labels.set_index("job_id")["tre_norm"].rename("tre_norm")

    cycle_decisions = {}
    e1_decisions = {}
    for pipeline in cfg.pipelines:
        if not pipeline.in_common_block:
            continue
        jobs = registrations.loc[
            registrations["pipeline_id"].astype(str) == pipeline.id, "job_id"
        ].astype(str)
        cycle = cycle_values.reindex(jobs).to_numpy(dtype=np.float64)
        cycle_decisions[pipeline.id] = classify_cycle(
            cycle, tolerance=cfg.features.cycle_tolerance
        )
        joined = pd.concat(
            (
                correspondence.reindex(jobs),
                seed_only.reindex(jobs),
                error.reindex(jobs),
            ),
            axis=1,
        )
        e1_decisions[pipeline.id] = classify_e1(
            joined.iloc[:, 0].to_numpy(dtype=np.float64),
            joined.iloc[:, 1].to_numpy(dtype=np.float64),
            joined.iloc[:, 2].to_numpy(dtype=np.float64),
            repeatability_tolerance=cfg.features.e1_repeatability_tolerance,
            seed_dominance_max_ratio=cfg.features.e1_seed_dominance_max_ratio,
            min_error_spearman=cfg.features.e1_min_error_spearman,
        )

    e1_classes = {decision.classification for decision in e1_decisions.values()}
    if e1_classes == {"informative"}:
        gate = "informative"
        frozen_families = ["A", "B", "E1"]
    elif "unresolved" in e1_classes:
        gate = "unresolved"
        frozen_families = []
    else:
        gate = "degenerate"
        frozen_families = list(cfg.features.fallback_families_if_e1_degenerate)

    generated_at = _utc_now()
    reports = cfg.paths.resolve(root)["reports"]
    cycle_lines = [
        "# Development cycle diagnostic",
        "",
        f"Generated: {generated_at}",
        "",
        "Independent reverse registrations are used; analytic inversion is only a comparison.",
        "",
        "| Pipeline | Classification | Finite cases | Median | IQR | Tolerance |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pipeline_id, decision in cycle_decisions.items():
        c = decision.criteria
        cycle_lines.append(
            f"| `{pipeline_id}` | {decision.classification} | {int(c['n_cases'])} | "
            f"{float(c.get('median', np.nan)):.6g} | {float(c.get('iqr', np.nan)):.6g} | "
            f"{float(c['tolerance']):.6g} |"
        )
    cycle_lines += [
        "",
        "Classification applies the frozen median-and-across-case-IQR rule in normalized "
        "fixed-frame diagonal units. An `informative` result means the cycle signal is "
        "non-degenerate; it does not establish predictive utility.",
        "",
    ]
    atomic_write_text(reports / "cycle_diagnostic.md", "\n".join(cycle_lines))

    e1_lines = [
        "# Development E1 diagnostic",
        "",
        f"Generated: {generated_at}",
        "",
        "| Pipeline | Classification | Cases | Median bootstrap spread | Median seed-only "
        "spread | Seed ratio | Spearman with TRE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for pipeline_id, decision in e1_decisions.items():
        c = decision.criteria
        e1_lines.append(
            f"| `{pipeline_id}` | {decision.classification} | {int(c['n_cases'])} | "
            f"{float(c.get('median_correspondence_spread', np.nan)):.6g} | "
            f"{float(c.get('median_seed_only_spread', np.nan)):.6g} | "
            f"{float(c.get('seed_dominance_ratio', np.nan)):.6g} | "
            f"{float(c.get('error_spearman', np.nan)):.6g} |"
        )
        if decision.reasons:
            e1_lines.append(f"|  | Reasons |  |  |  |  | {'; '.join(decision.reasons)} |")
    e1_lines += [
        "",
        f"Common-block gate: **{gate}**.",
        "",
        "Frozen feature families: "
        + (", ".join(f"`{family}`" for family in frozen_families) or "not frozen"),
        "",
        "The gate uses only development labels. Confirmatory rows are neither loaded nor scored.",
        "",
    ]
    atomic_write_text(reports / "e1_diagnostic.md", "\n".join(e1_lines))

    e2_columns = {
        name: _feature_column(features, "E2", name)
        for name in (
            "perturbation_valid_fits",
            "perturbation_invalid_fit_fraction",
            "perturbation_spread_mean",
            "perturbation_spread_p95",
            "perturbation_common_support_fraction",
            "perturbation_spread_common_mean",
        )
    }
    e2_rows: list[dict[str, object]] = []
    for _, baseline in registrations.iterrows():
        job_id = str(baseline["job_id"])
        valid_fits = e2_columns["perturbation_valid_fits"].get(job_id, np.nan)
        if not np.isfinite(valid_fits):
            continue
        auxiliaries = all_registrations[
            (all_registrations["pair_id"].astype(str) == str(baseline["pair_id"]))
            & (
                all_registrations["pipeline_id"].astype(str)
                == str(baseline["pipeline_id"])
            )
            & all_registrations["condition"].astype(str).str.startswith("e2:")
        ]
        runtimes = pd.to_numeric(auxiliaries["runtime_s"], errors="coerce")
        e2_rows.append(
            {
                "pipeline_id": str(baseline["pipeline_id"]),
                "dataset_id": str(baseline["dataset_id"]),
                "pair_id": str(baseline["pair_id"]),
                "attempted": int(len(auxiliaries)),
                "valid_fits": int(valid_fits),
                "invalid_fraction": float(
                    e2_columns["perturbation_invalid_fit_fraction"].get(job_id, np.nan)
                ),
                "spread_mean": float(
                    e2_columns["perturbation_spread_mean"].get(job_id, np.nan)
                ),
                "spread_p95": float(
                    e2_columns["perturbation_spread_p95"].get(job_id, np.nan)
                ),
                "common_support": float(
                    e2_columns["perturbation_common_support_fraction"].get(job_id, np.nan)
                ),
                "spread_common_mean": float(
                    e2_columns["perturbation_spread_common_mean"].get(job_id, np.nan)
                ),
                "total_runtime_s": float(runtimes.sum()),
                "median_runtime_s": float(runtimes.median()),
                "conditions": sorted(set(auxiliaries["condition"].astype(str))),
            }
        )
    e2_lines = [
        "# Development E2 smoke test",
        "",
        f"Generated: {generated_at}",
        "",
        "This smoke test adapts the transformation-equivariance baseline of Tian, Hu, "
        "and Iglesias to 2-D sparse retinal registration. It applies B=8 deterministic "
        "moving-image translations drawn from a uniform square with maximum displacement "
        "1.5% of the shorter working-image side. Every perturbation reruns the full matcher. "
        "The estimated map is corrected as `inv(P_f) @ T_perturbed @ P_m` before spread is "
        "measured. This translation-only smoke distribution is not the focused-study "
        "perturbation freeze.",
        "",
        "| Pipeline | Dataset | Pair | Fits | Invalid | Mean spread | P95 spread | Common "
        "support | Total rerun s | Median rerun s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in e2_rows:
        e2_lines.append(
            f"| `{row['pipeline_id']}` | {row['dataset_id']} | `{row['pair_id']}` | "
            f"{row['valid_fits']}/{row['attempted']} | {float(row['invalid_fraction']):.3f} | "
            f"{float(row['spread_mean']):.6g} | {float(row['spread_p95']):.6g} | "
            f"{float(row['common_support']):.3f} | {float(row['total_runtime_s']):.3f} | "
            f"{float(row['median_runtime_s']):.3f} |"
        )
    if not e2_rows:
        e2_lines += ["| — | — | No current E2 smoke evidence | — | — | — | — | — | — | — |"]
    e2_lines += [
        "",
        "Spread is normalized by the fixed working-frame diagonal. Invalid fits and lost "
        "common support remain explicit; no failed perturbation is dropped.",
        "",
    ]
    atomic_write_text(reports / "e2_smoke_test.md", "\n".join(e2_lines))

    manifest = {
        "generated_at": generated_at,
        "config_hash": cfg.hash,
        "code_hash": _code_identity(root),
        "registration_code_hash": _registration_code_identity(root),
        "feature_code_hash": _feature_code_identity(root),
        "split": "development",
        "direction": "canonical",
        "condition": "clean",
        "cycle": {key: asdict(value) for key, value in cycle_decisions.items()},
        "e1": {key: asdict(value) for key, value in e1_decisions.items()},
        "e2_smoke": e2_rows,
        "common_block_e1_gate": gate,
        "frozen_feature_families": frozen_families,
    }
    gate_path = cfg.paths.resolve(root)["manifests"] / "feature_gate.json"
    atomic_write_text(gate_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": gate, "families": frozen_families, "path": str(gate_path)}))
    return 0


def command_plan_study(args: argparse.Namespace) -> int:
    """Screen both transfer directions for information feasibility (spec §10.3).

    Facts come from the immutable pair manifest and from development outcomes
    only. Everything about the confirmatory reserve is an explicitly labelled
    projection: unseen class-bearing groups, unseen accepted groups, and the
    three one-sided H4 bounds under declared design scenarios. No confirmatory
    outcome is read, and none is imputed.
    """
    cfg = load_config(args.config)
    root = _project_root(args.config)
    pairs, source = _pairs_manifest(cfg, root)
    plan_cfg = cfg.planning

    development_groups = sorted(
        set(pairs.loc[pairs["is_development"].astype(bool), "group_id"].astype(str))
    )
    groups_by_dataset = {
        str(dataset_id): sorted(set(frame["group_id"].astype(str)))
        for dataset_id, frame in pairs.groupby("dataset_id")
    }

    evidence = _development_evidence_table(cfg, root, pairs)
    common_pipelines = [p.id for p in cfg.pipelines if p.in_common_block]
    missing_evidence = [
        f"{dataset}/{pipeline}"
        for dataset in groups_by_dataset
        for pipeline in common_pipelines
        if (dataset, pipeline) not in evidence
    ]
    if missing_evidence:
        raise CommandError(
            "development outcomes are absent for "
            f"{sorted(missing_evidence)}; run `warpaudit register` and `warpaudit labels` "
            "on the development split before planning the study"
        )

    directions: list[dict[str, object]] = []
    for spec in cfg.direction_priority:
        if "->" not in spec:
            raise CommandError(f"invalid direction {spec!r}; expected SOURCE->TARGET", EXIT_USAGE)
        source_dataset, target_dataset = (part.strip() for part in spec.split("->", 1))
        unknown = {source_dataset, target_dataset} - set(groups_by_dataset)
        if unknown:
            raise CommandError(f"direction {spec!r} names unknown dataset(s) {sorted(unknown)}")
        directions.append(
            _assess_direction(
                cfg,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                groups_by_dataset=groups_by_dataset,
                development_groups=development_groups,
                evidence=evidence,
                common_pipelines=common_pipelines,
            )
        )

    feasible = [d for d in directions if d["feasible"]]
    recommendation = feasible[0]["direction"] if feasible else None
    manifest = {
        "generated": _utc_now(),
        "config_hash": cfg.hash,
        "planning_parameters": _to_jsonable(asdict(plan_cfg)),
        "pair_manifest": str(source),
        "development_evidence": [_to_jsonable(e.to_dict()) for e in evidence.values()],
        "directions": _to_jsonable(directions),
        "recommended_direction": recommendation,
        "recommendation_basis": (
            "highest-priority direction meeting every screening minimum"
            if recommendation
            else "no direction met the screening minima; see blocking criteria"
        ),
    }
    manifest_path = cfg.paths.resolve(root)["manifests"] / "m2_feasibility.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    report = cfg.paths.resolve(root)["reports"] / "M2_FEASIBILITY.md"
    atomic_write_text(report, _render_m2_report(cfg, manifest, directions, evidence, source))

    jobs_path = _write_proposed_jobs(cfg, root, pairs, directions, recommendation)
    print(
        json.dumps(
            {
                "recommended_direction": recommendation,
                "report": str(report),
                "manifest": str(manifest_path),
                "proposed_jobs": str(jobs_path),
            }
        )
    )
    return 0


def _to_jsonable(value: object) -> object:
    """Plain JSON types, so a manifest never carries numpy scalars."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _development_evidence_table(
    cfg: Config, root: Path, pairs: pd.DataFrame
) -> dict[tuple[str, str], DevelopmentEvidence]:
    """Observed development outcomes per (dataset, pipeline), from current rows only."""
    cache_root = cfg.paths.resolve(root)["cache_root"]
    registrations = ShardedTable(cache_root, "registrations").load()
    if registrations.empty:
        raise CommandError("registration cache is empty; nothing to plan from")
    registrations = _current_registration_rows(registrations, pairs, cfg, root)
    registrations = registrations[
        registrations["is_development"].astype(bool)
        & (registrations["direction"].astype(str) == "canonical")
        & (registrations["condition"].astype(str) == "clean")
    ].drop_duplicates("job_id", keep="last")
    if registrations.empty:
        raise CommandError("no current clean canonical development registrations")

    labels = ShardedTable(cache_root, "labels").load()
    if labels.empty:
        raise CommandError("label cache is empty; run `warpaudit labels --split development`")
    labels = labels.drop_duplicates("job_id", keep="last")

    joined = registrations.merge(
        labels[["job_id", "operational_failure", "silent_failure", "tre_defined"]],
        on="job_id",
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise CommandError("no development registration has a scored label")

    out: dict[tuple[str, str], DevelopmentEvidence] = {}
    for (dataset_id, pipeline_id), frame in joined.groupby(["dataset_id", "pipeline_id"]):
        out[(str(dataset_id), str(pipeline_id))] = development_evidence(
            frame, dataset_id=str(dataset_id), pipeline_id=str(pipeline_id)
        )
    return out


def _assess_direction(
    cfg: Config,
    *,
    source_dataset: str,
    target_dataset: str,
    groups_by_dataset: dict[str, list[str]],
    development_groups: list[str],
    evidence: dict[tuple[str, str], DevelopmentEvidence],
    common_pipelines: list[str],
) -> dict[str, object]:
    """Allocate one direction and project whether its folds can carry the claim."""
    plan_cfg = cfg.planning
    blocking: list[str] = []
    try:
        plan = make_direction_plan(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            source_groups=groups_by_dataset[source_dataset],
            target_groups=groups_by_dataset[target_dataset],
            development_groups=development_groups,
            preferred_folds=cfg.splits.n_outer_folds,
            fallback_folds=cfg.splits.min_outer_folds,
            low_information_threshold=cfg.splits.low_information_group_threshold,
            seed=cfg.splits.seed,
        )
    except ValueError as exc:
        return {
            "direction": f"{source_dataset}->{target_dataset}",
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "allocated": False,
            "feasible": False,
            "blocking": [f"allocation failed: {exc}"],
            "pipelines": [],
        }

    # The scarcest fold governs feasibility: a claim is only as strong as its
    # weakest evaluated fold, so projections use the smallest test block.
    test_sizes = [len(fold.target_test) for fold in plan.folds]
    n_test_groups = min(test_sizes)
    fold_notes = sorted({note for fold in plan.folds for note in fold.notes})
    limitations: list[str] = []
    if n_test_groups < cfg.splits.low_information_group_threshold:
        blocking.append(
            f"smallest target test block has {n_test_groups} group(s), below the "
            f"{cfg.splits.low_information_group_threshold}-group low-information screen"
        )
    # The split-conformal quantile is an optional extension (§9.3: "not necessary
    # for the core paper", "omit this extension if the group count cannot support
    # an informative analysis"). Too few calibration groups removes that
    # extension; it does not block the primary joint claim, and treating it as a
    # blocker would stop a feasible study for an endpoint it does not report.
    conformal_available = (
        plan.n_calibration_groups >= cfg.splits.conformal_min_calibration_groups
    )
    if not conformal_available:
        limitations.append(
            f"{plan.n_calibration_groups} calibration group(s) per side is below the "
            f"{cfg.splits.conformal_min_calibration_groups} needed for a finite "
            "split-conformal quantile at the configured alpha, so the optional "
            "conformal extension is unavailable and is not reported"
        )

    accepted = project_accepted_groups(
        n_test_groups=n_test_groups,
        n_calibration_groups=plan.n_calibration_groups,
        nominal_acceptance=cfg.policy.nominal_acceptance,
        minimum=plan_cfg.min_accepted_groups,
        n_simulations=plan_cfg.n_simulations,
        seed=plan_cfg.seed,
    )
    if accepted.probability_at_least_minimum < plan_cfg.feasibility_probability:
        blocking.append(
            f"probability of at least {plan_cfg.min_accepted_groups} accepted test groups is "
            f"{accepted.probability_at_least_minimum:.3f}, below the required "
            f"{plan_cfg.feasibility_probability:.2f}"
        )

    pipelines: list[dict[str, object]] = []
    for pipeline_id in common_pipelines:
        target_evidence = evidence[(target_dataset, pipeline_id)]
        support = project_class_support(
            target_evidence,
            n_test_groups=n_test_groups,
            minimum=plan_cfg.min_class_bearing_groups,
            n_simulations=plan_cfg.n_simulations,
            seed=plan_cfg.seed,
        )
        if support.probability_both_at_least_minimum < plan_cfg.feasibility_probability:
            blocking.append(
                f"{pipeline_id}: probability of at least "
                f"{plan_cfg.min_class_bearing_groups} groups bearing each class is "
                f"{support.probability_both_at_least_minimum:.3f}, below the required "
                f"{plan_cfg.feasibility_probability:.2f}"
            )
        scenarios = [
            simulate_iut_scenario(
                name,
                assumed_target_auroc=auroc,
                assumed_gap_auc=gap,
                assumed_delta_policy=delta,
                failure_bearing_groups=support.failure_bearing_mean,
                success_bearing_groups=support.success_bearing_mean,
                accepted_groups=accepted.accepted_groups_mean,
                auroc_floor=cfg.inference.auroc_floor,
                gap_margin=cfg.inference.gap_noninferiority_margin,
                delta_policy_margin=cfg.inference.delta_policy_margin,
                alpha=cfg.inference.alpha_one_sided,
                n_simulations=plan_cfg.n_simulations,
                seed=plan_cfg.seed,
                paired_correlation=plan_cfg.paired_correlation,
                accepted_failure_risk=plan_cfg.accepted_failure_risk,
            )
            for name, (auroc, gap, delta) in sorted(plan_cfg.scenarios.items())
        ]
        pipelines.append(
            {
                "pipeline_id": pipeline_id,
                "target_development_evidence": target_evidence.to_dict(),
                "source_development_evidence": evidence[
                    (source_dataset, pipeline_id)
                ].to_dict(),
                "class_support": support.to_dict(),
                "scenarios": [s.to_dict() for s in scenarios],
            }
        )

    return {
        "direction": plan.direction,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "allocated": True,
        "n_folds": plan.n_folds,
        "n_train_groups": plan.n_train_groups,
        "n_calibration_groups": plan.n_calibration_groups,
        "target_test_groups_per_fold": test_sizes,
        "smallest_target_test_groups": n_test_groups,
        "source_test_groups_per_fold": [len(fold.source_test) for fold in plan.folds],
        "fold_notes": fold_notes,
        "fold_hashes": [fold.hash for fold in plan.folds],
        "accepted_group_projection": accepted.to_dict(),
        "pipelines": pipelines,
        "blocking": blocking,
        "limitations": limitations,
        "conformal_extension_available": conformal_available,
        "feasible": not blocking,
    }


def _render_m2_report(
    cfg: Config,
    manifest: dict[str, object],
    directions: list[dict[str, object]],
    evidence: dict[tuple[str, str], DevelopmentEvidence],
    source: Path,
) -> str:
    plan_cfg = cfg.planning
    lines = [
        "# M2 information feasibility and direction decision",
        "",
        f"Generated: {manifest['generated']}",
        f"Config hash: `{cfg.hash}`",
        f"Pair manifest: `{source}`",
        "",
        "Development outcomes are observations. Everything stated about the confirmatory",
        "reserve is a projection from development evidence under declared assumptions, and",
        "the joint-bound numbers are design scenarios, not effect estimates. None of this",
        "is confirmatory inference; that requires the group-refit bootstrap.",
        "",
        "## Observed development evidence",
        "",
        "| Dataset | Pipeline | Groups | Pairs | Failures | Failure-bearing groups |"
        " Success-bearing groups | Group-weighted prevalence |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(evidence):
        e = evidence[key]
        lines.append(
            f"| {e.dataset_id} | {e.pipeline_id} | {e.n_groups} | {e.n_pairs} | "
            f"{e.n_failures} | {e.failure_bearing_groups} | {e.success_bearing_groups} | "
            f"{e.group_weighted_failure_prevalence:.4f} |"
        )
    lines += [
        "",
        "## Screening rule",
        "",
        f"A direction is feasible when the smallest evaluated fold keeps at least "
        f"{plan_cfg.min_class_bearing_groups} group(s) bearing each class and at least "
        f"{plan_cfg.min_accepted_groups} accepted group(s) with probability "
        f"{plan_cfg.feasibility_probability:.2f}, and when its allocation clears the "
        "low-information and split-conformal group requirements. This screens information, "
        "it does not guarantee power.",
        "",
    ]
    for d in directions:
        lines += [f"## Direction `{d['direction']}`", ""]
        if not d["allocated"]:
            lines += ["Allocation failed:", ""]
            lines += [f"- {reason}" for reason in d["blocking"]] + [""]
            continue
        lines += [
            f"- Outer folds: {d['n_folds']}",
            f"- Matched budget per fold: {d['n_train_groups']} training and "
            f"{d['n_calibration_groups']} calibration group(s) per side",
            f"- Target test groups per fold: {d['target_test_groups_per_fold']}",
            f"- Independent source test groups per fold: {d['source_test_groups_per_fold']}",
            "",
        ]
        accepted = d["accepted_group_projection"]
        lines += [
            "### Accepted-group projection",
            "",
            f"At nominal acceptance {accepted['nominal_acceptance']:.2f} with "
            f"{accepted['n_calibration_groups']} calibration group(s), realised coverage is "
            f"{accepted['realised_coverage_mean']:.3f} "
            f"(90% interval {accepted['realised_coverage_p05']:.3f}-"
            f"{accepted['realised_coverage_p95']:.3f}; threshold quantile SD "
            f"{accepted['threshold_quantile_sd']:.3f}). Accepted test groups: "
            f"{accepted['accepted_groups_mean']:.2f} "
            f"({accepted['accepted_groups_p05']:.0f}-{accepted['accepted_groups_p95']:.0f}); "
            f"P(>= {plan_cfg.min_accepted_groups}) = "
            f"{accepted['probability_at_least_minimum']:.3f}.",
            "",
            "### Class support and joint-bound scenarios",
            "",
            "| Pipeline | Failure-bearing groups | Success-bearing groups |"
            " P(both >= min) | Scenario | P(AUROC floor) | P(gap) | P(policy) | P(joint) |",
            "|---|---|---|---:|---|---:|---:|---:|---:|",
        ]
        for p in d["pipelines"]:
            support = p["class_support"]
            for scenario in p["scenarios"]:
                lines.append(
                    f"| {p['pipeline_id']} | "
                    f"{support['failure_bearing_mean']:.2f} "
                    f"({support['failure_bearing_p05']:.0f}-"
                    f"{support['failure_bearing_p95']:.0f}) | "
                    f"{support['success_bearing_mean']:.2f} "
                    f"({support['success_bearing_p05']:.0f}-"
                    f"{support['success_bearing_p95']:.0f}) | "
                    f"{support['probability_both_at_least_minimum']:.3f} | "
                    f"{scenario['name']} | "
                    f"{scenario['probability_target_auc_passes']:.3f} | "
                    f"{scenario['probability_gap_passes']:.3f} | "
                    f"{scenario['probability_delta_policy_passes']:.3f} | "
                    f"{scenario['probability_joint_passes']:.3f} |"
                )
        lines.append("")
        if d["fold_notes"]:
            lines += ["Allocation notes:", ""]
            lines += [f"- {note}" for note in d["fold_notes"]] + [""]
        if d["limitations"]:
            lines += ["Recorded limitations (not blocking the primary claim):", ""]
            lines += [f"- {note}" for note in d["limitations"]] + [""]
        if d["blocking"]:
            lines += ["Blocking criteria:", ""]
            lines += [f"- {reason}" for reason in d["blocking"]] + [""]
        else:
            lines += ["No blocking criterion applies to this direction.", ""]

    recommendation = manifest["recommended_direction"]
    lines += ["## Recommendation", ""]
    if recommendation:
        lines += [
            f"Recommended primary direction: **`{recommendation}`** -- the highest-priority "
            "direction meeting every screening minimum.",
            "",
            "Set `primary_direction` in the configuration and freeze it at G1 before any "
            "confirmatory outcome is read. The scenario probabilities above are design "
            "assumptions and must not be reported as evidence about the reserve.",
            "",
        ]
    else:
        lines += [
            "**No direction met the screening minima.** The blocking criteria above are the "
            "decision: the pilot allocation does not carry the joint claim as configured. The "
            "admissible responses are to widen the group supply, weaken the claim to the "
            "components that remain feasible, or reduce the fold count -- not to proceed and "
            "report an underpowered joint bound.",
            "",
        ]
    lines += [
        "Priority order: " + ", ".join(f"`{d}`" for d in cfg.direction_priority) + ".",
        "",
    ]
    return "\n".join(lines) + "\n"


def _write_proposed_jobs(
    cfg: Config,
    root: Path,
    pairs: pd.DataFrame,
    directions: list[dict[str, object]],
    recommendation: str | None,
) -> Path:
    """Enumerate proposed confirmatory jobs, costed from measured runtimes only."""
    cache_root = cfg.paths.resolve(root)["cache_root"]
    measured = ShardedTable(cache_root, "registrations").load()
    seconds: dict[str, float] = {}
    if not measured.empty and {"pipeline_id", "runtime_s", "condition"} <= set(measured):
        current = _current_registration_rows(measured, pairs, cfg, root)
        clean = current[current["condition"].astype(str) == "clean"]
        runtimes = pd.to_numeric(clean.get("runtime_s"), errors="coerce")
        finite = clean[np.isfinite(runtimes)]
        if not finite.empty:
            seconds = (
                pd.to_numeric(finite["runtime_s"], errors="coerce")
                .groupby(finite["pipeline_id"].astype(str))
                .median()
                .to_dict()
            )

    # With no recommended direction, no confirmatory job is in scope. Saying so
    # is the point of the screen: silently marking everything runnable would
    # turn a blocked decision into an implicit go-ahead.
    in_scope_datasets: set[str] = set()
    for d in directions:
        if d["direction"] == recommendation:
            in_scope_datasets = {str(d["source_dataset"]), str(d["target_dataset"])}
    confirmatory_status = "proposed" if recommendation else "blocked_by_feasibility"

    jobs: list[dict[str, object]] = []
    for _, row in pairs.iterrows():
        for pipeline in cfg.pipelines:
            for direction in ("canonical", "reverse"):
                is_development = bool(row["is_development"])
                jobs.append(
                    {
                        "dataset_id": row["dataset_id"],
                        "pair_id": row["pair_id"],
                        "group_id": row["group_id"],
                        "is_development": is_development,
                        "pipeline_id": pipeline.id,
                        "direction": direction,
                        "condition": "clean",
                        "severity": 0,
                        "timing_basis": "measured_median" if seconds else "unmeasured",
                        "seconds_per_job": seconds.get(pipeline.id, float("nan")),
                        "contingency": 1.5,
                        "recommended_direction": recommendation or "",
                        "in_recommended_scope": is_development
                        or str(row["dataset_id"]) in in_scope_datasets,
                        "status": "proposed" if is_development else confirmatory_status,
                    }
                )
    jobs_path = cfg.paths.resolve(root)["manifests"] / "proposed_jobs.csv"
    atomic_write_text(jobs_path, pd.DataFrame(jobs).to_csv(index=False))
    return jobs_path


def command_freeze(args: argparse.Namespace) -> int:
    """Write the G1 freeze record that gates confirmatory outcome access.

    Everything recorded here is read back from artefacts the earlier stages
    produced -- the pair manifest, the frozen feature gate, and the M2
    feasibility screen -- rather than restated by hand, so the freeze cannot
    quietly disagree with the evidence that justified it.
    """
    cfg = load_config(args.config)
    root = _project_root(args.config)
    paths = cfg.paths.resolve(root)
    manifests = paths["manifests"]

    gate_path = manifests / "feature_gate.json"
    if not gate_path.is_file():
        raise CommandError(
            "the development feature gate is absent; run `warpaudit diagnose-development` "
            "before freezing, so the frozen families come from evidence"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    frozen_families = tuple(gate.get("frozen_feature_families", ()))
    if not frozen_families:
        raise CommandError(f"{gate_path} names no frozen feature families")

    feasibility_path = manifests / "m2_feasibility.json"
    if not feasibility_path.is_file():
        raise CommandError(
            "the M2 feasibility screen is absent; run `warpaudit plan-study` before freezing"
        )
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    if feasibility.get("config_hash") != cfg.hash:
        raise CommandError(
            f"{feasibility_path} was produced under configuration "
            f"{feasibility.get('config_hash')!r}, not {cfg.hash!r}; rerun `plan-study`"
        )
    # M2 exists to recommend the direction from development evidence, so taking
    # its recommendation is the specified path, not a shortcut. What §3.3 forbids
    # is choosing a direction from completed *test* results, which nothing here
    # can see. An explicit configuration or flag still wins, and whichever source
    # was used is recorded in the freeze.
    direction_source = "argument"
    direction = args.direction
    if not direction:
        direction, direction_source = cfg.primary_direction or "", "configuration"
    if not direction:
        direction = str(feasibility.get("recommended_direction") or "")
        direction_source = "m2_recommendation"
    if not direction:
        raise CommandError(
            "no primary direction to freeze: the M2 screen recommended none, and "
            "neither `primary_direction` nor --direction supplies one. The screen's "
            "blocking criteria are the decision to act on, not to override silently.",
            EXIT_USAGE,
        )
    if "->" not in direction:
        raise CommandError(f"invalid direction {direction!r}; expected SOURCE->TARGET", EXIT_USAGE)
    source_dataset, target_dataset = (part.strip() for part in direction.split("->", 1))

    assessment = next(
        (d for d in feasibility.get("directions", ()) if d.get("direction") == direction), None
    )
    if assessment is None:
        raise CommandError(f"the feasibility screen contains no assessment for {direction!r}")
    if not assessment.get("allocated"):
        raise CommandError(
            f"{direction} could not be allocated: {assessment.get('blocking')}"
        )
    blocking = list(assessment.get("blocking", ()))
    if blocking and not args.acknowledge_infeasible:
        raise CommandError(
            f"{direction} does not meet the M2 screening minima:\n  - "
            + "\n  - ".join(blocking)
            + "\nFreezing anyway is a deliberate protocol decision, not a default. "
            "Pass --acknowledge-infeasible with --review-note explaining the decision, "
            "and record it in PROTOCOL_DEVIATIONS.md."
        )
    if not args.signed_off_by or not args.review_note:
        raise CommandError(
            "--signed-off-by and --review-note are required: §14 makes G1 a reviewed "
            "gate, and an unattributed freeze is not a review",
            EXIT_USAGE,
        )

    record = FreezeRecord(
        generated_at=_utc_now(),
        config_hash=cfg.hash,
        git_commit=_git_commit(root),
        registration_code_identity=_registration_code_identity(root),
        feature_code_identity=_feature_code_identity(root),
        primary_direction=direction,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        n_folds=int(assessment["n_folds"]),
        n_train_groups=int(assessment["n_train_groups"]),
        n_calibration_groups=int(assessment["n_calibration_groups"]),
        fold_hashes=tuple(assessment["fold_hashes"]),
        frozen_feature_families=frozen_families,
        common_block_pipelines=tuple(p.id for p in cfg.pipelines if p.in_common_block),
        learner=cfg.learners.primary,
        nominal_acceptance=cfg.policy.nominal_acceptance,
        auroc_floor=cfg.inference.auroc_floor,
        gap_noninferiority_margin=cfg.inference.gap_noninferiority_margin,
        delta_policy_margin=cfg.inference.delta_policy_margin,
        alpha_one_sided=cfg.inference.alpha_one_sided,
        signed_off_by=args.signed_off_by,
        review_note=args.review_note,
        direction_source=direction_source,
        feasibility={
            "blocking": blocking,
            "limitations": list(assessment.get("limitations", ())),
            "conformal_extension_available": assessment.get(
                "conformal_extension_available", False
            ),
            "smallest_target_test_groups": assessment.get("smallest_target_test_groups"),
            "accepted_group_projection": assessment.get("accepted_group_projection"),
        },
        acknowledged_infeasible=bool(blocking),
    )
    path = manifests / FREEZE_FILENAME
    atomic_write_text(path, json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "frozen": str(path),
                "direction": direction,
                "freeze_hash": record.hash,
                "frozen_feature_families": list(frozen_families),
                "acknowledged_infeasible": record.acknowledged_infeasible,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _feature_matrix(
    features: pd.DataFrame, families: Sequence[str], job_ids: Sequence[str]
) -> pd.DataFrame:
    """Wide feature table for the frozen families, indexed by job.

    A feature that is unavailable for a row becomes ``nan`` here, which the
    source-fitted imputer and its missingness indicators then handle (§7.3).
    Availability is never silently read as a value of zero.
    """
    present = set(features["family"].astype(str))
    missing = sorted(set(families) - present)
    if missing:
        # A frozen composite silently missing a family would evaluate a
        # different predictor than the one G1 fixed.
        raise CommandError(
            f"the feature cache lacks the frozen families {missing}; the composite "
            "cannot be assembled as frozen. Run `warpaudit features --split "
            f"confirmatory --families {' '.join(missing)}`"
        )
    selected = features[features["family"].astype(str).isin(set(families))].copy()
    if selected.empty:
        raise CommandError(f"the feature cache has no rows for families {sorted(families)}")
    selected["column"] = (
        selected["family"].astype(str) + ":" + selected["feature_name"].astype(str)
    )
    selected["numeric"] = pd.to_numeric(selected["value"], errors="coerce")
    selected.loc[~selected["available"].astype(bool), "numeric"] = np.nan
    wide = selected.pivot_table(
        index="job_id", columns="column", values="numeric", aggfunc="last", dropna=False
    )
    return wide.reindex(list(job_ids))


def _assemble_cases(
    cfg: Config, root: Path, families: Sequence[str]
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """One scored row per cached canonical clean registration.

    §8.2 requires a single canonical direction for primary endpoints, so
    reverse runs stay out of the evaluation table: they are inputs to the cycle
    signal, not extra independent cases.
    """
    pair_rows, _ = _pairs_manifest(cfg, root)
    cache_root = cfg.paths.resolve(root)["cache_root"]
    registrations = ShardedTable(cache_root, "registrations").load()
    if registrations.empty:
        raise CommandError("registration cache is empty")
    registrations = _current_registration_rows(registrations, pair_rows, cfg, root)
    registrations = registrations[
        (registrations["direction"].astype(str) == "canonical")
        & (registrations["condition"].astype(str) == "clean")
    ].drop_duplicates("job_id", keep="last")
    if registrations.empty:
        raise CommandError("no current clean canonical registrations to evaluate")

    labels = ShardedTable(cache_root, "labels").load()
    if labels.empty:
        raise CommandError("label cache is empty")
    labels = labels.drop_duplicates("job_id", keep="last")

    features = ShardedTable(cache_root, "features", key_column="feature_id").load()
    if features.empty:
        raise CommandError("feature cache is empty")
    features = features[
        (features["code_hash"].astype(str) == _feature_code_identity(root))
        & (features["config_hash"].astype(str) == cfg.hash)
    ]
    if features.empty:
        raise CommandError(
            "no feature rows match the current feature code and configuration; "
            "rerun `warpaudit features`"
        )

    cases = registrations.merge(
        labels[
            [
                "job_id",
                "operational_failure",
                "silent_failure",
                "eligible_for_acceptance",
                "bounded_loss",
                "tre_norm",
                "tre_defined",
            ]
        ],
        on="job_id",
        how="inner",
        validate="one_to_one",
    )
    if cases.empty:
        raise CommandError("no registration has both a current cache row and a scored label")
    cases["explicit_failure"] = ~cases["eligible_for_acceptance"].astype(bool)
    matrix = _feature_matrix(features, families, cases["job_id"].astype(str).tolist())
    feature_names = tuple(str(c) for c in matrix.columns)
    if not feature_names:
        raise CommandError("the frozen families produced no feature columns")
    cases = pd.concat(
        (cases.reset_index(drop=True), matrix.reset_index(drop=True)), axis=1
    )
    keep = [
        "job_id",
        "dataset_id",
        "pair_id",
        "group_id",
        "pipeline_id",
        "operational_failure",
        "silent_failure",
        "explicit_failure",
        "bounded_loss",
        "tre_norm",
        *feature_names,
    ]
    cases = cases.loc[:, keep].sort_values(["dataset_id", "pair_id", "pipeline_id"], kind="stable")
    cases["operational_failure"] = cases["operational_failure"].astype(float)
    return cases.reset_index(drop=True), feature_names


def _fold_role_groups(split: TransferSplit) -> dict[str, list[str]]:
    return {role: list(getattr(split, role)) for role in EXPERIMENT_ROLES}


def command_evaluate(args: argparse.Namespace) -> int:
    """Fit, freeze, apply, and test the joint claim on the frozen direction.

    Every design choice this reads -- direction, folds, budget, families,
    learner, policy, margins -- comes from the G1 record rather than from the
    command line, so a run cannot quietly evaluate a protocol other than the
    frozen one.
    """
    cfg = load_config(args.config)
    root = _project_root(args.config)
    paths = cfg.paths.resolve(root)
    try:
        freeze = require_confirmatory_access(
            paths["manifests"], config_hash=cfg.hash, what="confirmatory evaluation"
        )
    except FreezeError as exc:
        raise CommandError(str(exc)) from exc

    families = list(freeze.frozen_feature_families)
    cases, feature_names = _assemble_cases(cfg, root, families)
    pair_rows, _ = _pairs_manifest(cfg, root)
    development_groups = sorted(
        set(pair_rows.loc[pair_rows["is_development"].astype(bool), "group_id"].astype(str))
    )
    groups_by_dataset = {
        str(dataset_id): sorted(set(frame["group_id"].astype(str)))
        for dataset_id, frame in pair_rows.groupby("dataset_id")
    }
    plan = make_direction_plan(
        source_dataset=freeze.source_dataset,
        target_dataset=freeze.target_dataset,
        source_groups=groups_by_dataset[freeze.source_dataset],
        target_groups=groups_by_dataset[freeze.target_dataset],
        development_groups=development_groups,
        preferred_folds=cfg.splits.n_outer_folds,
        fallback_folds=cfg.splits.min_outer_folds,
        low_information_threshold=cfg.splits.low_information_group_threshold,
        seed=cfg.splits.seed,
    )
    if tuple(fold.hash for fold in plan.folds) != tuple(freeze.fold_hashes):
        raise CommandError(
            "the reconstructed fold allocation does not match the frozen one. The "
            "split definition changed after G1; restore it or re-freeze deliberately."
        )

    experiment_id = short_hash(
        {
            "freeze": freeze.hash,
            "families": families,
            "learner": freeze.learner,
            "code": _code_identity(root),
        }
    )
    predictions = ShardedTable(paths["cache_root"], "predictions", key_column="prediction_id")
    outcomes: list[FoldOutcome] = []
    prediction_rows: list[dict[str, object]] = []
    for split in plan.folds:
        enforce(
            check_split(split),
            f"{freeze.primary_direction} fold {split.fold}",
        )
        outcome = evaluate_fold(
            cases,
            _fold_role_groups(split),
            feature_names,
            fold=split.fold,
            protocol=split.protocol,
            experiment_id=experiment_id,
            nominal_acceptance=freeze.nominal_acceptance,
            tie_rule=cfg.policy.tie_rule,
            calibration_method=cfg.policy.calibration_method,
            C_values=cfg.learners.logistic_C,
            seed=cfg.splits.seed,
            requested_coverages=cfg.policy.requested_coverages,
            secondary_learner=cfg.learners.secondary,
            lightgbm_leaves=cfg.learners.lightgbm_leaves,
            lightgbm_min_child_samples=cfg.learners.lightgbm_min_child_samples,
        )
        outcomes.append(outcome)
        for row in outcome.prediction_rows:
            row["prediction_id"] = short_hash(
                {
                    "job": row["job_id"],
                    "arm": row["arm"],
                    "fold": row["fold"],
                    "experiment": experiment_id,
                }
            )
            prediction_rows.append(row)
        print(
            f"[fold {split.fold}] "
            + " ".join(
                f"{k}={outcome.estimands.get(k, float('nan')):.4f}"
                for k in (
                    "target_auroc_transferred",
                    "target_auroc_reference",
                    "gap_auc",
                    "delta_policy",
                )
            ),
            flush=True,
        )

    usable = [o for o in outcomes if o.estimands]
    if not usable:
        raise CommandError(
            "no fold produced an estimate; every fold lacked rows in at least one role: "
            + "; ".join(note for o in outcomes for note in o.notes)
        )
    predictions.append(prediction_rows, shard_hint=experiment_id)

    summary = _aggregate_folds(usable)
    bootstrap = _confirmatory_bootstrap(
        cfg, cases, feature_names, plan, freeze, experiment_id=experiment_id,
        n_resamples=args.refit_bootstrap or cfg.inference.refit_bootstrap,
    )
    claim = decide(
        bootstrap["target_auroc_transferred"],
        bootstrap["gap_auc"],
        bootstrap["delta_policy"],
        auroc_floor=freeze.auroc_floor,
        gap_margin=freeze.gap_noninferiority_margin,
        policy_margin=freeze.delta_policy_margin,
        alpha_one_sided=freeze.alpha_one_sided,
        multiplicity_rule=cfg.inference.multiplicity_rule,
        realised_coverages={
            "source_on_target": summary["realised_coverage_source_on_target"],
            "reference_on_target": summary["realised_coverage_reference_on_target"],
        },
        accepted_group_counts={
            "source_on_target": int(summary["accepted_groups_source_on_target"]),
            "reference_on_target": int(summary["accepted_groups_reference_on_target"]),
        },
    )

    results = {
        "generated": _utc_now(),
        "experiment_id": experiment_id,
        "freeze_hash": freeze.hash,
        "direction": freeze.primary_direction,
        "protocol": "P3",
        "frozen_feature_families": families,
        "feature_columns": list(feature_names),
        "n_folds": len(usable),
        "fold_estimands": [
            {"fold": o.fold, **_to_jsonable(o.estimands), "budget": dict(o.budget)}
            for o in usable
        ],
        "fold_notes": sorted({note for o in usable for note in o.notes}),
        "aggregate": _to_jsonable(summary),
        "bootstrap": {name: _to_jsonable(r.as_row()) for name, r in bootstrap.items()},
        "monte_carlo_stability": {
            name: _to_jsonable(monte_carlo_stability(result))
            for name, result in bootstrap.items()
        },
        "joint_claim": _to_jsonable(claim.as_rows()),
        "conclusion": claim.summary(),
        "policy_comparisons": [
            {
                "fold": o.fold,
                **{name: _to_jsonable(c.as_row()) for name, c in o.comparisons.items()},
            }
            for o in usable
        ],
        "prevalence_decomposition": [
            {
                "fold": o.fold,
                "available": bool(o.prevalence.available),
                "pi_star": float(o.prevalence.pi_star),
                "delta_domain_raw": float(o.prevalence.delta_domain_raw),
                "delta_domain_standardised": float(o.prevalence.delta_domain_standardised),
                "delta_coverage_raw": float(o.prevalence.delta_coverage_raw),
                "delta_coverage_standardised": float(o.prevalence.delta_coverage_standardised),
                "delta_a1": float(o.prevalence.delta_a1),
                "delta_a0": float(o.prevalence.delta_a0),
                "interpretation": o.prevalence.interpretation(),
                "notes": list(o.prevalence.notes),
            }
            for o in usable
            if o.prevalence is not None
        ],
    }
    results_path = paths["manifests"] / "results.json"
    atomic_write_text(results_path, json.dumps(results, indent=2, sort_keys=True) + "\n")
    report_path = _write_results_report(cfg, root, results, freeze)
    print(json.dumps({
        "results": str(results_path),
        "report": str(report_path),
        "predictions": len(prediction_rows),
        "conclusion": claim.summary(),
    }, indent=2, sort_keys=True))
    return 0


def command_freeze_full_study(args: argparse.Namespace) -> int:
    """Freeze the new multi-domain design before any external label access."""
    cfg = load_config(args.config)
    if cfg.tier != "full":
        raise CommandError("freeze-full-study requires tier: full", EXIT_USAGE)
    if not args.signed_off_by or not args.review_note:
        raise CommandError("--signed-off-by and --review-note are required", EXIT_USAGE)
    root = _project_root(args.config)
    pairs, _ = _pairs_manifest(cfg, root)
    present = set(pairs["dataset_id"].astype(str))
    required = set(cfg.full_study.development_datasets) | set(
        cfg.full_study.external_confirmatory_datasets
    ) | set(cfg.full_study.external_descriptive_datasets)
    if missing := sorted(required - present):
        raise CommandError(f"audited pair manifest lacks full-study datasets {missing}")
    fs = cfg.full_study
    unresolved_contracts = []
    for dataset_id in (*fs.external_confirmatory_datasets, *fs.external_descriptive_datasets):
        dataset = cfg.dataset(dataset_id)
        contract = f"{dataset.licence} {dataset.access_note}".lower()
        if "verify" in contract or "unresolved" in contract:
            unresolved_contracts.append(dataset_id)
    if unresolved_contracts:
        raise CommandError(
            "external dataset access/licensing remains unresolved for "
            f"{sorted(unresolved_contracts)}; update the reviewed configuration and rerun "
            "audit-data plus plan-full-study before G2"
        )
    plan_path = cfg.paths.resolve(root)["manifests"] / "full_study_information_plan.json"
    if not plan_path.is_file():
        raise CommandError("run `warpaudit plan-full-study` before freezing")
    information_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if information_plan.get("config_hash") != cfg.hash or information_plan.get(
        "code_identity"
    ) != _code_identity(root):
        raise CommandError("full-study information plan is stale; regenerate it")
    blocked_cells = [cell for cell in information_plan.get("cells", ()) if not cell["passed"]]
    if blocked_cells:
        names = [f"{cell['dataset']}/{cell['pipeline']}" for cell in blocked_cells]
        raise CommandError(f"information gate blocks confirmatory cells {names}")
    development_mask = pairs["dataset_id"].astype(str).isin(fs.development_datasets)
    external_mask = pairs["dataset_id"].astype(str).isin(
        (*fs.external_confirmatory_datasets, *fs.external_descriptive_datasets)
    )
    if not pairs.loc[development_mask, "is_development"].astype(bool).all():
        raise CommandError("a declared development dataset contains locked external rows")
    if pairs.loc[external_mask, "is_development"].astype(bool).any():
        raise CommandError("an external dataset was exposed through the development partition")
    for dataset_id in fs.external_confirmatory_datasets:
        dataset_rows = pairs[pairs["dataset_id"].astype(str) == dataset_id]
        n_groups = dataset_rows["group_id"].astype(str).nunique()
        minimum = 2 * fs.min_class_bearing_groups
        if n_groups < minimum:
            raise CommandError(
                f"{dataset_id}: {n_groups} independent groups cannot possibly supply "
                f"{fs.min_class_bearing_groups} groups in each outcome class; need at least {minimum}"
            )

    # Fail before freezing if the development-side method is not executable.
    families = tuple(dict.fromkeys((*fs.baseline_families, *fs.augmented_families)))
    development_cases, feature_names = _assemble_cases(cfg, root, families)
    development_cases = development_cases[
        development_cases["dataset_id"].astype(str).isin(fs.development_datasets)
    ]
    feature_cache = ShardedTable(
        cfg.paths.resolve(root)["cache_root"], "features", key_column="feature_id"
    ).load()
    for pipeline_id in cfg.common_block:
        pipeline_cases = development_cases[
            development_cases["pipeline_id"].astype(str) == pipeline_id
        ]
        if pipeline_cases["group_id"].astype(str).nunique() < 5:
            raise CommandError(f"{pipeline_id}: fewer than five development groups are scored")
        if pipeline_cases["operational_failure"].nunique() < 2:
            raise CommandError(f"{pipeline_id}: development outcomes contain only one class")
        job_ids = set(pipeline_cases["job_id"].astype(str))
        available = feature_cache[
            feature_cache["job_id"].astype(str).isin(job_ids)
            & feature_cache["available"].astype(bool)
        ]
        for family in ("E1", "E2"):
            family_rows = available[available["family"].astype(str) == family]
            if family_rows.empty or not pd.to_numeric(
                family_rows["value"], errors="coerce"
            ).notna().any():
                raise CommandError(
                    f"{pipeline_id}: no finite development {family} evidence; "
                    "the factorized model cannot be frozen"
                )
    if not feature_names:
        raise CommandError("the full-study feature matrix is empty")
    try:
        evaluate_full_study(
            development_cases,
            feature_names,
            development_datasets=fs.development_datasets,
            external_datasets=(),
            confirmatory_datasets=(),
            pipelines=cfg.common_block,
            baseline_families=fs.baseline_families,
            augmented_families=fs.augmented_families,
            calibration_method=cfg.policy.calibration_method,
            nominal_acceptance=cfg.policy.nominal_acceptance,
            high_confidence_cutoff=fs.high_confidence_cutoff,
            min_class_bearing_groups=fs.min_class_bearing_groups,
            min_brier_improvement=fs.min_brier_improvement,
            bootstrap_resamples=1,
            C_values=cfg.learners.logistic_C,
            seed=cfg.splits.seed,
            workers=getattr(args, "workers", 1),
            worker_threads=getattr(args, "worker_threads", 1),
        )
    except ValueError as exc:
        raise CommandError(f"development model preflight failed: {exc}") from exc
    record = FullStudyFreeze(
        generated_at=_utc_now(),
        config_hash=cfg.hash,
        git_commit=_git_commit(root),
        code_identity=_code_identity(root),
        development_datasets=fs.development_datasets,
        external_confirmatory_datasets=fs.external_confirmatory_datasets,
        external_descriptive_datasets=fs.external_descriptive_datasets,
        pipelines=cfg.common_block,
        baseline_families=fs.baseline_families,
        augmented_families=fs.augmented_families,
        primary_metric=fs.primary_metric,
        min_brier_improvement=fs.min_brier_improvement,
        min_passing_cell_fraction=fs.min_passing_cell_fraction,
        min_class_bearing_groups=fs.min_class_bearing_groups,
        information_plan_hash=short_hash(information_plan),
        bootstrap_resamples=fs.bootstrap_resamples,
        high_confidence_cutoff=fs.high_confidence_cutoff,
        signed_off_by=args.signed_off_by,
        review_note=args.review_note,
    )
    path = cfg.paths.resolve(root)["manifests"] / "g2_full_study_freeze.json"
    atomic_write_text(path, json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"frozen": str(path), "freeze_hash": record.hash}, indent=2))
    return 0


def command_plan_full_study(args: argparse.Namespace) -> int:
    """Project whether external group counts can support both outcome classes."""
    cfg = load_config(args.config)
    if cfg.tier != "full":
        raise CommandError("plan-full-study requires tier: full", EXIT_USAGE)
    root = _project_root(args.config)
    fs = cfg.full_study
    pairs, _ = _pairs_manifest(cfg, root)
    families = tuple(dict.fromkeys((*fs.baseline_families, *fs.augmented_families)))
    cases, _ = _assemble_cases(cfg, root, families)
    development = cases[cases["dataset_id"].astype(str).isin(fs.development_datasets)]
    rng = np.random.default_rng(cfg.splits.seed)
    cells = []
    for pipeline_id in cfg.common_block:
        pipe = development[development["pipeline_id"].astype(str) == pipeline_id]
        categories = []
        for _, group in pipe.groupby("group_id"):
            labels = set(group["operational_failure"].astype(int))
            categories.append(2 if labels == {0, 1} else (1 if labels == {1} else 0))
        counts = np.bincount(categories, minlength=3)
        if not len(categories) or not (counts[0] + counts[2]) or not (counts[1] + counts[2]):
            raise CommandError(
                f"{pipeline_id}: development groups do not contain both outcome classes"
            )
        for dataset_id in fs.external_confirmatory_datasets:
            n_groups = int(
                pairs[pairs["dataset_id"].astype(str) == dataset_id]["group_id"]
                .astype(str)
                .nunique()
            )
            successes = 0
            for _ in range(cfg.planning.n_simulations):
                probabilities = rng.dirichlet(counts + 0.5)
                only_success, only_failure, both = rng.multinomial(n_groups, probabilities)
                successes += int(
                    only_failure + both >= fs.min_class_bearing_groups
                    and only_success + both >= fs.min_class_bearing_groups
                )
            probability = successes / cfg.planning.n_simulations
            cells.append(
                {
                    "dataset": dataset_id,
                    "pipeline": pipeline_id,
                    "external_groups": n_groups,
                    "development_group_categories": {
                        "success_only": int(counts[0]),
                        "failure_only": int(counts[1]),
                        "both": int(counts[2]),
                    },
                    "probability_of_class_support": probability,
                    "required_probability": fs.min_information_probability,
                    "passed": probability >= fs.min_information_probability,
                }
            )
    payload = {
        "generated_at": _utc_now(),
        "config_hash": cfg.hash,
        "code_identity": _code_identity(root),
        "simulations": cfg.planning.n_simulations,
        "cells": cells,
        "all_passed": bool(cells) and all(cell["passed"] for cell in cells),
        "analysis_mode": fs.analysis_mode,
        "note": (
            "Descriptive external study; no confirmatory information gate or primary claim."
            if fs.analysis_mode == "descriptive"
            else "Development-informed class-support projection; not effect-size power."
        ),
    }
    path = cfg.paths.resolve(root)["manifests"] / "full_study_information_plan.json"
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"plan": str(path), "all_passed": payload["all_passed"]}, indent=2))
    return 0


def command_evaluate_full_study(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if cfg.tier != "full":
        raise CommandError("evaluate-full-study requires tier: full", EXIT_USAGE)
    root = _project_root(args.config)
    try:
        freeze = require_full_freeze(
            cfg.paths.resolve(root)["manifests"], config_hash=cfg.hash
        )
    except FreezeError as exc:
        raise CommandError(str(exc)) from exc
    current_code = _code_identity(root)
    if current_code != freeze.code_identity:
        raise CommandError(
            "executable code differs from the G2 freeze; restore the frozen revision or "
            "document a blinded recovery and create a reviewed replacement freeze"
        )
    plan_path = cfg.paths.resolve(root)["manifests"] / "full_study_information_plan.json"
    if not plan_path.is_file():
        raise CommandError("the information plan recorded at G2 is absent")
    information_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if short_hash(information_plan) != freeze.information_plan_hash:
        raise CommandError("the full-study information plan differs from the G2 freeze")
    families = tuple(dict.fromkeys((*freeze.baseline_families, *freeze.augmented_families)))
    cases, feature_names = _assemble_cases(cfg, root, families)
    external = (*freeze.external_confirmatory_datasets, *freeze.external_descriptive_datasets)
    rows = evaluate_full_study(
        cases,
        feature_names,
        development_datasets=freeze.development_datasets,
        external_datasets=external,
        confirmatory_datasets=freeze.external_confirmatory_datasets,
        pipelines=freeze.pipelines,
        baseline_families=freeze.baseline_families,
        augmented_families=freeze.augmented_families,
        calibration_method=cfg.policy.calibration_method,
        nominal_acceptance=cfg.policy.nominal_acceptance,
        high_confidence_cutoff=freeze.high_confidence_cutoff,
        min_class_bearing_groups=freeze.min_class_bearing_groups,
        min_brier_improvement=freeze.min_brier_improvement,
        bootstrap_resamples=freeze.bootstrap_resamples,
        C_values=cfg.learners.logistic_C,
        seed=cfg.splits.seed,
            workers=getattr(args, "workers", 1),
            worker_threads=getattr(args, "worker_threads", 1),
    )
    confirmatory = [row for row in rows if row.get("confirmatory") and row.get("eligible")]
    passing = sum(bool(row.get("primary_passed")) for row in confirmatory)
    passing_fraction = passing / len(confirmatory) if confirmatory else float("nan")
    primary_passed = bool(
        confirmatory
        and len(confirmatory)
        == len(freeze.external_confirmatory_datasets) * len(freeze.pipelines)
        and passing_fraction >= freeze.min_passing_cell_fraction
    )
    payload = {
        "generated_at": _utc_now(),
        "freeze_hash": freeze.hash,
        "primary_metric": freeze.primary_metric,
        "cells": rows,
        "confirmatory_cells": sum(bool(row.get("confirmatory")) for row in rows),
        "eligible_confirmatory_cells": len(confirmatory),
        "passed_confirmatory_cells": passing,
        "passing_confirmatory_cell_fraction": passing_fraction,
        "required_passing_cell_fraction": freeze.min_passing_cell_fraction,
        "primary_passed": primary_passed if freeze.external_confirmatory_datasets else None,
        "analysis_mode": cfg.full_study.analysis_mode,
        "conclusion": (
            "descriptive external study; no confirmatory primary claim was tested"
            if not freeze.external_confirmatory_datasets else
            "factorized model met the frozen multi-cell criterion"
            if primary_passed
            else "factorized model did not meet the frozen multi-cell criterion"
        ),
    }
    output = cfg.paths.resolve(root)["manifests"] / "full_study_results.json"
    atomic_write_text(output, json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
        for section in ("baseline", "augmented", "contrast"):
            flat.update({f"{section}_{key}": value for key, value in row.get(section, {}).items()})
        flat_rows.append(flat)
    table_path = cfg.paths.resolve(root)["tables"] / "full_study_cells.csv"
    atomic_write_text(table_path, pd.DataFrame(flat_rows).to_csv(index=False))
    report_path = cfg.paths.resolve(root)["reports"] / "FULL_STUDY_RESULTS.md"
    report_lines = [
        "# Full-study results",
        "",
        f"Freeze hash: `{freeze.hash}`",
        "",
        f"**{payload['conclusion']}.**",
        "",
        f"Eligible confirmatory cells: {len(confirmatory)}; passed: {passing}; "
        f"fraction: {passing_fraction:.3f}; required: {freeze.min_passing_cell_fraction:.3f}.",
        "",
        "| Dataset | Pipeline | Eligible | Brier improvement | 95% CI | AUROC improvement | Passed |",
        "|---|---|:--:|---:|---|---:|:--:|",
    ]
    for row in rows:
        contrast = row.get("contrast", {})
        interval = contrast.get("brier_improvement_95ci", [float("nan"), float("nan")])
        report_lines.append(
            f"| {row['dataset']} | {row['pipeline']} | {'yes' if row.get('eligible') else 'no'} | "
            f"{contrast.get('brier_improvement', float('nan')):.4f} | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
            f"{contrast.get('auroc_improvement', float('nan')):.4f} | "
            f"{'yes' if row.get('primary_passed') else 'no'} |"
        )
    atomic_write_text(report_path, "\n".join(report_lines) + "\n")
    print(
        json.dumps(
            {
                "results": str(output),
                "report": str(report_path),
                "confirmatory_cells": payload["confirmatory_cells"],
                "eligible_confirmatory_cells": payload["eligible_confirmatory_cells"],
                "passed_confirmatory_cells": payload["passed_confirmatory_cells"],
                "primary_passed": payload["primary_passed"],
            },
            indent=2,
        )
    )
    return 0


def _aggregate_folds(outcomes: Sequence[FoldOutcome]) -> dict[str, float]:
    """Average fold estimands, keeping an undefined cell undefined (§10.1)."""
    keys = sorted({key for o in outcomes for key in o.estimands})
    summary: dict[str, float] = {}
    for key in keys:
        values = [o.estimands.get(key, float("nan")) for o in outcomes]
        finite = [v for v in values if np.isfinite(v)]
        # A missing or undefined fold is not silently dropped: the aggregate is
        # undefined unless every fold contributed.
        summary[key] = float(np.mean(finite)) if len(finite) == len(values) else float("nan")
        summary[f"{key}__folds_defined"] = float(len(finite))
    return summary

def _confirmatory_bootstrap(
    cfg: Config,
    cases: pd.DataFrame,
    feature_names: Sequence[str],
    plan: DirectionPlan,
    freeze: FreezeRecord,
    *,
    experiment_id: str,
    n_resamples: int,
) -> dict[str, BootstrapResult]:
    """The complete refit bootstrap behind the primary claim (§10.2).

    Every resample rebuilds the whole fold procedure -- source fitting and
    calibration, target-reference fitting and calibration, threshold selection,
    and evaluation -- so the interval carries fitting uncertainty rather than
    conditioning it away. Roles are resampled within themselves, which keeps
    fold membership intact and makes it impossible for a repeated copy of a
    group to occupy two roles of the same fold.

    Folds are pooled by resampling the union of each role across folds, so a
    resample perturbs the whole evaluated design rather than one fold at a time.
    """
    # Roles are keyed by fold, not pooled across folds. §10.2 requires role
    # separation *within* an evaluated fold; a group that tests in one fold
    # legitimately trains in another, so pooling would both violate the
    # disjointness the resampler checks and destroy fold membership.
    role_groups: dict[str, list[str]] = {
        f"{role}@{split.fold}": list(getattr(split, role))
        for split in plan.folds
        for role in EXPERIMENT_ROLES
    }

    def procedure(resampled: Mapping[str, Sequence[str]]) -> Mapping[str, float]:
        per_fold: list[dict[str, float]] = []
        for split in plan.folds:
            roles = {
                role: list(resampled[f"{role}@{split.fold}"]) for role in EXPERIMENT_ROLES
            }
            if any(not groups for groups in roles.values()):
                continue
            outcome = evaluate_fold(
                cases,
                roles,
                feature_names,
                fold=split.fold,
                protocol=split.protocol,
                experiment_id=experiment_id,
                nominal_acceptance=freeze.nominal_acceptance,
                tie_rule=cfg.policy.tie_rule,
                calibration_method=cfg.policy.calibration_method,
                C_values=cfg.learners.logistic_C,
                seed=cfg.splits.seed,
                with_predictions=False,
                with_controls=False,
            )
            if outcome.estimands:
                per_fold.append(outcome.estimands)
        if not per_fold:
            return {}
        wanted = ("target_auroc_transferred", "target_auroc_reference", "gap_auc", "delta_policy")
        out: dict[str, float] = {}
        for key in wanted:
            values = [f.get(key, float("nan")) for f in per_fold]
            finite = [v for v in values if np.isfinite(v)]
            out[key] = float(np.mean(finite)) if len(finite) == len(values) else float("nan")
        return out

    return refit_bootstrap(
        role_groups,
        procedure,
        n_resamples=int(n_resamples),
        seed=cfg.splits.seed,
        alpha=freeze.alpha_one_sided,
        scope_of=lambda key: key.rsplit("@", 1)[-1],
    )


def _write_results_report(
    cfg: Config, root: Path, results: dict[str, object], freeze: FreezeRecord
) -> Path:
    aggregate = results["aggregate"]
    bootstrap = results["bootstrap"]
    lines = [
        "# Confirmatory results",
        "",
        f"Generated: {results['generated']}",
        f"Direction: `{results['direction']}` (protocol {results['protocol']})",
        f"Freeze hash: `{results['freeze_hash']}`",
        f"Experiment id: `{results['experiment_id']}`",
        f"Frozen feature families: {', '.join(results['frozen_feature_families'])}",
        f"Evaluated folds: {results['n_folds']}",
        "",
        "## Joint claim",
        "",
        f"**{results['conclusion']}**",
        "",
        "| Component | Estimate | One-sided bound | Margin | Passed | Two-sided 95% |",
        "|---|---:|---:|---:|:--:|---|",
    ]
    if not results["joint_claim"]:
        raise CommandError(
            "the joint claim produced no components; the decision cannot be reported"
        )
    for row in results["joint_claim"]:
        low, high = row.get("ci_low", float("nan")), row.get("ci_high", float("nan"))
        reliable = "" if row.get("interval_reliable", True) else " (unreliable)"
        lines.append(
            f"| {row['component']} | {row['estimate']:.4f} | "
            f"{row['one_sided_bound']:.4f} ({row['bound_kind']}) | {row['margin']:.2f} | "
            f"{'yes' if row['passed'] else 'no'} | [{low:.4f}, {high:.4f}]{reliable} |"
        )
    lines += [
        "",
        "Bounds come from the complete refit bootstrap: source fitting and",
        "calibration, target-reference fitting and calibration, and target-test",
        "sampling are all resampled, so these are not conditional-on-fit intervals.",
        "",
        "## Primary estimands",
        "",
        "| Quantity | Estimate | Refit bootstrap 95% | Invalid resamples |",
        "|---|---:|---|---:|",
    ]
    for name in ("target_auroc_transferred", "target_auroc_reference", "gap_auc", "delta_policy"):
        row = bootstrap.get(name, {})
        lines.append(
            f"| `{name}` | {aggregate.get(name, float('nan')):.4f} | "
            f"[{row.get('ci_low', float('nan')):.4f}, {row.get('ci_high', float('nan')):.4f}] | "
            f"{row.get('n_invalid', 0)}/{row.get('n_resamples', 0)} |"
        )
    lines += [
        "",
        "## Realised policy operating points",
        "",
        "| Arm | Realised target coverage | Accepted groups |",
        "|---|---:|---:|",
        f"| frozen source policy | {aggregate.get('realised_coverage_source_on_target', float('nan')):.4f} "
        f"| {aggregate.get('accepted_groups_source_on_target', float('nan')):.1f} |",
        f"| matched-budget target reference | "
        f"{aggregate.get('realised_coverage_reference_on_target', float('nan')):.4f} "
        f"| {aggregate.get('accepted_groups_reference_on_target', float('nan')):.1f} |",
        "",
        "Neither threshold was chosen on target test scores; both come from their own",
        "calibration groups at the same nominal acceptance target.",
        "",
        "## Secondary endpoints",
        "",
        f"- `Delta_domain` (deployment/domain shift): {aggregate.get('delta_domain', float('nan')):.4f}",
        f"- `Delta_threshold` (threshold transport, target-adapted): "
        f"{aggregate.get('delta_threshold', float('nan')):.4f}",
        f"- Source-test AUROC of the transferred detector: "
        f"{aggregate.get('source_test_auroc_transferred', float('nan')):.4f}",
        f"- Target test failure prevalence: {aggregate.get('target_test_prevalence', float('nan')):.4f}",
        f"- Prevalence-standardised `Delta_domain` (§9.2a): "
        f"{aggregate.get('delta_domain_standardised', float('nan')):.4f} at reference "
        f"prevalence {aggregate.get('prevalence_reference_pi_star', float('nan')):.4f}",
        f"- Class-conditional acceptance change: a1 "
        f"{aggregate.get('delta_accept_given_failure', float('nan')):+.4f}, a0 "
        f"{aggregate.get('delta_accept_given_success', float('nan')):+.4f}",
        "",
        "### Ranking utility",
        "",
        "| Requested coverage | Risk on target test |",
        "|---:|---:|",
        "`Delta_domain` compares one frozen policy on two different populations and is",
        "not the transfer-cost estimand; `Delta_threshold` uses target adaptation and is",
        "outside the strict frozen-source result.",
        "",
    ]
    for key in sorted(k for k in aggregate if k.startswith("risk_at_coverage_")):
        lines.append(f"| {key.removeprefix('risk_at_coverage_')} | {aggregate[key]:.4f} |")
    lines += [
        "",
        f"Area under the risk-coverage curve over the attainable interval "
        f"[{aggregate.get('attainable_coverage_lower', float('nan')):.3f}, "
        f"{aggregate.get('attainable_coverage_upper', float('nan')):.3f}]: "
        f"{aggregate.get('risk_coverage_area', float('nan')):.4f}. A requested coverage "
        "above the attainable maximum is reported as undefined, not as the best "
        "coverage that could be delivered.",
        "",
        "## Required controls",
        "",
        "| Arm | Target AUROC |",
        "|---|---:|",
    ]
    for key in sorted(k for k in aggregate if k.startswith(("control_", "single_", "best_single"))):
        if key.endswith("__folds_defined"):
            continue
        lines.append(f"| `{key.removesuffix('_auroc')}` | {aggregate[key]:.4f} |")
    notes = results.get("fold_notes") or []
    if notes:
        lines += ["", "## Recorded limitations", ""] + [f"- {note}" for note in notes]
    if freeze.acknowledged_infeasible:
        lines += [
            "",
            "This direction was frozen despite unmet M2 screening minima:",
            "",
        ] + [f"- {reason}" for reason in freeze.feasibility.get("blocking", ())]
    lines.append("")
    path = cfg.paths.resolve(root)["reports"] / "RESULTS.md"
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def command_report(args: argparse.Namespace) -> int:
    """Build the manuscript tables and figures from frozen caches (§16.2).

    Nothing here recomputes a scientific value: every number is read from the
    manifests, caches, and results the earlier stages wrote, so a table cell can
    be traced to the row it came from. Figures are optional -- their data is
    always written as CSV, and a missing plotting library degrades the run to
    those tables instead of failing it.
    """
    cfg = load_config(args.config)
    root = _project_root(args.config)
    paths = cfg.paths.resolve(root)
    tables_dir, figures_dir = paths["tables"], paths["figures"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pair_rows, _ = _pairs_manifest(cfg, root)
    cache_root = paths["cache_root"]
    features = ShardedTable(cache_root, "features", key_column="feature_id").load()
    if not features.empty:
        features = features[features["config_hash"].astype(str) == cfg.hash]
    registrations = ShardedTable(cache_root, "registrations").load()

    written: dict[str, pd.DataFrame] = {
        "data_provenance": provenance_table(pair_rows, cfg.datasets),
    }
    if not features.empty and not registrations.empty:
        job_pipeline = (
            registrations.drop_duplicates("job_id", keep="last")
            .set_index("job_id")["pipeline_id"]
            .astype(str)
        )
        annotated = features.copy()
        annotated["pipeline_id"] = annotated["job_id"].astype(str).map(job_pipeline)
        written["pipeline_capability"] = capability_table(annotated, cfg.pipelines)
        cost = (
            annotated.drop_duplicates(["job_id", "family"])
            .groupby("family")["incremental_wall_s"]
            .agg(["median", "mean", "max", "size"])
            .reset_index()
            .rename(
                columns={
                    "median": "median_wall_s",
                    "mean": "mean_wall_s",
                    "max": "max_wall_s",
                    "size": "cases",
                }
            )
        )
        written["signal_cost"] = cost
    else:
        cost = pd.DataFrame()

    results_path = paths["manifests"] / "results.json"
    results: dict[str, object] = {}
    reliability = pd.DataFrame()
    if results_path.is_file():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        written["primary_joint_claim"] = primary_table(results)
        written["policy_operating_points"] = policy_table(results)
        written["signal_family_comparison"] = signal_family_table(results)
        secondary = results.get("aggregate", {})
        written["secondary_endpoints"] = pd.DataFrame(
            [
                {"endpoint": name, "estimate": secondary.get(name)}
                for name in (
                    "delta_domain",
                    "delta_threshold",
                    "source_test_auroc_transferred",
                    "target_test_prevalence",
                    "target_test_explicit_failure_fraction",
                    "brier_transferred",
                    "brier_reference",
                    "calibration_in_the_large_transferred",
                    "calibration_in_the_large_reference",
                )
            ]
        )
        predictions = ShardedTable(cache_root, "predictions", key_column="prediction_id").load()
        if not predictions.empty:
            families = list(results.get("frozen_feature_families", ()))
            cases, _ = _assemble_cases(cfg, root, families)
            written["per_pipeline_cells"] = per_pipeline_table(cases, predictions)
            reliability = reliability_frame(
                predictions, cases, bins=cfg.policy.reliability_bins
            )
            if not reliability.empty:
                written["calibration_bins"] = reliability
    else:
        print(
            "no results.json yet: writing the provenance, capability, and cost tables "
            "only. Run `warpaudit evaluate` for the confirmatory tables.",
            file=sys.stderr,
        )

    deviations = root / "PROTOCOL_DEVIATIONS.md"
    if deviations.is_file():
        written["protocol_deviations"] = pd.DataFrame(
            [{"source": "PROTOCOL_DEVIATIONS.md", "lines": len(deviations.read_text().splitlines())}]
        )

    empty: list[str] = []
    for name, frame in written.items():
        if frame is None or frame.empty:
            # Silence here once hid the primary table entirely; an expected table
            # that comes out empty is a finding, not a file to skip quietly.
            empty.append(name)
            continue
        atomic_write_text(tables_dir / f"{name}.csv", frame.to_csv(index=False))
        atomic_write_text(tables_dir / f"{name}.md", markdown_table(frame))
    if empty:
        print(f"tables with no rows: {sorted(empty)}", file=sys.stderr)

    figures = render_figures(
        figures_dir, written, results, reliability=reliability, costs=cost
    )
    available, reason = matplotlib_available()
    if not available:
        print(
            f"figures skipped: matplotlib is unavailable ({reason}). Install the "
            "reporting extra with `pip install -e '.[report]'`; the figure data is "
            "in the tables directory.",
            file=sys.stderr,
        )
    index = {
        "generated": _utc_now(),
        "config_hash": cfg.hash,
        "tables": sorted(f"{name}.csv" for name, f in written.items() if f is not None and not f.empty),
        "figures": sorted(figures),
        "figure_specs": [
            {"name": spec.name, "title": spec.title, "description": spec.description}
            for spec in FIGURE_SPECS
        ],
        "matplotlib_available": available,
        "results_present": bool(results),
        "empty_tables": sorted(empty),
    }
    index_path = paths["reports"] / "manuscript_outputs.json"
    atomic_write_text(index_path, json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


def command_failure_gallery(args: argparse.Namespace) -> int:
    """Render the failure gallery under a deterministic, disclosed selection rule.

    §16.2 requires the gallery to be selected *after* numeric results and its
    rule disclosed, and a hand-picked image is an illustration rather than
    statistical support. The rule here is fixed and label-free at selection
    time except for the class it is illustrating:

    * high-confidence errors -- operational failures the detector scored as
      least risky;
    * low-confidence successes -- successes it scored as most risky;
    * explicit failures -- rejected by construction, shown for completeness.

    Ties break on ``job_id``, so the selection is reproducible. Panels contain
    source pixels and are written only to the ignored local review directory.
    """
    cfg = load_config(args.config)
    root = _project_root(args.config)
    paths = cfg.paths.resolve(root)
    results_path = paths["manifests"] / "results.json"
    if not results_path.is_file():
        raise CommandError(
            "the gallery is selected after numeric results (§16.2); run "
            "`warpaudit evaluate` first"
        )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    predictions = ShardedTable(paths["cache_root"], "predictions", key_column="prediction_id").load()
    if predictions.empty:
        raise CommandError("no cached predictions to select a gallery from")
    predictions = predictions[predictions["arm"].astype(str) == args.arm]
    if predictions.empty:
        raise CommandError(f"no predictions for arm {args.arm!r}")

    cases, _ = _assemble_cases(cfg, root, list(results.get("frozen_feature_families", ())))
    merged = predictions.merge(
        cases[["job_id", "pair_id", "pipeline_id", "operational_failure", "explicit_failure",
               "tre_norm"]],
        on="job_id",
        how="inner",
    ).sort_values(["probability", "job_id"], kind="stable")
    if merged.empty:
        raise CommandError("no prediction row joins a scored case")

    silent_failures = merged[
        (merged["operational_failure"] > 0.5) & (~merged["explicit_failure"].astype(bool))
    ]
    successes = merged[merged["operational_failure"] <= 0.5]
    explicit = merged[merged["explicit_failure"].astype(bool)]
    selection = [
        ("high_confidence_error", silent_failures.head(args.per_class)),
        ("low_confidence_success", successes.tail(args.per_class).iloc[::-1]),
        ("explicit_failure", explicit.head(args.per_class)),
    ]

    pair_rows, _ = _pairs_manifest(cfg, root)
    manifest_by_pair = {str(row["pair_id"]): row for _, row in pair_rows.iterrows()}
    registrations = ShardedTable(paths["cache_root"], "registrations").load()
    registrations = registrations.drop_duplicates("job_id", keep="last").set_index("job_id")

    output_dir = paths["reports"] / "failure_gallery"
    output_dir.mkdir(parents=True, exist_ok=True)
    panels: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for category, frame in selection:
        for _, row in frame.iterrows():
            job_id = str(row["job_id"])
            if job_id not in registrations.index:
                continue
            registration = registrations.loc[job_id]
            pair = _pair_from_row(
                manifest_by_pair[str(row["pair_id"])],
                cfg,
                root,
                direction=str(registration["direction"]),
            )
            result = _result_from_row(registration)
            if result.forward_moving_to_fixed is None:
                records.append(
                    {
                        "category": category,
                        "job_id": job_id,
                        "pair_id": str(row["pair_id"]),
                        "pipeline_id": str(row["pipeline_id"]),
                        "probability": float(row["probability"]),
                        "accepted": bool(row["accepted"]),
                        "tre_norm": float(row["tre_norm"]),
                        "status": str(registration["status"]),
                        "file": "",
                        "note": "no transform to render",
                    }
                )
                continue
            with Image.open(pair.moving_path) as image:
                moving = np.asarray(image.convert("RGB"))
            with Image.open(pair.fixed_path) as image:
                fixed = np.asarray(image.convert("RGB"))
            panel, overlap = render_registration_panel(
                moving,
                fixed,
                result.forward_moving_to_fixed,
                pair.coordinates,
                title=f"{category}; {pair.pair_id}; {row['pipeline_id']}; "
                f"p={float(row['probability']):.3f}; tre_norm={float(row['tre_norm']):.4f}",
            )
            filename = f"{category}-{len(panels):03d}-{job_id}.png"
            atomic_write_bytes(output_dir / filename, _png_bytes(panel))
            panels.append(panel)
            records.append(
                {
                    "category": category,
                    "job_id": job_id,
                    "pair_id": str(row["pair_id"]),
                    "pipeline_id": str(row["pipeline_id"]),
                    "probability": float(row["probability"]),
                    "accepted": bool(row["accepted"]),
                    "tre_norm": float(row["tre_norm"]),
                    "status": str(registration["status"]),
                    "overlap_fraction": float(overlap),
                    "file": filename,
                    "note": "",
                }
            )
    if panels:
        atomic_write_bytes(
            output_dir / "contact_sheet.png",
            _png_bytes(contact_sheet(panels, columns=args.columns, thumbnail_width=720)),
        )
    ledger = {
        "generated": _utc_now(),
        "arm": args.arm,
        "selection_rule": (
            "per class, ordered by predicted failure probability with job_id breaking "
            "ties: lowest-probability operational failures, highest-probability "
            "successes, and explicit failures"
        ),
        "per_class": args.per_class,
        "experiment_id": results.get("experiment_id"),
        "records": records,
    }
    atomic_write_text(
        output_dir / "index.json", json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    )
    # The panels carry source pixels; only the text ledger is releasable.
    atomic_write_text(
        cfg.paths.resolve(root)["reports"] / "FAILURE_GALLERY.md",
        _render_gallery_ledger(ledger, output_dir.relative_to(root)),
    )
    print(json.dumps({"panels": len(panels), "directory": str(output_dir)}))
    return 0


def _render_gallery_ledger(ledger: dict[str, object], directory: Path) -> str:
    lines = [
        "# Failure gallery",
        "",
        f"Generated: {ledger['generated']}",
        f"Arm: `{ledger['arm']}`",
        f"Panels: `{directory}` (local only; the panels contain source pixels)",
        "",
        f"Selection rule: {ledger['selection_rule']}.",
        "",
        "The gallery illustrates; it is never statistical support.",
        "",
        "| Category | Pair | Pipeline | Predicted risk | Accepted | tre_norm | Status |",
        "|---|---|---|---:|:--:|---:|---|",
    ]
    for record in ledger["records"]:
        lines.append(
            f"| {record['category']} | {record['pair_id']} | {record['pipeline_id']} | "
            f"{record['probability']:.4f} | {'yes' if record['accepted'] else 'no'} | "
            f"{record['tre_norm']:.4f} | {record['status']} |"
        )
    return "\n".join(lines) + "\n"


def command_reproduce(args: argparse.Namespace) -> int:
    """Rebuild every manuscript output from frozen caches and check it agrees.

    This is the release claim of §12.2 made executable: a third party with the
    caches must be able to regenerate the reported numbers. The command reruns
    the confirmatory evaluation and the manuscript build, then compares the new
    estimands against the recorded ones and reports any drift instead of
    overwriting the record silently.
    """
    cfg = load_config(args.config)
    root = _project_root(args.config)
    paths = cfg.paths.resolve(root)
    results_path = paths["manifests"] / "results.json"
    previous = (
        json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else {}
    )
    if not previous and not args.allow_missing_baseline:
        raise CommandError(
            "there is no recorded results.json to reproduce. Run `warpaudit evaluate` "
            "first, or pass --allow-missing-baseline to build the outputs from scratch."
        )

    evaluate_args = argparse.Namespace(
        config=args.config, refit_bootstrap=args.refit_bootstrap
    )
    command_evaluate(evaluate_args)
    command_report(argparse.Namespace(config=args.config))

    current = json.loads(results_path.read_text(encoding="utf-8"))
    checked = (
        "target_auroc_transferred",
        "target_auroc_reference",
        "gap_auc",
        "delta_policy",
        "delta_domain",
        "delta_threshold",
    )
    drift: list[dict[str, object]] = []
    for key in checked:
        before = float(previous.get("aggregate", {}).get(key, float("nan")))
        after = float(current.get("aggregate", {}).get(key, float("nan")))
        if not previous:
            continue
        both_nan = not np.isfinite(before) and not np.isfinite(after)
        agree = both_nan or (
            np.isfinite(before)
            and np.isfinite(after)
            and abs(before - after) <= args.tolerance
        )
        if not agree:
            drift.append({"estimand": key, "recorded": before, "rebuilt": after})

    identity_drift = [
        {"field": field, "recorded": previous.get(field), "rebuilt": current.get(field)}
        for field in ("experiment_id", "freeze_hash", "direction", "frozen_feature_families")
        if previous and previous.get(field) != current.get(field)
    ]
    payload = {
        "generated": _utc_now(),
        "config_hash": cfg.hash,
        "baseline_present": bool(previous),
        "tolerance": args.tolerance,
        "estimand_drift": drift,
        "identity_drift": identity_drift,
        "reproduced": not drift and not identity_drift,
    }
    atomic_write_text(
        paths["reports"] / "REPRODUCTION.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if drift or identity_drift:
        raise CommandError(
            "the rebuilt outputs do not match the recorded ones; see "
            f"{paths['reports'] / 'REPRODUCTION.json'}"
        )
    return 0


def command_inspect_provenance(args: argparse.Namespace) -> int:
    """Report what this machine's matcher environments actually resolved.

    The provenance gate compares exact strings, so a machine whose CPython
    patch release or CUDA build differs from the recorded pins will refuse to
    register anything. That refusal is correct, and relaxing the check would
    defeat it -- but the operator still needs to know precisely what to record.

    This runs the matcher on the synthetic probe pair with the comparison
    switched off, prints the observed values beside the configured ones, and
    emits a ``provenance:`` block ready to paste. Nothing is written to the
    configuration and no cached row can be produced: re-pinning stays a
    deliberate protocol change to record in PROTOCOL_DEVIATIONS.md.
    """
    cfg = load_config(args.config)
    root = _project_root(args.config)
    pipelines = (
        [cfg.pipeline(args.pipeline)]
        if args.pipeline
        else [p for p in cfg.pipelines if p.adapter == "subprocess"]
    )
    if not pipelines:
        raise CommandError("no subprocess pipeline is configured")

    report: list[dict[str, object]] = []
    for pipeline in pipelines:
        if pipeline.adapter != "subprocess":
            raise CommandError(
                f"pipeline {pipeline.id!r} is not a subprocess adapter; provenance is "
                "only reported for isolated matcher environments",
                EXIT_USAGE,
            )
        try:
            registrar = subprocess_factory(
                pipeline, project_root=root, strict_provenance=False
            )
            result = registrar.register(_probe_pair(root), seed=cfg.splits.seed)
        except (AdapterUnavailable, AdapterProtocolError, OSError, ValueError) as exc:
            report.append({"pipeline": pipeline.id, "error": str(exc)})
            print(f"{pipeline.id}: could not reach the matcher: {exc}", file=sys.stderr)
            continue
        actual = {
            str(k): str(v) for k, v in (result.diagnostics.get("provenance") or {}).items()
        }
        differences = {
            key: {"configured": expected, "observed": actual.get(key)}
            for key, expected in pipeline.provenance.items()
            if actual.get(key) != expected
        }
        report.append(
            {
                "pipeline": pipeline.id,
                "matches": not differences,
                "observed": actual,
                "differences": differences,
            }
        )
        print(f"\n{pipeline.id}: {'matches the configuration' if not differences else 'DIFFERS'}")
        for key in sorted(set(pipeline.provenance) | set(actual)):
            expected = pipeline.provenance.get(key, "(not configured)")
            observed = actual.get(key, "(not reported)")
            mark = " " if expected == observed else "*"
            print(f"  {mark} {key}: configured {expected!r}, observed {observed!r}")
        if differences:
            print(f"\n  Paste into configs for pipeline {pipeline.id}:")
            print("    provenance:")
            for key in sorted(actual):
                print(f'      {key}: "{actual[key]}"')

    changed = [entry for entry in report if entry.get("differences")]
    protected = {"upstream_commit"} | {
        key for entry in report for key in entry.get("observed", {}) if key.endswith("_sha256")
    }
    if changed and args.write:
        # Sources and checkpoints are the identity of what is being measured, not
        # a property of the machine. A machine that resolves a different commit
        # or weight hash has the wrong artefact, and re-pinning would record the
        # mistake as the protocol.
        blocked = {
            entry["pipeline"]: sorted(set(entry["differences"]) & protected)
            for entry in changed
            if set(entry["differences"]) & protected
        }
        if blocked:
            raise CommandError(
                f"refusing to re-pin: {blocked} differ, and an upstream commit or "
                "checkpoint hash mismatch means the sources or weights are wrong, not "
                "the pins. Fix the checkout or re-download the weights."
            )
        # Writing what a matcher failed to report would erase the pins rather
        # than update them, and an empty provenance block gates nothing.
        dropped = {
            entry["pipeline"]: sorted(
                set(cfg.pipeline(str(entry["pipeline"])).provenance) - set(entry["observed"])
            )
            for entry in changed
            if set(cfg.pipeline(str(entry["pipeline"])).provenance) - set(entry["observed"])
        }
        if dropped:
            raise CommandError(
                f"refusing to re-pin: the matcher reported no value for {dropped}. "
                "Writing that would delete those pins instead of updating them; the "
                "worker's provenance payload is incomplete."
            )
        for entry in changed:
            replace_pipeline_provenance(
                args.config, str(entry["pipeline"]), dict(entry["observed"])
            )
        deviations = root / "PROTOCOL_DEVIATIONS.md"
        record = [
            "",
            f"## {_utc_now()} -- matcher provenance re-pinned for this machine",
            "",
            f"Host: {platform.platform()} ({platform.machine()}).",
            "",
        ]
        for entry in changed:
            for key, value in sorted(entry["differences"].items()):
                record.append(
                    f"- `{entry['pipeline']}.{key}`: `{value['configured']}` -> "
                    f"`{value['observed']}`"
                )
        record += [
            "",
            "Upstream commits and checkpoint hashes are unchanged; only interpreter and "
            "library builds differ. The configuration hash moves with these pins, so this "
            "machine computes its own caches instead of reusing rows produced under a "
            "different stack.",
            "",
        ]
        with open(deviations, "a", encoding="utf-8") as fh:
            fh.write("\n".join(record))
        print(
            f"\nRe-pinned {[e['pipeline'] for e in changed]} in {args.config} and recorded "
            f"the change in {deviations}. Rerun the probes before the study.",
            file=sys.stderr,
        )
    elif changed:
        print(
            "\nThese are protocol pins. Update them deliberately (or rerun with --write), "
            "rerun the probes until they pass, and record the change in "
            "PROTOCOL_DEVIATIONS.md. The configuration hash changes with them, so this "
            "machine computes its own caches rather than reusing rows produced under a "
            "different stack.",
            file=sys.stderr,
        )
    if args.output:
        atomic_write_text(
            _resolve(root, args.output), json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    return 0


def _probe_pair(root: Path) -> PairInput:
    """The synthetic pair used by probe-adapter, without dataset access."""
    rng = np.random.default_rng(20260907)
    height = width = 256
    moving = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    moving[30:70, 25:110] = (230, 40, 30)
    moving[120:190, 145:210] = (20, 210, 90)
    fixed = np.roll(moving, shift=(3, 5), axis=(0, 1))
    directory = Path(tempfile.mkdtemp(prefix="warpaudit-provenance-"))
    moving_path, fixed_path = directory / "moving.png", directory / "fixed.png"
    Image.fromarray(moving).save(moving_path)
    Image.fromarray(fixed).save(fixed_path)
    return PairInput(
        dataset_id="adapter-probe",
        pair_id="adapter-probe/provenance",
        group_id="adapter-probe/group",
        group_basis="image_component",
        moving_image_id="probe-moving",
        fixed_image_id="probe-fixed",
        moving_path=moving_path,
        fixed_path=fixed_path,
        coordinates=CoordinateMetadata(
            moving=make_frame("probe-moving", (height, width), long_edge=width),
            fixed=make_frame("probe-fixed", (height, width), long_edge=width),
        ),
    )


def command_estimate_cost(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    root = _project_root(args.config)
    cache_root = cfg.paths.resolve(root)["cache_root"]
    frame = ShardedTable(cache_root, "registrations").load()
    if frame.empty:
        raise CommandError(
            "no measured registration rows found; cost estimation refuses to substitute "
            "literature examples for local timings"
        )
    pair_rows, _ = _pairs_manifest(cfg, root)
    frame = _current_registration_rows(frame, pair_rows, cfg, root)
    if frame.empty:
        raise CommandError("no registration timings match the current execution contract")
    required = {"pipeline_id", "runtime_s"}
    if not required <= set(frame):
        raise CommandError(
            f"registration cache lacks timing columns: {sorted(required - set(frame))}"
        )
    valid = frame[np.isfinite(pd.to_numeric(frame["runtime_s"], errors="coerce"))].copy()
    if valid.empty:
        raise CommandError("registration cache contains no finite measured runtime_s values")
    valid["job_class"] = np.where(
        valid["condition"].astype(str).str.startswith("e2:"),
        "e2_perturbed",
        valid["condition"].astype(str),
    )
    summary = valid.groupby(["pipeline_id", "job_class"])["runtime_s"].agg(
        ["count", "median", "mean", "max"]
    )
    report = cfg.paths.resolve(root)["reports"] / "COST_ESTIMATE.md"
    table = [
        "| Pipeline | Job class | Jobs | Median s | Mean s | Max s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (pipeline_id, job_class), row in summary.iterrows():
        table.append(
            f"| {pipeline_id} | {job_class} | {int(row['count'])} | {row['median']:.4f} | "
            f"{row['mean']:.4f} | {row['max']:.4f} |"
        )
    body = [
        "# Measured cost basis",
        "",
        f"Generated: {_utc_now()}",
        "",
        *table,
        "",
        "All values are measured wall-clock seconds per cached job.",
        "",
    ]
    atomic_write_text(report, "\n".join(body))
    print(report)
    return 0


def command_evaluation_card(args: argparse.Namespace) -> int:
    """Evaluate somebody else's scores using the paper's three-layer contract."""
    source = Path(args.input).resolve()
    if not source.is_file():
        raise CommandError(f"score CSV does not exist: {source}", EXIT_USAGE)
    try:
        frame = pd.read_csv(source)
        requested = tuple(args.coverage) if args.coverage else (0.5, 0.7, 0.8, 0.9)
        card = evaluate_scores(frame, requested_coverages=requested)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise CommandError(f"invalid score table: {exc}", EXIT_USAGE) from exc
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    curve_path = destination.with_name(destination.stem + "_curve.csv")
    clean = json.loads(
        json.dumps(
            card.summary,
            default=lambda x: x.item() if isinstance(x, np.generic) else x,
            allow_nan=True,
        ).replace("NaN", "null")
    )
    atomic_write_text(destination, json.dumps(clean, indent=2, allow_nan=False) + "\n")
    atomic_write_text(curve_path, card.curve.to_csv(index=False))
    print(json.dumps({"summary": str(destination), "curve": str(curve_path)}, indent=2))
    return 0


def command_run_study(args: argparse.Namespace) -> int:
    """Run every executable stage of the study in dependency order, unattended.

    The run is resumable by construction: each stage is an ordinary CLI command
    whose work is already keyed by job identity, so a rerun recomputes only what
    is missing or whose provenance contract changed. A failed stage stops the
    stages that declare it as a prerequisite and never silently invalidates the
    ones that already succeeded.
    """
    cfg = load_config(args.config)
    root = _project_root(args.config)
    plan = build_plan(
        cfg,
        config_path=str(args.config),
        download=args.download,
        acknowledge_fire_terms=args.acknowledge_fire_terms_unresolved,
        include_confirmatory=not args.development_only,
        e2_limit=args.e2_limit,
        e2_sample_cap=args.e2_sample_cap,
        signed_off_by=args.signed_off_by,
        review_note=args.review_note,
        acknowledge_infeasible=args.acknowledge_infeasible,
        refit_bootstrap=args.refit_bootstrap,
        feature_workers=getattr(args, "feature_workers", 1),
        feature_worker_threads=getattr(args, "feature_worker_threads", 0),
    )
    try:
        plan = plan.select(
            only=tuple(args.only), skip=tuple(args.skip), start=args.start_at or ""
        )
    except ValueError as exc:
        raise CommandError(
            f"{exc}; known stages: {list(build_plan(cfg, config_path=str(args.config)).ids())}",
            EXIT_USAGE,
        ) from exc
    if not len(plan):
        raise CommandError("stage selection is empty", EXIT_USAGE)

    preflight = _study_preflight(cfg, root, min_free_gb=args.min_free_gb)
    blocking = [check for check in preflight if not check["ok"]]
    if blocking and not args.dry_run:
        raise CommandError(
            "preflight failed before any stage ran:\n  - "
            + "\n  - ".join(str(check["detail"]) for check in blocking)
        )

    started = _utc_now()
    run_id = started.replace("-", "").replace(":", "").replace("+0000", "Z")
    log_root = cfg.paths.resolve(root)["reports"] / "study_runs" / run_id
    log_root.mkdir(parents=True, exist_ok=True)
    ledger = StatusLedger(log_root / "stages.jsonl")

    header = {
        "run_id": run_id,
        "started": started,
        "preflight": preflight,
        "config": str(args.config),
        "config_hash": cfg.hash,
        "git_commit": _git_commit(root),
        "stages": [
            {
                "id": stage.id,
                "title": stage.title,
                "requires": list(stage.requires),
                "heavy": stage.heavy,
                "commands": [list(c) for c in stage.commands],
            }
            for stage in plan
        ],
    }
    print(json.dumps(header, indent=2, sort_keys=True))
    if args.dry_run:
        for line in _project_study_cost(cfg, root, plan):
            print(line)
        return 0

    results: list[dict[str, object]] = []
    succeeded: set[str] = set()
    failed: set[str] = set()
    for stage in plan:
        unmet = [r for r in stage.requires if r in failed]
        if unmet:
            outcome = {
                "stage": stage.id,
                "state": "skipped",
                "message": f"unmet prerequisite(s): {unmet}",
                "runtime_s": 0.0,
                "commands": [],
            }
            results.append(outcome)
            failed.add(stage.id)
            ledger.append(
                JobRecord(
                    job_id=stage.id,
                    kind="stage",
                    state="failed",
                    status="skipped",
                    message=str(outcome["message"]),
                    runtime_s=0.0,
                    finished_at=time.time(),
                )
            )
            print(f"[skip] {stage.id}: {outcome['message']}", flush=True)
            continue

        print(f"[run ] {stage.id}: {stage.title}", flush=True)
        stage_started = time.perf_counter()
        command_records: list[dict[str, object]] = []
        state = "done"
        message = ""
        for index, argv in enumerate(stage.commands):
            log_path = log_root / f"{stage.id}.{index:02d}.log"
            command_started = time.perf_counter()
            code = _run_stage_command(argv, root=root, log_path=log_path, echo=not args.quiet)
            elapsed = time.perf_counter() - command_started
            command_records.append(
                {
                    "argv": list(argv),
                    "exit_code": code,
                    "runtime_s": round(elapsed, 3),
                    "log": str(log_path.relative_to(root)),
                }
            )
            print(
                f"       {'ok  ' if code == 0 else 'FAIL'} "
                f"{' '.join(argv[:3])} [{elapsed:.1f}s] -> {log_path.name}",
                flush=True,
            )
            if code != 0:
                state = "failed"
                message = f"`warpaudit {' '.join(argv)}` exited {code}; see {log_path}"
                break
        runtime = time.perf_counter() - stage_started
        results.append(
            {
                "stage": stage.id,
                "title": stage.title,
                "state": state,
                "message": message,
                "runtime_s": round(runtime, 3),
                "commands": command_records,
            }
        )
        ledger.append(
            JobRecord(
                job_id=stage.id,
                kind="stage",
                state=state,
                status=state,
                message=message,
                runtime_s=round(runtime, 3),
                finished_at=time.time(),
            )
        )
        if state == "done":
            succeeded.add(stage.id)
        else:
            failed.add(stage.id)
            if not args.keep_going and not stage.optional:
                print(f"[stop] {stage.id} failed; {message}", file=sys.stderr, flush=True)
                break

    summary_path = _write_study_run_report(
        cfg, root, plan, header, results, log_root=log_root, finished=_utc_now()
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "succeeded": sorted(succeeded),
                "failed": sorted(failed),
                "not_run": [s.id for s in plan if s.id not in succeeded | failed],
                "logs": str(log_root),
                "summary": str(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failed else EXIT_STAGE_FAILED


def _project_study_cost(cfg: Config, root: Path, plan: StagePlan) -> list[str]:
    """Project the run's wall time from this machine's own measured timings.

    Only this machine's measurements are used, and totals follow the *mean*
    rather than the median: E1's long tail makes a median-based projection
    optimistic by roughly a factor of three. With an empty cache it says so
    instead of substituting a literature figure, because an unmeasured estimate
    is exactly what would mislead an operator deciding whether a run fits in a
    night.
    """
    cache_root = cfg.paths.resolve(root)["cache_root"]
    try:
        pairs, _ = _pairs_manifest(cfg, root)
    except CommandError:
        return ["", "Cost projection unavailable: no pair manifest yet."]

    registrations = ShardedTable(cache_root, "registrations").load()
    seconds: dict[str, float] = {}
    if not registrations.empty and {"pipeline_id", "runtime_s", "condition"} <= set(registrations):
        clean = registrations[registrations["condition"].astype(str) == "clean"]
        runtimes = pd.to_numeric(clean.get("runtime_s"), errors="coerce")
        finite = clean[np.isfinite(runtimes)]
        if not finite.empty:
            seconds = (
                pd.to_numeric(finite["runtime_s"], errors="coerce")
                .groupby(finite["pipeline_id"].astype(str))
                .mean()
                .to_dict()
            )

    features = ShardedTable(cache_root, "features", key_column="feature_id").load()
    family_seconds: dict[str, float] = {}
    if not features.empty and {"family", "job_id", "incremental_wall_s"} <= set(features):
        per_job = (
            features.drop_duplicates(["job_id", "family"])
            .groupby("family")["incremental_wall_s"]
            .mean()
        )
        family_seconds = {str(k): float(v) for k, v in per_job.items() if np.isfinite(v)}

    split_pairs = {
        "development": int(pairs["is_development"].astype(bool).sum()),
        "confirmatory": int((~pairs["is_development"].astype(bool)).sum()),
    }
    lines = ["", "| Stage | Jobs | Projected wall time | Basis |", "|---|---:|---:|---|"]
    total = 0.0
    unmeasured: list[str] = []
    for stage in plan:
        jobs = 0
        projected = 0.0
        basis = ""
        for argv in stage.commands:
            if argv[0] == "register":
                split = argv[argv.index("--split") + 1]
                pipeline_id = argv[argv.index("--pipeline") + 1]
                count = split_pairs[split] * 2  # canonical and reverse
                jobs += count
                if pipeline_id in seconds:
                    projected += count * seconds[pipeline_id]
                    basis = "measured mean per registration"
                else:
                    unmeasured.append(stage.id)
            elif argv[0] == "features":
                split = argv[argv.index("--split") + 1]
                families = list(
                    itertools.takewhile(
                        lambda token: not token.startswith("--"),
                        argv[argv.index("--families") + 1 :],
                    )
                )
                count = split_pairs[split] * len([p for p in cfg.pipelines if p.in_common_block])
                jobs += count
                known = [f for f in families if f in family_seconds]
                if len(known) == len(families):
                    projected += count * sum(family_seconds[f] for f in known)
                    basis = "measured mean per family"
                else:
                    unmeasured.append(stage.id)
        if not jobs:
            continue
        total += projected
        lines.append(
            f"| `{stage.id}` | {jobs} | "
            + (f"{projected / 3600:.2f} h" if projected else "unmeasured")
            + f" | {basis or 'no local measurement yet'} |"
        )
    lines.append("")
    if total:
        lines.append(f"Projected heavy-stage wall time: {total / 3600:.2f} h.")
    if unmeasured:
        lines.append(
            "Unprojected stage(s): "
            + ", ".join(sorted(set(unmeasured)))
            + ". They have no measured timing on this machine yet."
        )
    lines.append("Download, audit, and report stages are excluded; they are minutes, not hours.")
    return lines


def _study_preflight(cfg: Config, root: Path, *, min_free_gb: float) -> list[dict[str, object]]:
    """Cheap checks that would otherwise fail hours into an unattended run.

    Each check reports rather than raises, so one call names every problem the
    operator has to fix instead of revealing them one restart at a time.
    """
    checks: list[dict[str, object]] = []

    free_gb = shutil.disk_usage(root).free / 1024**3
    checks.append(
        {
            "check": "free_disk",
            "ok": free_gb >= min_free_gb,
            "detail": f"{free_gb:.1f} GB free at {root}; the archives, extracted images, "
            f"and caches need about {min_free_gb:.0f} GB",
        }
    )

    paths = cfg.paths.resolve(root)
    for name in ("cache_root", "manifests", "reports"):
        directory = paths[name]
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".writable"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            writable = True
            detail = f"{name} is writable at {directory}"
        except OSError as exc:
            writable = False
            detail = f"{name} is not writable at {directory}: {exc}"
        checks.append({"check": f"writable:{name}", "ok": writable, "detail": detail})

    if sys.platform == "win32":
        # COph100 examination filenames run to about 80 characters, and Windows
        # refuses paths past 260 unless long-path support is enabled. A project
        # rooted deep in a user profile silently exceeds that only once the
        # extraction reaches those files, which is a confusing place to fail.
        longest_expected = 140
        headroom = 260 - (len(str(root)) + longest_expected)
        checks.append(
            {
                "check": "windows_path_length",
                "ok": headroom > 0,
                "detail": (
                    f"project path is {len(str(root))} characters; the longest dataset "
                    f"path needs about {longest_expected} more and Windows stops at 260. "
                    "Clone nearer the drive root (C:\\warpaudit), or enable long paths."
                )
                if headroom <= 0
                else f"{headroom} characters of path headroom for dataset filenames",
            }
        )

    unverified = [
        pipeline.id
        for pipeline in cfg.pipelines
        if pipeline.in_common_block
        and "[VERIFY]" in f"{pipeline.version}{pipeline.checkpoint}"
    ]
    checks.append(
        {
            "check": "pipeline_contracts",
            "ok": not unverified,
            "detail": (
                f"common-block pipelines still marked [VERIFY]: {unverified}; run "
                "./scripts/setup_matcher_envs.sh on this machine first"
            )
            if unverified
            else "every common-block pipeline has a pinned version and checkpoint",
        }
    )

    missing_env: list[str] = []
    for pipeline in cfg.pipelines:
        if not pipeline.in_common_block or not pipeline.command:
            continue
        interpreter = Path(pipeline.command[0])
        if not interpreter.is_absolute():
            interpreter = root / interpreter
        # Same POSIX/Windows venv-layout translation the adapter applies, so the
        # preflight cannot report a missing interpreter that will in fact run.
        interpreter = _venv_interpreter(interpreter)
        if not interpreter.exists():
            missing_env.append(f"{pipeline.id}: {interpreter}")
    checks.append(
        {
            "check": "matcher_environments",
            "ok": not missing_env,
            "detail": (
                f"matcher interpreter(s) absent: {missing_env}; run "
                "./scripts/setup_matcher_envs.sh on this machine first"
            )
            if missing_env
            else "every configured matcher interpreter exists",
        }
    )
    return checks


def _run_stage_command(
    argv: tuple[str, ...], *, root: Path, log_path: Path, echo: bool
) -> int:
    """Run one CLI command as a subprocess, teeing its output to a log file.

    A subprocess per stage keeps a long sweep's memory from accumulating across
    stages and contains a hard crash in an upstream matcher stack, which an
    in-process call would not.
    """
    command = [sys.executable, "-m", "warpaudit", *argv]
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if echo:
                sys.stdout.write(f"       | {line}")
                sys.stdout.flush()
        return process.wait()


def _write_study_run_report(
    cfg: Config,
    root: Path,
    plan: StagePlan,
    header: dict[str, object],
    results: list[dict[str, object]],
    *,
    log_root: Path,
    finished: str,
) -> Path:
    by_id = {str(r["stage"]): r for r in results}
    total = sum(float(r["runtime_s"]) for r in results)
    lines = [
        "# Study run",
        "",
        f"Run id: `{header['run_id']}`",
        f"Started: {header['started']}",
        f"Finished: {finished}",
        f"Config hash: `{header['config_hash']}`",
        f"Git commit: `{header['git_commit']}`",
        f"Stage logs: `{log_root.relative_to(root)}`",
        "",
        "| Stage | State | Wall time | Detail |",
        "|---|---|---:|---|",
    ]
    for stage in plan:
        record = by_id.get(stage.id)
        if record is None:
            lines.append(f"| `{stage.id}` | not run | - | run stopped before this stage |")
            continue
        detail = str(record.get("message") or stage.purpose.split(".")[0])
        lines.append(
            f"| `{stage.id}` | {record['state']} | {float(record['runtime_s']) / 60:.1f} min | "
            f"{detail} |"
        )
    lines += [
        "",
        f"Total stage wall time: {total / 3600:.2f} h.",
        "",
        "## Boundaries this run did not cross",
        "",
        "- Confirmatory outcomes were never read. Registration and feature stages "
        "over the reserve are ground-truth-free; no label command in the plan "
        "accepts the confirmatory split.",
        "- No direction was selected from results. `plan-study` screens information "
        "from development evidence only, and its recommendation is a precondition "
        "for the freeze, not a finding.",
        "",
    ]
    failures = [r for r in results if r["state"] != "done"]
    if failures:
        lines += ["## Failures", ""]
        for record in failures:
            lines.append(f"- `{record['stage']}`: {record['message']}")
        lines.append("")
    path = cfg.paths.resolve(root)["reports"] / "STUDY_RUN.md"
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path


#: Configuration fields that provably cannot change a cached registration,
#: label, or feature value. Each is read only by the M2 direction planner and
#: the confirmatory fold allocation -- stages that run *after* every cached
#: value already exists. Editing one of them moves ``Config.hash``, which keys
#: ``feature_id``, so the feature cache would otherwise be discarded for a
#: change that cannot alter a single feature. Nothing that feeds registration,
#: label, or feature computation belongs in this set: adding such a field here
#: would silently reuse rows that the new configuration should have recomputed.
FEATURE_NEUTRAL_CONFIG_FIELDS: frozenset[tuple[str, ...]] = frozenset(
    {
        ("primary_direction",),
        ("splits", "min_outer_folds"),
        ("splits", "low_information_group_threshold"),
        ("splits", "conformal_min_calibration_groups"),
    }
)


def _flatten_config(payload: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    """Flatten a config mapping to leaf paths so two configs can be diffed."""
    if isinstance(payload, dict):
        out: dict[tuple[str, ...], object] = {}
        for key, value in payload.items():
            out.update(_flatten_config(value, (*prefix, str(key))))
        return out
    return {prefix: payload}


def _changed_hashed_fields(previous: Config, current: Config) -> set[tuple[str, ...]]:
    """Leaf paths that differ between two configs, ignoring hash-excluded sections."""
    def hashed(cfg: Config) -> dict[tuple[str, ...], object]:
        payload = cfg.as_dict()
        for section in Config.HASH_EXCLUDED_SECTIONS:
            payload.pop(section, None)
        return _flatten_config(payload)

    before, after = hashed(previous), hashed(current)
    return {key for key in set(before) | set(after) if before.get(key) != after.get(key)}


def _rekey_registrations(cache_root: Path, previous_hash: str, current_hash: str) -> int:
    """Move cached registration rows onto the current execution contract.

    Registration rows key on ``job_id`` and carry ``config_hash`` as a contract
    field, so a re-key is a plain column rewrite: the appended rows supersede
    their predecessors under the same key.
    """
    table = ShardedTable(cache_root, "registrations")
    frame = table.load()
    if frame.empty or "config_hash" not in frame:
        return 0
    carried = frame[frame["config_hash"].astype(str) == previous_hash].copy()
    if carried.empty:
        return 0
    carried["config_hash"] = current_hash
    table.append(carried.to_dict(orient="records"), shard_hint="rekey")
    return len(carried)


def _rekey_features(cache_root: Path, previous_hash: str, current_hash: str) -> int:
    """Move cached feature rows onto the current contract, recomputing their key.

    ``feature_id`` hashes the configuration, so unlike a registration row a
    feature row changes identity when the configuration hash moves.
    """
    table = ShardedTable(cache_root, "features", key_column="feature_id")
    frame = table.load()
    if frame.empty:
        return 0
    required = {"feature_id", "job_id", "feature_hash", "config_hash"}
    if not required <= set(frame):
        raise CommandError(
            f"feature cache lacks provenance columns: {sorted(required - set(frame))}"
        )
    carried = frame[frame["config_hash"].astype(str) == previous_hash].copy()
    if carried.empty:
        return 0
    carried["config_hash"] = current_hash
    carried["feature_id"] = [
        short_hash(
            {"job_id": job_id, "feature_hash": feature_hash, "config_hash": current_hash}
        )
        for job_id, feature_hash in zip(
            carried["job_id"].astype(str), carried["feature_hash"].astype(str), strict=True
        )
    ]
    already = set(
        frame[frame["config_hash"].astype(str) == current_hash]["feature_id"].astype(str)
    )
    rows = [row for row in carried.to_dict(orient="records") if row["feature_id"] not in already]
    if rows:
        table.append(rows, shard_hint="rekey")
    return len(rows)


def command_rekey_cache(args: argparse.Namespace) -> int:
    """Carry a cache across a configuration change that cannot alter its values (§12.4).

    Registration and feature rows both record ``config_hash`` as part of their
    execution contract, and both are discarded when it moves -- correctly, since
    almost every configuration field can change what they contain. A few fields
    cannot: they are read only by the direction planner and the confirmatory
    fold allocation, after every cached value already exists. This re-keys the
    cache for exactly those, and refuses otherwise, because a cache that
    silently survives a change to how its values are computed is worse than no
    cache. Writes are append-only, so superseded rows stay on disk and the
    migration can be audited or ignored.
    """
    cfg = load_config(args.config)
    root = _project_root(args.config)
    previous = load_config(args.previous_config)
    if previous.hash == cfg.hash:
        print(f"configuration hash is unchanged ({cfg.hash}); the cache already applies")
        return 0

    changed = _changed_hashed_fields(previous, cfg)
    unsafe = sorted(changed - FEATURE_NEUTRAL_CONFIG_FIELDS)
    if unsafe:
        raise CommandError(
            "refusing to re-key the cache: these fields can change a cached value, so "
            "the affected rows must be recomputed rather than relabelled:\n  - "
            + "\n  - ".join(".".join(field) for field in unsafe)
        )

    cache_root = cfg.paths.resolve(root)["cache_root"]
    print(f"{previous.hash} -> {cfg.hash}")
    print("cache-neutral change(s): " + ", ".join(sorted(".".join(f) for f in changed)))
    if args.dry_run:
        for name, key in (("registrations", "job_id"), ("features", "feature_id")):
            frame = ShardedTable(cache_root, name, key_column=key).load()
            n = (
                0
                if frame.empty or "config_hash" not in frame
                else int((frame["config_hash"].astype(str) == previous.hash).sum())
            )
            print(f"  {name}: {n} row(s) would be re-keyed")
        print("dry run: no shard written")
        return 0

    registrations = _rekey_registrations(cache_root, previous.hash, cfg.hash)
    features = _rekey_features(cache_root, previous.hash, cfg.hash)
    if not registrations and not features:
        raise CommandError(
            f"no cached row carries {previous.hash} under {cache_root}; nothing to re-key"
        )
    print(f"re-keyed {registrations} registration row(s) and {features} feature row(s)")
    print("superseded rows are retained; labels carry no config contract and were untouched")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="warpaudit",
        description="Reproducible audit of registration-failure detector transfer.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config", help="strictly validate and hash a YAML config")
    validate.add_argument("--config", default="configs/pilot.yaml")
    validate.set_defaults(func=command_validate)

    env = sub.add_parser("environment", help="record the evaluation environment and code identity")
    env.add_argument("--config", default="configs/pilot.yaml")
    env.add_argument("--output", help="JSON output path (prints to stdout when omitted)")
    env.set_defaults(func=command_environment)

    prepare = sub.add_parser(
        "prepare-data", help="reconstruct local datasets from checksum-pinned official archives"
    )
    prepare.add_argument("--config", default="configs/pilot.yaml")
    prepare.add_argument("--dataset", default="all", choices=("all", "FIRE", "COph100"))
    prepare.add_argument("--download", action="store_true")
    prepare.add_argument("--acknowledge-fire-terms-unresolved", action="store_true")
    prepare.add_argument(
        "--reuse-existing",
        action="store_true",
        help="keep an already-extracted destination instead of refusing; the archives "
        "are still verified and audit-data still checksums every file",
    )
    prepare.set_defaults(func=command_prepare_data)

    audit = sub.add_parser("audit-data", help="audit local datasets and create identity manifests")
    audit.add_argument("--config", default="configs/pilot.yaml")
    audit.add_argument(
        "--subject-map",
        action="append",
        default=[],
        metavar="DATASET=CSV",
        help="reviewed pair-to-subject mapping; may be repeated",
    )
    audit.add_argument("--workers", type=int, default=1)
    audit.add_argument("--worker-threads", type=int, default=1)
    audit.set_defaults(func=command_audit_data)

    register = sub.add_parser("register", help="run a configured isolated registration adapter")
    register.add_argument("--config", default="configs/pilot.yaml")
    register.add_argument("--split", choices=("development", "confirmatory"), required=True)
    register.add_argument("--pipeline", required=True)
    register.add_argument(
        "--direction", choices=("canonical", "reverse", "both"), default="canonical"
    )
    register.add_argument("--max-attempts", type=int, default=3)
    register.add_argument("--pair", action="append", default=[], help="exact manifest pair ID")
    register.add_argument("--dataset", action="append", default=[], help="dataset ID filter")
    register.add_argument("--limit", type=int, help="stable maximum number of selected pairs")
    register.add_argument("--workers", type=int, default=1)
    register.add_argument("--worker-threads", type=int, default=1)
    register.set_defaults(func=command_register)

    e2 = sub.add_parser(
        "e2",
        aliases=["e2-smoke"],
        help="run coordinate-corrected input-perturbation reruns",
    )
    e2.add_argument("--config", default="configs/pilot.yaml")
    e2.add_argument("--pipeline", required=True)
    e2.add_argument(
        "--split",
        choices=("development", "confirmatory"),
        default="development",
        help="confirmatory perturbation requires a matching G1 freeze record",
    )
    e2.add_argument("--pair", action="append", default=[], help="exact manifest pair ID")
    e2.add_argument("--dataset", action="append", default=[], help="dataset ID filter")
    e2.add_argument("--limit", type=int, default=None, help="stable maximum selected pairs")
    e2.add_argument(
        "--sample-cap",
        type=int,
        default=0,
        help="seeded group-balanced cap on perturbed pairs per dataset (0 disables)",
    )
    e2.add_argument("--reruns", type=int, help="full perturbed reruns (default: configured B)")
    e2.add_argument("--translation-fraction", type=float, default=0.015)
    e2.add_argument("--max-attempts", type=int, default=3)
    e2.add_argument("--workers", type=int, default=1)
    e2.add_argument("--worker-threads", type=int, default=1)
    e2.set_defaults(func=command_e2)

    labels = sub.add_parser(
        "labels", help="score cached development registrations in the restricted evaluator"
    )
    labels.add_argument("--config", default="configs/pilot.yaml")
    labels.add_argument(
        "--split",
        choices=("development", "confirmatory"),
        required=True,
        help="confirmatory scoring requires a matching G1 freeze record",
    )
    labels.add_argument("--pipeline", required=True)
    labels.add_argument(
        "--direction",
        action="append",
        choices=("canonical", "reverse"),
        default=None,
        help="direction to score; may be repeated (default: both cached directions)",
    )
    labels.add_argument("--pair", action="append", default=[], help="exact manifest pair ID")
    labels.add_argument("--dataset", action="append", default=[], help="dataset ID filter")
    labels.add_argument("--workers", type=int, default=1)
    labels.add_argument("--worker-threads", type=int, default=1)
    labels.set_defaults(func=command_labels)

    overlays = sub.add_parser(
        "render-overlays", help="render ignored local panels for development geometry review"
    )
    overlays.add_argument("--config", default="configs/pilot.yaml")
    overlays.add_argument("--split", choices=("development",), required=True)
    overlays.add_argument("--pipeline", required=True)
    overlays.add_argument("--direction", choices=("canonical", "reverse"), default="canonical")
    overlays.add_argument("--pair", action="append", default=[], help="exact manifest pair ID")
    overlays.add_argument("--dataset", action="append", default=[], help="dataset ID filter")
    overlays.add_argument("--limit", type=int, default=20)
    overlays.add_argument("--columns", type=int, default=2)
    overlays.add_argument("--thumbnail-width", type=int, default=720)
    overlays.add_argument("--output-dir", default="reports/geometry_overlays")
    overlays.set_defaults(func=command_render_overlays)

    probe = sub.add_parser(
        "probe-adapter", help="smoke-test one matcher environment on a synthetic pair"
    )
    probe.add_argument("--config", default="configs/pilot.yaml")
    probe.add_argument("--pipeline", required=True)
    probe.add_argument("--output", help="write the probe evidence as JSON")
    probe.add_argument("--repeatability-tolerance-px", type=float, default=1e-4)
    probe.set_defaults(func=command_probe_adapter)

    features = sub.add_parser("features", help="extract ground-truth-free cached signal families")
    features.add_argument("--config", default="configs/pilot.yaml")
    features.add_argument("--split", choices=("development", "confirmatory"), required=True)
    features.add_argument("--families", nargs="+", default=[])
    features.add_argument(
        "--families-from-gate",
        action="store_true",
        help="extract exactly the families the development gate froze",
    )
    features.add_argument("--pipeline", action="append", default=[])
    features.add_argument("--dataset", action="append", default=[])
    features.add_argument("--pair", action="append", default=[])
    features.add_argument("--workers", type=int, default=1,
                          help="independent feature processes; parent alone writes the cache")
    features.add_argument("--worker-threads", type=int, default=0,
                          help="native threads per case worker; 0 keeps library defaults")
    features.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="publish accumulated feature rows every N jobs, on job boundaries",
    )
    features.add_argument(
        "--direction", action="append", choices=("canonical", "reverse"), default=[]
    )
    features.set_defaults(func=command_features)

    diagnose = sub.add_parser(
        "diagnose-development",
        help="classify cycle and E1 informativeness using development evidence only",
    )
    diagnose.add_argument("--config", default="configs/pilot.yaml")
    diagnose.set_defaults(func=command_diagnose_development)

    plan = sub.add_parser(
        "plan-study",
        help="screen both transfer directions for M2 information feasibility",
    )
    plan.add_argument("--config", default="configs/pilot.yaml")
    plan.set_defaults(func=command_plan_study)

    rekey = sub.add_parser(
        "rekey-cache",
        help="carry registration and feature caches across a cache-neutral config change",
    )
    rekey.add_argument("--config", default="configs/pilot.yaml")
    rekey.add_argument(
        "--previous-config",
        required=True,
        help="the configuration the cached features were computed under",
    )
    rekey.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be re-keyed without writing a shard",
    )
    rekey.set_defaults(func=command_rekey_cache)

    cost = sub.add_parser("estimate-cost", help="summarise measured cached runtimes")
    cost.add_argument("--config", default="configs/pilot.yaml")
    cost.set_defaults(func=command_estimate_cost)

    card = sub.add_parser(
        "evaluation-card",
        help="evaluate an external score CSV with ranking, probability, and acceptance metrics",
    )
    card.add_argument("--input", required=True, help="CSV with score,failure,group columns")
    card.add_argument("--output", required=True, help="destination JSON summary")
    card.add_argument(
        "--coverage",
        action="append",
        type=float,
        default=[],
        help="requested coverage; repeat as needed (defaults: 0.5, 0.7, 0.8, 0.9)",
    )
    card.set_defaults(func=command_evaluation_card)

    evaluate = sub.add_parser(
        "evaluate",
        help="fit, freeze, and test the joint claim on the G1-frozen direction",
    )
    evaluate.add_argument("--config", default="configs/pilot.yaml")
    evaluate.add_argument(
        "--refit-bootstrap",
        type=int,
        default=0,
        help="complete refit resamples (default: the configured value)",
    )
    evaluate.set_defaults(func=command_evaluate)

    plan_full = sub.add_parser(
        "plan-full-study",
        help="project external class-support information from development groups",
    )
    plan_full.add_argument("--config", default="configs/full_study.yaml")
    plan_full.set_defaults(func=command_plan_full_study)

    freeze_full = sub.add_parser(
        "freeze-full-study",
        help="freeze the factorized multi-domain study before external outcomes",
    )
    freeze_full.add_argument("--config", default="configs/full_study.yaml")
    freeze_full.add_argument("--signed-off-by", required=True)
    freeze_full.add_argument("--review-note", required=True)
    freeze_full.add_argument("--workers", type=int, default=1)
    freeze_full.add_argument("--worker-threads", type=int, default=1)
    freeze_full.set_defaults(func=command_freeze_full_study)

    evaluate_full = sub.add_parser(
        "evaluate-full-study",
        help="evaluate frozen baseline-versus-factorized models in all external cells",
    )
    evaluate_full.add_argument("--config", default="configs/full_study.yaml")
    evaluate_full.add_argument("--workers", type=int, default=1)
    evaluate_full.add_argument("--worker-threads", type=int, default=1)
    evaluate_full.set_defaults(func=command_evaluate_full_study)

    reproduce = sub.add_parser(
        "reproduce",
        help="rebuild manuscript outputs from frozen caches and verify they agree",
    )
    reproduce.add_argument("--config", default="configs/pilot.yaml")
    reproduce.add_argument("--refit-bootstrap", type=int, default=0)
    reproduce.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="absolute agreement required of each rebuilt estimand",
    )
    reproduce.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="build the outputs when no recorded results.json exists yet",
    )
    reproduce.set_defaults(func=command_reproduce)

    gallery = sub.add_parser(
        "failure-gallery",
        help="render the deterministic failure gallery into the ignored review directory",
    )
    gallery.add_argument("--config", default="configs/pilot.yaml")
    gallery.add_argument("--arm", default="transferred")
    gallery.add_argument("--per-class", type=int, default=4)
    gallery.add_argument("--columns", type=int, default=2)
    gallery.set_defaults(func=command_failure_gallery)

    inspect = sub.add_parser(
        "inspect-provenance",
        help="report what this machine's matcher environments actually resolved",
    )
    inspect.add_argument("--config", default="configs/pilot.yaml")
    inspect.add_argument("--pipeline", default="", help="default: every subprocess pipeline")
    inspect.add_argument("--output", default="", help="write the comparison as JSON")
    inspect.add_argument(
        "--write",
        action="store_true",
        help="record the observed pins in the config and append a protocol deviation; "
        "refuses when an upstream commit or checkpoint hash differs",
    )
    inspect.set_defaults(func=command_inspect_provenance)

    report = sub.add_parser(
        "report", help="build manuscript tables and figures from frozen caches"
    )
    report.add_argument("--config", default="configs/pilot.yaml")
    report.set_defaults(func=command_report)

    freeze = sub.add_parser(
        "freeze", help="write the reviewed G1 record that gates confirmatory outcomes"
    )
    freeze.add_argument("--config", default="configs/pilot.yaml")
    freeze.add_argument("--direction", default="", help="defaults to primary_direction")
    freeze.add_argument("--signed-off-by", default="", help="who reviewed this freeze")
    freeze.add_argument("--review-note", default="", help="what the review concluded")
    freeze.add_argument(
        "--acknowledge-infeasible",
        action="store_true",
        help="freeze a direction the M2 screen blocks, recording that decision",
    )
    freeze.set_defaults(func=command_freeze)

    study = sub.add_parser(
        "run-study",
        help="run every executable stage in dependency order, unattended",
    )
    study.add_argument("--config", default="configs/pilot.yaml")
    study.add_argument(
        "--download",
        action="store_true",
        help="allow dataset preparation to retrieve the checksum-pinned archives",
    )
    study.add_argument(
        "--acknowledge-fire-terms-unresolved",
        action="store_true",
        help="required for any local FIRE extraction; see DATA_LICENCES.md",
    )
    study.add_argument(
        "--development-only",
        action="store_true",
        help="omit the confirmatory registration and feature sweeps",
    )
    study.add_argument(
        "--e2-limit", type=int, default=1, help="development pairs given full E2 reruns"
    )
    study.add_argument(
        "--e2-sample-cap",
        type=int,
        default=0,
        help="seeded group-balanced E2 cap per dataset instead of --e2-limit",
    )
    study.add_argument(
        "--signed-off-by",
        default="",
        help="reviewer for the G1 freeze; without it the run stops at the M2 screen",
    )
    study.add_argument("--review-note", default="", help="what the G1 review concluded")
    study.add_argument("--feature-workers", type=int, default=1,
                       help="parallel case workers for feature stages, preserving all seeds")
    study.add_argument("--feature-worker-threads", type=int, default=0,
                       help="native threads per feature worker; 0 keeps library defaults")
    study.add_argument(
        "--acknowledge-infeasible",
        action="store_true",
        help="freeze a direction the M2 screen blocks, recording that decision",
    )
    study.add_argument(
        "--refit-bootstrap",
        type=int,
        default=0,
        help="complete refit resamples for the confirmatory claim",
    )
    study.add_argument("--only", nargs="+", default=[], help="run just these stage ids")
    study.add_argument("--skip", nargs="+", default=[], help="omit these stage ids")
    study.add_argument("--start-at", default="", help="begin at this stage id")
    study.add_argument(
        "--keep-going",
        action="store_true",
        help="continue past a failed stage, skipping only what depends on it",
    )
    study.add_argument(
        "--min-free-gb",
        type=float,
        default=20.0,
        help="free disk the preflight requires before any stage runs",
    )
    study.add_argument("--quiet", action="store_true", help="log stage output without echoing it")
    study.add_argument(
        "--dry-run", action="store_true", help="print the resolved plan and exit"
    )
    study.set_defaults(func=command_run_study)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except CommandError as exc:
        print(f"prerequisite error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("interrupted; completed atomic shards remain resumable", file=sys.stderr)
        return 130


__all__ = ["build_parser", "main"]
