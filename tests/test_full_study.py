"""The whole study, end to end, on synthetic datasets.

This is the regression test behind the claim that a single unattended run
completes: it drives every stage the runner schedules, in order, through the
real CLI -- audit, register, score, extract, diagnose, screen, freeze, score the
reserve, evaluate, and report -- and checks the artefacts each one is supposed
to leave behind.

The data is synthetic so the test is fast, but nothing about the machinery is
stubbed: real manifests, real caches, real splits, real frozen policies, and
the real intersection-union decision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from warpaudit.cache import ShardedTable
from warpaudit.cli import main
from warpaudit.data.loaders import PairListing, register_loader
from warpaudit.geometry.transforms import HomographyTransform
from warpaudit.registration.adapters import register_adapter
from warpaudit.types import RegistrationResult, RegistrationStatus

SOURCE, TARGET = "SRCSET", "TGTSET"
SOURCE_GROUPS, TARGET_GROUPS = 40, 46
PAIRS_PER_GROUP = 2
IMAGE_HW = (48, 64)


def _seed_of(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _write_dataset(root: Path, dataset_id: str, n_groups: int) -> None:
    """A FIRE-shaped fixture: one image per examination, one control-point file."""
    images = root / "Images"
    annotations = root / "Ground Truth"
    images.mkdir(parents=True, exist_ok=True)
    annotations.mkdir(parents=True, exist_ok=True)
    for group in range(n_groups):
        for index in range(PAIRS_PER_GROUP + 1):
            rng = np.random.default_rng(_seed_of(f"{dataset_id}/{group}/{index}"))
            pixels = rng.integers(0, 255, size=(*IMAGE_HW, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(images / f"S{group:03d}_{index + 1}.jpg", quality=95)
    for group in range(n_groups):
        for pair in range(PAIRS_PER_GROUP):
            points = np.array(
                [[5, 5], [50, 6], [8, 40], [52, 41], [30, 22], [12, 30], [45, 15]],
                dtype=float,
            )
            lines = "\n".join(
                f"{x:.1f} {y:.1f} {x:.1f} {y:.1f}" for x, y in points
            )
            (
                annotations / f"control_points_S{group:03d}_{pair + 1}_{pair + 2}.txt"
            ).write_text(lines + "\n", encoding="utf-8")


def _listing(dataset_id: str, root: Path, version: str) -> list[PairListing]:
    images = Path(root) / "Images"
    annotations = Path(root) / "Ground Truth"
    out: list[PairListing] = []
    for path in sorted(annotations.glob("control_points_*.txt")):
        stem = path.stem.removeprefix("control_points_")
        subject, left, right = stem.rsplit("_", 2)
        out.append(
            PairListing(
                dataset_id=dataset_id,
                pair_id=f"{dataset_id}/{subject}_{left}_{right}",
                moving_image_id=f"{dataset_id}/{subject}_{left}",
                fixed_image_id=f"{dataset_id}/{subject}_{right}",
                moving_path=images / f"{subject}_{left}.jpg",
                fixed_path=images / f"{subject}_{right}.jpg",
                moving_hw=IMAGE_HW,
                fixed_hw=IMAGE_HW,
                annotation_paths=(path,),
                annotation_kind="landmarks",
                annotation_provenance=f"{dataset_id} fixture control points",
                subject_id=subject,
                subject_basis="patient",
                subject_evidence="fixture subject id in filename",
            )
        )
    return out


class _Registrar:
    """A matcher whose error depends deterministically on the pair and pipeline.

    Both classes have to occur in every role for the evaluation to be defined,
    and the composite has to be able to beat chance, so the injected error is a
    reproducible function of the case rather than noise.
    """

    def __init__(self, pipeline_id: str, difficulty: float) -> None:
        self.pipeline_id = pipeline_id
        self.difficulty = difficulty

    def register(self, pair, seed):
        rng = np.random.default_rng(_seed_of(f"{self.pipeline_id}/{pair.pair_id}") ^ seed)
        severity = rng.uniform()
        shift = 0.0 if severity > self.difficulty else rng.uniform(2.0, 9.0)
        scale = pair.coordinates.moving.to_working[0, 0]
        points = np.array(
            [[5, 5], [50, 6], [8, 40], [52, 41], [30, 22], [12, 30], [45, 15]], dtype=float
        )
        working = (points + 0.5) * scale - 0.5
        moved = working + np.array([shift, shift * 0.5])
        # Match quality tracks the injected error, so family B carries signal.
        scores = np.clip(1.0 - shift / 12.0, 0.05, 1.0) * np.ones(len(points))
        return RegistrationResult(
            self.pipeline_id,
            RegistrationStatus.OK,
            HomographyTransform(
                np.array([[1.0, 0.0, shift], [0.0, 1.0, shift * 0.5], [0.0, 0.0, 1.0]])
            ),
            working,
            moved,
            np.ones(len(points), bool),
            scores,
            diagnostics={"retained_capacity": 16},
        )


CONFIG = f"""
project: fixture
tier: pilot
direction_priority: ["{SOURCE}->{TARGET}", "{TARGET}->{SOURCE}"]
geometry: {{working_long_edge: 64, grid_size: 6}}
features:
  families: [A, B, C, D, E1, F, G]
  bootstrap_B: 4
  perturbation_B: 2
  perturbation_B_sensitivity: [2]
splits: {{n_outer_folds: 5, min_outer_folds: 3, low_information_group_threshold: 10}}
planning: {{n_simulations: 400, min_class_bearing_groups: 2, min_accepted_groups: 2}}
datasets:
  - {{id: {SOURCE}, version: fixture, root: data/{SOURCE}, group_basis: patient,
      patient_ids_available: true}}
  - {{id: {TARGET}, version: fixture, root: data/{TARGET}, group_basis: patient,
      patient_ids_available: true}}
pipelines:
  - {{id: fx1, version: fixture-fx1, matcher: fx1, checkpoint: local, max_iters: 20}}
  - {{id: fx2, version: fixture-fx2, matcher: fx2, checkpoint: local, max_iters: 20}}
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    config = tmp_path / "configs" / "pilot.yaml"
    config.parent.mkdir()
    config.write_text(CONFIG, encoding="utf-8")
    _write_dataset(tmp_path / "data" / SOURCE, SOURCE, SOURCE_GROUPS)
    _write_dataset(tmp_path / "data" / TARGET, TARGET, TARGET_GROUPS)
    register_loader(SOURCE, lambda root, version: _listing(SOURCE, root, version))
    register_loader(TARGET, lambda root, version: _listing(TARGET, root, version))
    register_adapter("fx1", lambda pipeline: _Registrar("fx1", 0.55))
    register_adapter("fx2", lambda pipeline: _Registrar("fx2", 0.40))
    return config


def _run(*argv: str) -> None:
    assert main(list(argv)) == 0, f"command failed: {' '.join(argv)}"


@pytest.mark.slow
def test_the_whole_study_runs_in_one_pass(project: Path) -> None:
    config, root = str(project), project.parents[1]
    _run("audit-data", "--config", config)

    for pipeline in ("fx1", "fx2"):
        for split in ("development", "confirmatory"):
            _run("register", "--config", config, "--split", split,
                 "--pipeline", pipeline, "--direction", "both")
        _run("labels", "--config", config, "--split", "development",
             "--pipeline", pipeline, "--direction", "canonical")
    _run("features", "--config", config, "--split", "development",
         "--families", "A", "B", "C", "D", "E1", "F", "G", "--direction", "canonical")
    _run("diagnose-development", "--config", config)
    _run("plan-study", "--config", config)

    feasibility = json.loads((root / "manifests" / "m2_feasibility.json").read_text())
    assert feasibility["recommended_direction"], "no direction survived the screen"

    # Confirmatory outcomes are unreachable until the gate is passed.
    assert main(["labels", "--config", config, "--split", "confirmatory",
                 "--pipeline", "fx1"]) != 0

    _run("freeze", "--config", config, "--signed-off-by", "tester",
         "--review-note", "fixture run")
    freeze = json.loads((root / "manifests" / "g1_freeze.json").read_text())
    assert freeze["primary_direction"] == feasibility["recommended_direction"]

    for pipeline in ("fx1", "fx2"):
        _run("labels", "--config", config, "--split", "confirmatory",
             "--pipeline", pipeline, "--direction", "canonical")
    _run("features", "--config", config, "--split", "confirmatory",
         "--families", "A", "B", "C", "D", "F", "G", "--direction", "canonical")
    _run("evaluate", "--config", config, "--refit-bootstrap", "8")
    _run("report", "--config", config)

    results = json.loads((root / "manifests" / "results.json").read_text())
    aggregate = results["aggregate"]
    assert results["n_folds"] >= 3
    # The transferred detector must beat chance on a fixture where the signal
    # is real; otherwise the fold plumbing, not the statistics, is wrong.
    assert aggregate["target_auroc_transferred"] > 0.55
    assert np.isfinite(aggregate["gap_auc"])
    assert np.isfinite(aggregate["delta_policy"])
    assert 0.0 < aggregate["realised_coverage_source_on_target"] <= 1.0

    components = results["joint_claim"]
    assert len(components) == 3
    for row in components:
        assert np.isfinite(row["one_sided_bound"]) and isinstance(row["passed"], bool)
    assert results["conclusion"]
    # The primary table is the one the paper leads with; it must not be empty.
    assert (root / "paper" / "tables" / "primary_joint_claim.csv").is_file()
    outputs = json.loads((root / "reports" / "manuscript_outputs.json").read_text())
    assert "primary_joint_claim" not in outputs["empty_tables"]

    predictions = ShardedTable(root / "cache", "predictions", key_column="prediction_id").load()
    assert set(predictions["arm"]) == {"transferred", "target_reference"}
    # Both arms score exactly the same untouched target test cases.
    per_arm = predictions.groupby("arm")["job_id"].apply(set)
    assert per_arm["transferred"] == per_arm["target_reference"]

    tables = root / "paper" / "tables"
    for name in ("data_provenance", "primary_joint_claim", "policy_operating_points",
                 "signal_family_comparison", "per_pipeline_cells"):
        assert (tables / f"{name}.csv").is_file(), f"missing table {name}"
    assert (root / "reports" / "RESULTS.md").is_file()
    assert (root / "reports" / "manuscript_outputs.json").is_file()


@pytest.mark.slow
def test_a_rebuilt_run_reproduces_its_recorded_estimands(project: Path) -> None:
    config, root = str(project), project.parents[1]
    _run("audit-data", "--config", config)
    for pipeline in ("fx1", "fx2"):
        for split in ("development", "confirmatory"):
            _run("register", "--config", config, "--split", split,
                 "--pipeline", pipeline, "--direction", "both")
        _run("labels", "--config", config, "--split", "development",
             "--pipeline", pipeline, "--direction", "canonical")
    _run("features", "--config", config, "--split", "development",
         "--families", "A", "B", "C", "E1", "--direction", "canonical")
    _run("diagnose-development", "--config", config)
    _run("plan-study", "--config", config)
    _run("freeze", "--config", config, "--signed-off-by", "tester", "--review-note", "n")
    for pipeline in ("fx1", "fx2"):
        _run("labels", "--config", config, "--split", "confirmatory",
             "--pipeline", pipeline, "--direction", "canonical")
    _run("features", "--config", config, "--split", "confirmatory",
         "--families", "A", "B", "--direction", "canonical")
    _run("evaluate", "--config", config, "--refit-bootstrap", "4")
    # A rebuild from the same caches must land on the same numbers.
    _run("reproduce", "--config", config, "--refit-bootstrap", "4")
    payload = json.loads((root / "reports" / "REPRODUCTION.json").read_text())
    assert payload["reproduced"] is True
    assert payload["estimand_drift"] == []
