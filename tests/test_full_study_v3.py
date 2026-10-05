from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from warpaudit.config import load_config
from warpaudit.data.loaders import DatasetUnavailable, list_indexed_pairs
from warpaudit.evaluation.full_study import evaluate_full_study
from warpaudit.protocols.full_freeze import (
    FULL_FREEZE_FILENAME,
    FullStudyFreeze,
    load_full_freeze,
)
from warpaudit.protocols.splits import make_outer_folds
from warpaudit.registration.fitting import FittingPolicy, fit_correspondences


def test_repository_full_study_config_is_valid() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "full_study.yaml")
    assert config.tier == "full"
    assert config.full_study.analysis_mode == "descriptive"
    assert not config.full_study.external_confirmatory_datasets
    assert set(config.full_study.external_descriptive_datasets) == {
        "MultiRegHistology", "MultiRegCytology"
    }
    assert {"E1", "E2"} <= set(config.full_study.augmented_families)


def test_reviewed_external_index_loader(tmp_path: Path) -> None:
    for name in ("moving.png", "fixed.png"):
        Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(tmp_path / name)
    (tmp_path / "points.txt").write_text("1 2 3 4\n5 6 7 8\n", encoding="utf-8")
    with (tmp_path / "warpaudit_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pair_id", "moving_path", "fixed_path", "annotation_path", "subject_id"],
        )
        writer.writeheader()
        writer.writerow({"pair_id": "p1", "moving_path": "moving.png", "fixed_path": "fixed.png",
                         "annotation_path": "points.txt", "subject_id": "patient-1"})
    rows = list_indexed_pairs(tmp_path, "AN200")
    assert len(rows) == 1
    assert rows[0].subject_id == "patient-1"
    assert rows[0].moving_hw == (12, 16)


def test_reviewed_external_index_cannot_escape_dataset_root(tmp_path: Path) -> None:
    with (tmp_path / "warpaudit_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pair_id", "moving_path", "fixed_path", "annotation_path", "subject_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "p1",
                "moving_path": "../outside.png",
                "fixed_path": "../outside.png",
                "annotation_path": "../outside.txt",
                "subject_id": "patient-1",
            }
        )
    with pytest.raises(DatasetUnavailable, match="escapes"):
        list_indexed_pairs(tmp_path, "AN200")


def _cases() -> tuple[pd.DataFrame, tuple[str, ...]]:
    rng = np.random.default_rng(4)
    records = []
    for dataset, count in (("DEV", 100), ("EXT", 40)):
        groups = [f"{dataset}/patient/{i:03d}" for i in range(count)]
        if dataset == "DEV":
            folds = make_outer_folds(groups, n_folds=5, seed=17)
            within = {fold: 0 for fold in range(5)}
            labels = []
            for group in groups:
                fold = folds[group]
                labels.append(within[fold] % 2)
                within[fold] += 1
        else:
            labels = [i % 2 for i in range(count)]
        for group, label in zip(groups, labels, strict=True):
            signal = float(label) + rng.normal(0, 0.08)
            records.append({
                "job_id": f"job-{group}", "dataset_id": dataset, "pair_id": group,
                "group_id": group, "pipeline_id": "pipe", "operational_failure": float(label),
                "silent_failure": bool(label), "explicit_failure": False,
                "bounded_loss": float(label), "tre_norm": float(label),
                "A:noise": rng.normal(), "E1:factor": signal, "E2:input": signal,
            })
    return pd.DataFrame(records), ("A:noise", "E1:factor", "E2:input")


def test_full_evaluator_compares_factorized_model_on_external_cells() -> None:
    cases, names = _cases()
    rows = evaluate_full_study(
        cases, names, development_datasets=("DEV",), external_datasets=("EXT",),
        confirmatory_datasets=("EXT",), pipelines=("pipe",), baseline_families=("A",),
        augmented_families=("A", "E1", "E2"), calibration_method="platt",
        nominal_acceptance=0.7, high_confidence_cutoff=0.2,
        min_class_bearing_groups=10, min_brier_improvement=0.0,
        bootstrap_resamples=50, C_values=(0.1, 1.0), seed=17,
    )
    assert len(rows) == 1 and rows[0]["eligible"]
    assert rows[0]["contrast"]["brier_improvement"] > 0
    assert rows[0]["augmented"]["auroc"] > rows[0]["baseline"]["auroc"]


def test_tps_correspondence_control_fits_nonrigid_mapping() -> None:
    x, y = np.meshgrid(np.linspace(0, 100, 5), np.linspace(0, 80, 4))
    source = np.column_stack((x.ravel(), y.ravel()))
    target = source + np.column_stack(
        (4 * np.sin(source[:, 1] / 30), 3 * np.sin(source[:, 0] / 25))
    )
    fit = fit_correspondences(
        source,
        target,
        FittingPolicy(threshold_px=10, min_matches=6, max_iters=200),
        seed=9,
        transform_family="tps",
        tps_regularisation=1e-4,
    )
    assert fit.ok and fit.transform is not None
    assert np.median(np.linalg.norm(fit.transform.apply(source) - target, axis=1)) < 0.1


def test_full_freeze_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    record = FullStudyFreeze(
        generated_at="now", config_hash="cfg", git_commit="git", code_identity="code",
        development_datasets=("DEV",), external_confirmatory_datasets=("EXT",),
        external_descriptive_datasets=(), pipelines=("p",), baseline_families=("A",),
        augmented_families=("A", "E1", "E2"), primary_metric="brier_improvement",
        min_brier_improvement=0.01, min_passing_cell_fraction=0.5,
        min_class_bearing_groups=10, information_plan_hash="plan", bootstrap_resamples=1000,
        high_confidence_cutoff=0.2, signed_off_by="reviewer", review_note="reviewed",
    )
    path = tmp_path / FULL_FREEZE_FILENAME
    path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    assert load_full_freeze(tmp_path).hash == record.hash
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["min_brier_improvement"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="edited"):
        load_full_freeze(tmp_path)
