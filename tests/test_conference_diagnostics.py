from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from scripts.conference_diagnostics import (
    audit_sift,
    availability,
    coverage_contrasts,
    select_gallery,
    support_warnings,
)
from scripts.conference_extension import clean_json
from warpaudit.cache.hashing import file_digest
from warpaudit.geometry.transforms import HomographyTransform
from warpaudit.registration.runner import serialise_transform


def test_two_success_groups_are_a_warning_even_with_two_classes():
    counts = {"failures": 93, "successes": 2, "failure_groups": 20, "success_groups": 2}
    warnings = support_warnings({"datasets": {}, "calibration": counts})
    assert warnings[0]["code"] == "sparse_class_support"
    assert warnings[0]["success_groups"] == 2


def test_availability_keeps_markers_distinct_from_unmeasured_signals():
    frame = pd.DataFrame(
        {"A:family_available": [np.nan, np.nan], "A:ncc": [np.nan, np.nan], "E1:spread": [0.0, 0.0]}
    )
    info = availability(frame, tuple(frame.columns))
    assert info["all_missing_measurements"] == ["A:ncc"]
    assert info["finite_counts"]["E1:spread"] == 2  # measured zero is not missing


def test_unequal_tied_coverage_does_not_produce_matched_risk_claim():
    a = [{"requested_coverage": 0.5, "attainable": True, "achieved_coverage": 0.4, "risk": 0.3}]
    b = [{"requested_coverage": 0.5, "attainable": True, "achieved_coverage": 0.5, "risk": 0.1}]
    assert coverage_contrasts(a, b)[0]["risk_improvement"] is None
    b[0]["achieved_coverage"] = 0.4
    assert np.isclose(coverage_contrasts(a, b)[0]["risk_improvement"], 0.2)
    b[0].update(attainable=False, achieved_coverage=None, risk=None)
    assert not coverage_contrasts(a, b)[0]["matched_achieved_coverage"]


def test_gallery_selection_is_stable_and_spans_groups():
    frame = pd.DataFrame(
        {
            "job_id": ["c", "b", "a", "d"],
            "dataset_id": ["DEV"] * 4,
            "group_id": ["1", "1", "2", "3"],
            "explicit_failure": [False] * 4,
            "operational_failure": [1] * 4,
        }
    )
    assert select_gallery(frame, 2).job_id.tolist() == ["a", "b"]
    assert select_gallery(frame.iloc[::-1], 2).job_id.tolist() == ["a", "b"]


def test_real_landmark_audit_and_gallery_detect_stale_labels_without_external_access(
    tmp_path, monkeypatch
):
    from warpaudit import cli

    # Unequal frame dimensions and resize/padding exercise original-to-working conversion.
    cfg = SimpleNamespace(
        full_study=SimpleNamespace(development_datasets=("FIRE",)),
        geometry=SimpleNamespace(
            working_long_edge=80,
            pad_to_square=True,
            pixel_center_convention="integer-centre",
            interpolation="bilinear",
            padding_mode="zeros",
        ),
        labels=SimpleNamespace(
            tau_primary=0.005,
            tau_secondary=(0.01,),
            bounded_loss_cap=10.0,
            r8_landmark_success_px=12.5,
        ),
    )
    Image.fromarray(np.full((40, 60, 3), 130, dtype=np.uint8)).save(tmp_path / "moving.png")
    Image.fromarray(np.full((80, 120, 3), 130, dtype=np.uint8)).save(tmp_path / "fixed.png")
    points = tmp_path / "points.txt"
    points.write_text("10 12 20 24\n20 25 40 50\n30 20 60 40\n")
    pair = pd.Series(
        {
            "pair_id": "FIRE/p",
            "dataset_id": "FIRE",
            "group_id": "g",
            "group_basis": "patient",
            "moving_image_id": "m",
            "fixed_image_id": "f",
            "moving_path": "moving.png",
            "fixed_path": "fixed.png",
            "moving_h": 40,
            "moving_w": 60,
            "fixed_h": 80,
            "fixed_w": 120,
            "annotation_paths": ["points.txt"],
            "annotation_sha256": [file_digest(points)],
            "annotation_kind": "landmarks",
            "annotation_provenance": "test fixture",
        }
    )
    pair_input = cli._pair_from_row(pair, cfg, tmp_path, direction="canonical")
    original = np.diag([2.0, 2.0, 1.0])
    working = pair_input.coordinates.A_f @ original @ np.linalg.inv(pair_input.coordinates.A_m)
    row = pd.Series(
        {
            "job_id": "sift-dev",
            "pair_id": "FIRE/p",
            "dataset_id": "FIRE",
            "pipeline_id": "sift_h",
            "group_id": "g",
            "direction": "canonical",
            "status": "ok",
            "transform_params": serialise_transform(HomographyTransform(working)),
            "explicit_failure": False,
        }
    )
    row = (
        pd.concat([row, pd.Series(cli._compute_label_task((row, pair, cfg, tmp_path)))])
        .groupby(level=0)
        .last()
    )
    dev = pd.DataFrame([row])
    external = dev.copy()
    external["dataset_id"] = "EXT"
    external["pair_id"] = "MUST_NOT_READ"
    monkeypatch.setattr(cli, "_pairs_manifest", lambda *args: (pd.DataFrame([pair]), None))
    output = tmp_path / "review"
    result = audit_sift(pd.concat([dev, external]), cfg, tmp_path, output)
    assert result["passed"] and result["audited_cases"] == 1
    assert result["label_checks"][0]["tre_px"] < 1e-10
    assert (output / "sift_review/contact_sheet.png").is_file()
    assert clean_json(result)["gallery"][0]["landmarks_predicted_working"]
    dev["bounded_loss"] = 8.0
    failed = audit_sift(dev, cfg, tmp_path, output)
    assert not failed["passed"]
    assert failed["label_checks"][0]["mismatches"] == ["bounded_loss"]


def test_review_evidence_must_match_development_and_images(tmp_path):
    import pytest

    from scripts.conference_extension import validate_development_review, write_json
    from warpaudit.cache.hashing import short_hash

    image = tmp_path / "example.png"
    image.write_bytes(b"test-image")
    review = {
        "identity": {"code": "fixture"},
        "development_hash": "cases-1",
        "sift": {
            "passed": True,
            "gallery": [{"file": "example.png", "sha256": file_digest(image)}],
        },
    }
    write_json(tmp_path / "development_review.json", {**review, "review_hash": short_hash(review)})
    assert validate_development_review(tmp_path, review["identity"], "cases-1") == short_hash(
        review
    )
    with pytest.raises(ValueError, match="stale or edited"):
        validate_development_review(tmp_path, review["identity"], "cases-2")
    image.write_bytes(b"modified")
    with pytest.raises(ValueError, match="image changed"):
        validate_development_review(tmp_path, review["identity"], "cases-1")
    review["sift"]["passed"] = False
    write_json(tmp_path / "development_review.json", {**review, "review_hash": short_hash(review)})
    with pytest.raises(ValueError, match="audit failed"):
        validate_development_review(tmp_path, review["identity"], "cases-1")
