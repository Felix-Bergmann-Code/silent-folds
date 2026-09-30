"""The number generator must read exactly what the experiment scripts write."""

from __future__ import annotations

import json
import re

import pandas as pd

from scripts import build_isbi_numbers as numbers


def _write(root, relative, rows):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _fake_results(root):
    auroc_row = dict(population="all_attempted", dataset="ALL", clearance_auroc=0.778,
                     inlier_ratio_auroc=0.71, inlier_count_auroc=0.70,
                     perspective_auroc=0.65, clearance_plus_inlier_ratio_auroc=0.80)
    _write(root, "clearance/clearance_auroc.csv",
           [{**auroc_row, "pipeline": p} for p in ("xfeat_h", "sp_lg_h", "sift_h")])
    _write(root, "clearance/nested_threshold_summary.csv",
           [dict(pipeline=p, target_precision=0.95, median_precision=0.96, median_tau=1.2,
                 median_recall=0.3)
            for p in ("xfeat_h", "sp_lg_h")])
    conditional = []
    for p in ("xfeat_h", "sp_lg_h"):
        for arm in ("non_stability", "non_stability_e1_e2"):
            for variant in ("pole_guard", "pole_guard_plus_clearance",
                            "log_curvature", "quantile_curvature"):
                conditional.append(dict(
                    pipeline=p, feature_arm=arm, variant=variant,
                    median_delta_raw_auroc_vs_guard=0.002, median_calibrated_auroc=0.82,
                    calibration_reversals=3))
    _write(root, "clearance/conditional_value_summary.csv", conditional)
    _write(root, "clearance/exact_vs_sampled_folding_summary.csv", [
        dict(pipeline_id="xfeat_h", exact_rectangle_crossing=True,
             sampled_detj_sign_change=False, cached_folding_positive=False, cases=1,
             failures=1)])
    _write(root, "clearance/geometry_validity_summary.csv", [
        dict(pipeline_id=p, inverse_only=1, inverse_only_failed=1,
             global_reflections_noncrossing=0,
             crossings_passing_projected_quad_heuristic=2) for p in ("xfeat_h", "sp_lg_h")])
    summary = []
    for p in ("xfeat_h", "sp_lg_h", "sift_h"):
        for v in ("none", "sample_orientation", "rectangle", "fov"):
            summary.append(dict(pipeline_id=p, dataset_id="ALL", variant=v,
                                rescued_failure_to_success=1, silent_to_explicit=2,
                                rectangle_crossings=3))
    _write(root, "constrained_ransac/constrained_ransac_summary.csv", summary)
    _write(root, "constrained_ransac/constrained_ransac_cases.csv", [
        dict(job_id="a", pipeline_id="xfeat_h", variant="none", rectangle_crossing=True,
             status="ok", operational_failure=True, matrix_json="[1]"),
        dict(job_id="a", pipeline_id="xfeat_h", variant="rectangle", status="degenerate_fit",
             rectangle_crossing=False, operational_failure=True, matrix_json=""),
        dict(job_id="b", pipeline_id="xfeat_h", variant="none", rectangle_crossing=False,
             status="ok", operational_failure=False, matrix_json="[3]"),
        dict(job_id="b", pipeline_id="xfeat_h", variant="rectangle", status="ok",
             rectangle_crossing=False, operational_failure=False, matrix_json="[4]"),
        dict(job_id="c", pipeline_id="sp_lg_h", variant="none", rectangle_crossing=False,
             status="ok", operational_failure=False, matrix_json="[5]"),
        dict(job_id="c", pipeline_id="sp_lg_h", variant="rectangle", status="ok",
             rectangle_crossing=False, operational_failure=False, matrix_json="[5]"),
        dict(job_id="d", pipeline_id="sift_h", variant="none", rectangle_crossing=False,
             status="invalid_transform", operational_failure=True, matrix_json=""),
        dict(job_id="d", pipeline_id="sift_h", variant="rectangle", status="ok",
             rectangle_crossing=False, operational_failure=True, matrix_json="[6]"),
    ])
    _write(root, "superretina/superretina_cases.csv", [
        dict(direction="canonical", returned=True, failure=True, rectangle_crossing=True,
             fov_crossing=False, clearance=0.0, official_failed=True, group_id="g1"),
        dict(direction="canonical", returned=True, failure=False, rectangle_crossing=False,
             fov_crossing=False, clearance=5.0, official_failed=False, group_id="g2"),
    ])
    _write(root, "superretina/superretina_summary.csv", [
        dict(dataset="ALL", clearance_auroc=0.9, inlier_ratio_auroc=0.8,
             perspective_auroc=0.7, clearance_plus_inlier_ratio_auroc=0.95)])
    _write(root, "superretina/superretina_refit_cases.csv", [
        dict(job_key="k", direction="canonical", variant="none", status="ok", failure=True,
             rectangle_crossing=True),
        dict(job_key="k", direction="canonical", variant="rectangle", status="degenerate_fit",
             failure=True, rectangle_crossing=False),
    ])
    (root / "superretina/fire_official_reproduction.json").write_text(json.dumps({
        "reproduced": {"S": 0.94, "P": 0.56, "A": 0.78},
        "released": {"S": 0.95, "P": 0.554, "A": 0.783}}))


def _fake_review(root):
    ci = []
    for p in ("xfeat_h", "sp_lg_h", "superretina"):
        for population in ("all_attempted", "returned_only"):
            for score in ("clearance", "inlier_ratio", "fused", "scale"):
                ci.append(dict(pipeline=p, population=population, score=score, cases=10,
                               auroc=0.8, ci_low=-0.012, ci_high=0.9))
    ci += [dict(pipeline="superretina", population="with_inliers", score=s, cases=7,
                auroc=0.7, ci_low=0.6, ci_high=0.8) for s in ("clearance", "inlier_ratio")]
    _write(root, "auroc_ci.csv", ci)
    diff = [dict(pipeline=p, population="all_attempted", comparison=c, ci_low=-0.01, ci_high=0.07)
            for p in ("xfeat_h", "sp_lg_h", "superretina")
            for c in ("fused-detector", "fused-inlier_ratio", "clearance-detector",
                      "clearance-perspective")]
    _write(root, "auroc_differences.csv", diff)
    _write(root, "threshold_sensitivity.csv", [
        dict(pipeline=p, threshold=t, clearance_auroc=0.9, inlier_ratio_auroc=0.8,
             fused_auroc=0.85, crossings=2, crossings_failed=2)
        for p in ("xfeat_h", "sp_lg_h", "superretina") for t in (0.0025, 0.005, 0.01, 0.02)])


def test_generator_fills_every_macro_the_paper_uses(tmp_path, monkeypatch):
    _fake_results(tmp_path)
    _fake_review(tmp_path / "review")
    monkeypatch.setattr(numbers, "REVIEW", tmp_path / "review")
    output = tmp_path / "numbers.tex"
    values = numbers.main(["--results", str(tmp_path), "--output", str(output)])
    tex = (numbers.ROOT / "paper/isbi_2027/main.tex").read_text()
    used = set(re.findall(r"\\providecommand\{\\(num[A-Za-z]+)\}", tex))
    assert used, "main.tex declares no \\num macros"
    assert used <= set(values), f"not generated: {sorted(used - set(values))}"
    assert values["numSrFireDiff"] == "0.010"
    assert values["numChangedNonPole"] == "1"
    assert values["numRansacSr"] == "0/1/0"
    assert values["numRansacXf"] == "0/1/0"
    assert values["numRansacSift"] == "0/0/1"
    assert values["numPoleExplicit"] == "1"
    assert values["numSrZeroInlier"] == "1"
    assert values["numXfFusedDetDiff"] == "[$-$0.01, 0.07]"
    assert values["numXfClrCI"] == "$-$.01--.90"
    assert values["numCrossFailLoose"] == "6/6"
    assert "\\newcommand{\\numXfInlAUROC}{.710}" in output.read_text()
