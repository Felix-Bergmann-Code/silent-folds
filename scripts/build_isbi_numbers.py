#!/usr/bin/env python3
"""Write ``paper/isbi_2027/numbers.tex`` from the workstation outputs.

Every value the manuscript still marks as pending is a ``\\num...`` macro.
``main.tex`` loads this file when present and otherwise falls back to red
``\\pend`` placeholders, so a missing output is visible rather than silent.
Values that fail to compute are omitted (and listed), never guessed.

Run after ``scripts/run_isbi_submission.ps1`` has populated
``reports/isbi_submission_latest`` (locally, after pulling the committed
results), then rebuild the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports/isbi_submission_latest"
OUTPUT = ROOT / "paper/isbi_2027/numbers.tex"
MODERN = ("xfeat_h", "sp_lg_h")
ARMS = ("non_stability", "non_stability_e1_e2")  # Base, +Stab


def auc(value: float) -> str:
    """Table style: .812"""
    return f"{value:.3f}".lstrip("0") if np.isfinite(value) else "--"


def pct(value: float) -> str:
    return f"{100 * value:.0f}\\%"


def signed(value: float) -> str:
    return f"{value:+.3f}"


def read(relative: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / relative)


def configure(results: Path, output: Path) -> None:
    global RESULTS, OUTPUT
    RESULTS, OUTPUT = Path(results), Path(output)


def clearance_numbers(out: dict[str, str]) -> None:
    table = read("clearance/clearance_auroc.csv")
    table = table[table.population.eq("all_attempted") & table.dataset.eq("ALL")]
    row = {p: table[table.pipeline.eq(p)].iloc[0] for p in MODERN}
    for key, pipeline in (("Xf", "xfeat_h"), ("Sp", "sp_lg_h")):
        r = row[pipeline]
        out[f"num{key}ClrAUROC"] = auc(r.clearance_auroc)
        out[f"num{key}InlAUROC"] = auc(r.inlier_ratio_auroc)
        out[f"num{key}CountAUROC"] = auc(r.inlier_count_auroc)
        out[f"num{key}PerspAUROC"] = auc(r.perspective_auroc)
        out[f"num{key}ComboAUROC"] = auc(r.clearance_plus_inlier_ratio_auroc)
    low, high = sorted(row[p].inlier_ratio_auroc for p in MODERN)
    out["numInlAUROCRange"] = f"{low:.2f}--{high:.2f}"


def nested_numbers(out: dict[str, str]) -> None:
    table = read("clearance/nested_threshold_summary.csv")
    table = table[np.isclose(table.target_precision, 0.95)].set_index("pipeline")
    out["numNestedPrecision"] = "/".join(pct(table.loc[p, "median_precision"]) for p in MODERN)
    out["numNestedTau"] = "/".join(f"{table.loc[p, 'median_tau']:.2f}" for p in MODERN)
    out["numNestedRecall"] = "/".join(pct(table.loc[p, "median_recall"]) for p in MODERN)


def conditional_numbers(out: dict[str, str]) -> None:
    table = read("clearance/conditional_value_summary.csv")
    base = table[table.feature_arm.eq("non_stability")]
    plus = base[base.variant.eq("pole_guard_plus_clearance")].set_index("pipeline")
    out["numCondDelta"] = "/".join(
        signed(plus.loc[p, "median_delta_raw_auroc_vs_guard"]) for p in MODERN)
    for variant, key in (("log_curvature", "Log"), ("quantile_curvature", "Quant")):
        cell = table[table.variant.eq(variant)].set_index(["pipeline", "feature_arm"])
        for pipeline, short in (("xfeat_h", "Xf"), ("sp_lg_h", "Sp")):
            out[f"num{key}Rev{short}"] = "/".join(
                str(int(cell.loc[(pipeline, arm), "calibration_reversals"])) for arm in ARMS)
        out[f"num{key}AucB"] = auc(cell.loc[("xfeat_h", ARMS[0]), "median_calibrated_auroc"])
        out[f"num{key}AucS"] = auc(cell.loc[("xfeat_h", ARMS[1]), "median_calibrated_auroc"])
        out[f"num{key}RevTotal"] = str(int(
            cell.loc[[(p, a) for p in MODERN for a in ARMS], "calibration_reversals"].sum()))


def geometry_numbers(out: dict[str, str]) -> None:
    folding = read("clearance/exact_vs_sampled_folding_summary.csv")
    modern = folding[folding.pipeline_id.isin(MODERN)]
    misses = modern[modern.exact_rectangle_crossing.astype(bool)
                    & ~modern.sampled_detj_sign_change.astype(bool)].cases.sum()
    out["numSampledMisses"] = str(int(misses))
    sift = folding[folding.pipeline_id.eq("sift_h")
                   & folding.exact_rectangle_crossing.astype(bool)]
    sift_missed = sift[~sift.sampled_detj_sign_change.astype(bool)].cases.sum()
    out["numSampledMissesSift"] = f"{int(sift_missed)}/{int(sift.cases.sum())}"
    validity = read("clearance/geometry_validity_summary.csv")
    v = validity[validity.pipeline_id.isin(MODERN)].sum(numeric_only=True)
    out["numInverseOnly"] = str(int(v.inverse_only))
    out["numInverseOnlyFailed"] = str(int(v.inverse_only_failed))
    out["numReflections"] = str(int(v.global_reflections_noncrossing))
    out["numQuadPasses"] = str(int(v.crossings_passing_projected_quad_heuristic))


def outcome(status: pd.Series, failure: pd.Series) -> pd.Series:
    """success, silent (a wrong transform is returned) or explicit (none is)."""
    returned = status.fillna("").eq("ok")
    failed = failure.fillna(True).astype(bool)
    return pd.Series(np.where(~returned, "explicit", np.where(failed, "silent", "success")),
                     index=status.index)


def transitions(before: pd.Series, after: pd.Series) -> str:
    """Constrained-RANSAC cell: failures made successes / silent made explicit /
    explicit made silent."""
    gained = (before.ne("success") & after.eq("success")).sum()
    to_explicit = (before.eq("silent") & after.eq("explicit")).sum()
    to_silent = (before.eq("explicit") & after.eq("silent")).sum()
    return f"{int(gained)}/{int(to_explicit)}/{int(to_silent)}"


def ransac_numbers(out: dict[str, str]) -> None:
    summary = read("constrained_ransac/constrained_ransac_summary.csv")
    summary = summary[summary.dataset_id.eq("ALL")].set_index(["pipeline_id", "variant"])
    out["numMoisanLeft"] = str(int(sum(
        summary.loc[(p, "sample_orientation"), "rectangle_crossings"] for p in MODERN)))
    cases = read("constrained_ransac/constrained_ransac_cases.csv")
    base = cases[cases.variant.eq("none")].set_index(["pipeline_id", "job_id"])
    rect = cases[cases.variant.eq("rectangle")].set_index(["pipeline_id", "job_id"])
    joined = rect.join(base, rsuffix="_base")
    joined["before"] = outcome(joined.status_base, joined.operational_failure_base)
    joined["after"] = outcome(joined.status, joined.operational_failure)
    for pipeline, short in (("xfeat_h", "Xf"), ("sp_lg_h", "Sp"), ("sift_h", "Sift")):
        part = joined.loc[pipeline]
        out[f"numRansac{short}"] = transitions(part.before, part.after)
    to_silent = joined.loc["sift_h"]
    out["numSiftToSilent"] = str(int(
        (to_silent.before.eq("explicit") & to_silent.after.eq("silent")).sum()))
    modern = joined[joined.index.get_level_values(0).isin(MODERN)]
    pole = modern.rectangle_crossing_base.fillna(False).astype(bool)
    out["numPoleRescued"] = str(int((pole & modern.after.eq("success")).sum()))
    out["numPoleExplicit"] = str(int((pole & modern.after.eq("explicit")).sum()))
    out["numPoleSilent"] = str(int((pole & modern.after.eq("silent")).sum()))
    changed = modern.matrix_json.fillna("") != modern.matrix_json_base.fillna("")
    out["numChangedNonPole"] = str(int((~pole & changed).sum()))
    gained = (modern.after.eq("success") & modern.before.ne("success")).sum()
    lost = (modern.after.ne("success") & modern.before.eq("success")).sum()
    out["numNetSuccess"] = f"{int(gained - lost):+d}"


def superretina_numbers(out: dict[str, str]) -> None:
    cases = read("superretina/superretina_cases.csv")
    cases = cases[cases.direction.eq("canonical")]
    returned = cases[cases.returned.astype(bool)]
    failed = returned.failure.astype(bool)
    rect = returned.rectangle_crossing.astype(bool)
    fov = returned.fov_crossing.astype(bool)
    near = returned.clearance.astype(float) <= 1.0
    out["numSrReturned"] = str(len(returned))
    out["numSrFailed"] = str(int(failed.sum()))
    out["numSrRect"] = f"{int((rect & failed).sum())}/{int(rect.sum())}"
    out["numSrFov"] = f"{int((fov & failed).sum())}/{int(fov.sum())}"
    out["numSrRho"] = f"{int((near & failed).sum())}/{int(near.sum())}"
    out["numSrRectCount"] = str(int(rect.sum()))
    summary = read("superretina/superretina_summary.csv").set_index("dataset")
    out["numSrClrAUROC"] = auc(summary.loc["ALL", "clearance_auroc"])
    out["numSrInlAUROC"] = auc(summary.loc["ALL", "inlier_ratio_auroc"])
    out["numSrPerspAUROC"] = auc(summary.loc["ALL", "perspective_auroc"])
    out["numSrZeroInlier"] = str(int((rect & returned.official_failed.astype(bool)).sum()))
    out["numSrPoleGroups"] = str(int(returned[rect].group_id.nunique()))
    out["numSrComboAUROC"] = auc(summary.loc["ALL", "clearance_plus_inlier_ratio_auroc"])
    refit = read("superretina/superretina_refit_cases.csv")
    refit = refit[refit.direction.eq("canonical")]
    none = refit[refit.variant.eq("none")].set_index("job_key")
    rect_fit = refit[refit.variant.eq("rectangle")].set_index("job_key")
    joined = rect_fit.join(none, rsuffix="_base")
    before = outcome(joined.status_base, joined.failure_base)
    after = outcome(joined.status, joined.failure)
    out["numRansacSr"] = transitions(before, after)
    out["numSrRefitPoles"] = str(int(none.rectangle_crossing.fillna(False).astype(bool).sum()))
    reproduction = json.loads((RESULTS / "superretina/fire_official_reproduction.json")
                              .read_text())
    got, want = reproduction["reproduced"], reproduction["released"]
    out["numSrFireDiff"] = f"{max(abs(got[k] - want[k]) for k in ('S', 'P', 'A')):.3f}"


REVIEW = ROOT / "reports/isbi_review_analyses"
SHORT = {"xfeat_h": "Xf", "sp_lg_h": "Sp", "superretina": "Sr"}


def interval(low: float, high: float) -> str:
    """Two-decimal interval with a proper minus sign, e.g. $-$0.01--0.07."""
    def fmt(value):
        text = f"{value:.2f}"
        return "$-$" + text[1:] if text.startswith("-") else text
    return f"{fmt(low)}--{fmt(high)}"


def review_numbers(out: dict[str, str]) -> None:
    ci = pd.read_csv(REVIEW / "auroc_ci.csv")
    everyone = ci[ci.population.eq("all_attempted")].set_index(["pipeline", "score"])
    for pipeline, short in SHORT.items():
        for score, key in (("clearance", "Clr"), ("inlier_ratio", "Inl"), ("fused", "Fused")):
            row = everyone.loc[(pipeline, score)]
            out[f"num{short}{key}CI"] = interval(row.ci_low, row.ci_high).replace("0.", ".")
    returned = ci[ci.population.eq("returned_only")].set_index(["pipeline", "score"])
    shift = max(abs(returned.loc[key, "auroc"] - everyone.loc[key, "auroc"])
                for key in returned.index
                if key in everyone.index and key[1] in ("clearance", "inlier_ratio", "fused"))
    out["numReturnedShift"] = f"{shift:.2f}"
    out["numSrRetClr"] = f"{returned.loc[('superretina', 'clearance'), 'auroc']:.3f}"
    out["numSrRetInl"] = f"{returned.loc[('superretina', 'inlier_ratio'), 'auroc']:.3f}"
    inliers = ci[ci.population.eq("with_inliers")].set_index("score")
    out["numSrWithInl"] = str(int(inliers.loc["clearance", "cases"]))
    out["numSrWithInlClr"] = f"{inliers.loc['clearance', 'auroc']:.3f}"
    out["numSrWithInlInl"] = f"{inliers.loc['inlier_ratio', 'auroc']:.3f}"
    scale = everyone.xs("scale", level="score").auroc
    out["numScaleAUROC"] = "/".join(auc(scale.loc[p]) for p in MODERN)
    # The fused score is cross-fitted in the review analyses (deployable on a
    # single case); its AUROC supersedes the whole-cohort rank fusion.
    for pipeline, short in SHORT.items():
        out[f"num{short}ComboAUROC"] = auc(everyone.loc[(pipeline, "fused"), "auroc"])
    condition = everyone.xs("condition", level="score").auroc
    out["numCondAUROCRange"] = f"{condition.min():.2f}--{condition.max():.2f}"
    far = ci[ci.population.eq("rho_gt_1")].set_index(["pipeline", "score"])
    for score, key in (("clearance", "Clr"), ("inlier_ratio", "Inl"), ("fused", "Fused")):
        out[f"numFar{key}"] = "/".join(auc(far.loc[(p, score), "auroc"]) for p in SHORT)
    out["numFarCases"] = "/".join(str(int(far.loc[(p, "clearance"), "cases"])) for p in SHORT)

    diff = pd.read_csv(REVIEW / "auroc_differences.csv")
    diff = diff[diff.population.eq("all_attempted")].set_index(["pipeline", "comparison"])
    for comparison, key in (("fused-detector", "FusedDet"), ("fused-inlier_ratio", "FusedInl"),
                            ("clearance-detector", "ClrDet")):
        for pipeline, short in SHORT.items():
            if (pipeline, comparison) in diff.index:
                row = diff.loc[(pipeline, comparison)]
                low, high = interval(row.ci_low, row.ci_high).split("--")
                out[f"num{short}{key}Diff"] = f"[{low}, {high}]"
    persp = diff.xs("clearance-perspective", level="comparison")
    out["numClrPerspBound"] = f"{max(persp.ci_low.abs().max(), persp.ci_high.abs().max()):.3f}"
    affine = diff.xs("clearance-affine", level="comparison")
    out["numClrAffineBound"] = f"{max(affine.ci_low.abs().max(), affine.ci_high.abs().max()):.3f}"
    # Non-inferiority margin: how far below the detector the fused score can be.
    det = diff.xs("fused-detector", level="comparison")
    out["numFusedDetMargin"] = f"{-det.ci_low.min():.2f}"

    crossings = pd.read_csv(REVIEW / "crossings_by_group.csv")
    generic = crossings[crossings.pipeline.isin(MODERN)]
    per_group = generic.groupby("group_id").crossings.sum()
    out["numGenericCross"] = str(int(per_group.sum()))
    out["numGenericCrossGroups"] = str(int(per_group.gt(0).sum()))
    out["numGenericCrossTop"] = str(int(per_group.max()))
    sr = crossings[crossings.pipeline.eq("superretina")]
    out["numSrCrossMax"] = str(int(sr.crossings.max()))
    if crossings[crossings.dataset.eq("FIRE")].crossings.sum():
        raise ValueError("the paper states that no FIRE pair crosses")
    cohort = crossings[crossings.dataset.eq("COph100")].groupby("pipeline")[
        ["pairs", "crossings"]].sum()
    modern = cohort.loc[list(MODERN)].sum()
    out["numCophGenericRate"] = f"{100 * modern.crossings / modern.pairs:.1f}\\%"
    out["numCophSrRate"] = pct(cohort.loc["superretina", "crossings"]
                               / cohort.loc["superretina", "pairs"])

    sens = pd.read_csv(REVIEW / "threshold_sensitivity.csv")
    for pipeline, short in SHORT.items():
        part = sens[sens.pipeline.eq(pipeline)].set_index("threshold")
        out[f"numSens{short}"] = "/".join(f"{part.loc[t, 'clearance_auroc']:.2f}"
                                          for t in (0.01, 0.02))
    for column, key in (("clearance_auroc", "Clr"), ("inlier_ratio_auroc", "Inl"),
                        ("fused_auroc", "Fused")):
        out[f"numSensMin{key}"] = f"{sens[column].min():.2f}"
    loose = sens[np.isclose(sens.threshold, 0.02)]
    mid = sens[np.isclose(sens.threshold, 0.01)]
    out["numCrossFailLoose"] = f"{int(loose.crossings_failed.sum())}/{int(loose.crossings.sum())}"
    out["numCrossFailMid"] = f"{int(mid.crossings_failed.sum())}/{int(mid.crossings.sum())}"


def main(argv=None) -> dict[str, str]:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    configure(args.results, args.output)
    values: dict[str, str] = {}
    failed: list[str] = []
    for step in (clearance_numbers, nested_numbers, conditional_numbers,
                 geometry_numbers, ransac_numbers, superretina_numbers, review_numbers):
        try:
            step(values)
        except Exception as exc:  # report and keep going; the paper shows \pend
            failed.append(f"{step.__name__}: {type(exc).__name__}: {exc}")
    lines = ["% Generated by scripts/build_isbi_numbers.py -- do not edit by hand."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(values.items())]
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(values)} values to {OUTPUT}")
    for item in failed:
        print("MISSING", item)
    return values


if __name__ == "__main__":
    main()
