#!/usr/bin/env python3
"""Final ISBI submission experiments over the frozen full-study caches.

Every subcommand reads the current registration, label, and feature caches on
the workstation that produced ``reports/pole_guard_ablation_latest``.  None
launches a matcher.  New code deliberately lives outside ``warpaudit/`` and
``scripts/matchers/`` so that the registration and feature cache identities
(``cli._registration_code_identity``) do not change.

Subcommands
-----------
``clearance``
    Exp. A.  Normalized pole clearance as a training-free QC score: pooled,
    group-weighted AUROC with a group bootstrap; single-feature baselines;
    a nested (out-of-sample) high-precision clearance threshold over the 21
    group assignments; the conditional value of clearance when added to the
    learned detector; and exact versus lattice-sampled folding.

``constrained-ransac``
    Exp. B.  Re-fits every cached homography from its cached correspondences
    with the identical seeded RANSAC/DLT, then with hypothesis-level validity
    constraints (sample-point orientation, as in Moisan et al.; rectangle; and
    circular FOV) and a post-hoc rejection rule.  All variants are relabelled
    with the study's own landmark-error code.

``all``
    Runs both.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

# One BLAS/OpenMP thread per process: parallelism comes from the process pool.
# Spawned workers inherit these before they import numpy.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

HOMOGRAPHY_PIPELINES = ("xfeat_h", "sp_lg_h", "sift_h")
PRIMARY_PIPELINES = ("xfeat_h", "sp_lg_h")
CLEARANCE_FLOOR = 1e-3
CLEARANCE_CEILING = 1e3
N_ASSIGNMENTS = 21


# ---------------------------------------------------------------------------
# Pure geometry (unit-tested in tests/test_isbi_submission_experiments.py)
# ---------------------------------------------------------------------------


def clearance_score(clearance: np.ndarray | pd.Series, explicit_failure=None) -> np.ndarray:
    """Training-free failure score ``-log10(rho)``, floored and capped.

    A crossing (``rho = 0``) receives the largest score, an affine map
    (``rho = inf``) the smallest.  Explicit no-output cases receive the
    crossing score because any QC system flags a missing transform.
    """

    rho = np.asarray(pd.to_numeric(pd.Series(clearance), errors="coerce"), dtype=float)
    rho = np.where(np.isnan(rho), CLEARANCE_FLOOR, rho)
    score = -np.log10(np.clip(rho, CLEARANCE_FLOOR, CLEARANCE_CEILING))
    if explicit_failure is not None:
        score = np.where(np.asarray(explicit_failure, dtype=bool), -np.log10(CLEARANCE_FLOOR), score)
    return score


def rectangle_crossing(h: np.ndarray, width: float, height: float) -> bool:
    corners = np.array(
        [[0.0, 0.0, 1.0], [width - 1.0, 0.0, 1.0], [0.0, height - 1.0, 1.0],
         [width - 1.0, height - 1.0, 1.0]]
    )
    den = corners @ np.asarray(h, dtype=float)[2]
    return bool(den.min() <= 0.0 <= den.max())


def circle_crossing(h: np.ndarray, width: float, height: float) -> bool:
    a, b, c = np.asarray(h, dtype=float)[2]
    cx, cy = (width - 1.0) / 2.0, (height - 1.0) / 2.0
    radius = min(width - 1.0, height - 1.0) / 2.0
    norm = float(np.hypot(a, b))
    if norm == 0.0:
        return bool(c == 0.0)
    return bool(abs(a * cx + b * cy + c) <= radius * norm)


def sample_orientation_consistent(h: np.ndarray, sample_src: np.ndarray) -> bool:
    """Moisan-style check: the four sample points lie on one side of the pole."""

    den = np.c_[sample_src, np.ones(len(sample_src))] @ np.asarray(h, dtype=float)[2]
    return bool(np.all(den > 0) or np.all(den < 0))


def hypothesis_valid(
    h: np.ndarray, constraint: str, *, width: float, height: float, sample_src=None
) -> bool:
    if constraint == "none":
        return True
    if constraint == "sample_orientation":
        return sample_orientation_consistent(h, sample_src)
    if constraint == "rectangle":
        return not rectangle_crossing(h, width, height)
    if constraint == "fov":
        return not circle_crossing(h, width, height)
    raise ValueError(f"unknown constraint {constraint!r}")


def fit_homography_constrained(
    src: np.ndarray,
    dst: np.ndarray,
    policy,
    *,
    seed: int,
    constraint: str,
    width: float,
    height: float,
) -> dict:
    """Seeded RANSAC/DLT identical to ``fitting.fit_homography`` plus a validity test.

    The random stream is identical to the production fitter: every iteration
    draws the same minimal sample whether or not the resulting hypothesis is
    admissible.  Inadmissible hypotheses are skipped (they never become the
    incumbent and never shorten the adaptive iteration budget).  If the
    consensus refit violates the constraint, the admissible minimal
    hypothesis that produced the consensus is returned instead.  If no
    admissible hypothesis with four inliers exists, the fit is an explicit
    failure.  ``constraint='none'`` reproduces the cached production fit.
    """

    from warpaudit.geometry.transforms import HomographyTransform
    from warpaudit.registration.fitting import (
        MIN_CORRESPONDENCES,
        _symmetric_residuals,
        dlt_homography,
    )

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    result = {
        "status": "ok",
        "matrix": None,
        "n_iterations": 0,
        "rejected_hypotheses": 0,
        "refit_fallback": False,
        "n_inliers": 0,
    }
    if n == 0 or n < policy.min_matches:
        return {**result, "status": "no_matches"}
    if len(np.unique(src, axis=0)) < MIN_CORRESPONDENCES:
        return {**result, "status": "degenerate_fit"}
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    best_h = None
    iterations = 0
    rejected = 0
    max_iters = int(policy.max_iters)
    while iterations < max_iters:
        iterations += 1
        sample = rng.choice(n, size=MIN_CORRESPONDENCES, replace=False)
        h, _ = dlt_homography(src[sample], dst[sample])
        if h is None:
            continue
        if not hypothesis_valid(
            h, constraint, width=width, height=height, sample_src=src[sample]
        ):
            rejected += 1
            continue
        inliers = _symmetric_residuals(h, src, dst) <= policy.threshold_px
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers, best_h = count, inliers, h
            w = max(count / n, 1e-9)
            denom = np.log1p(-min(w**MIN_CORRESPONDENCES, 1 - 1e-12))
            if denom < 0:
                needed = np.log1p(-policy.confidence) / denom
                max_iters = int(min(max_iters, max(1, np.ceil(needed))))
    result.update(n_iterations=iterations, rejected_hypotheses=rejected, n_inliers=best_count)
    if best_count < MIN_CORRESPONDENCES:
        return {**result, "status": "degenerate_fit"}
    fit_src, fit_dst = (
        (src[best_inliers], dst[best_inliers]) if policy.refine_on_inliers else (src, dst)
    )
    h, _ = dlt_homography(fit_src, fit_dst)
    if h is None:
        return {**result, "status": "degenerate_fit"}
    if constraint not in ("none", "sample_orientation") and not hypothesis_valid(
        h, constraint, width=width, height=height
    ):
        h = best_h
        result["refit_fallback"] = True
    if not HomographyTransform(h).is_valid():
        return {**result, "status": "invalid_transform"}
    return {**result, "matrix": np.asarray(h, dtype=float).tolist()}


def perspective_score(h: np.ndarray, width: float, height: float) -> float:
    """Support-agnostic perspective strength: L * ||a|| / |d(center)|.

    The inverse of the pole's distance from the image *center* in diagonals.
    Unlike the clearance it ignores the image extent, so it measures only how
    projective H is.  Affine maps score 0.
    """

    hh = np.asarray(h, dtype=float)
    a = hh[2, :2]
    center = np.array([(width - 1.0) / 2.0, (height - 1.0) / 2.0])
    diagonal = float(np.hypot(width - 1.0, height - 1.0))
    denominator = abs(float(a @ center + hh[2, 2]))
    norm = float(np.hypot(*a))
    if norm == 0.0:
        return 0.0
    return float(np.inf) if denominator == 0.0 else diagonal * norm / denominator


def orientation_reversing(h: np.ndarray, width: float, height: float) -> bool:
    """No pole crossing, but det J = det(H)/d^3 < 0 on the whole rectangle."""

    if rectangle_crossing(h, width, height):
        return False
    hh = np.asarray(h, dtype=float)
    # d has one sign on the rectangle; evaluate it at the corner (0, 0).
    return bool(np.sign(np.linalg.det(hh)) * np.sign(hh[2, 2]) < 0)


def projected_quad_consistent(h: np.ndarray, width: float, height: float) -> bool:
    """Common heuristic: the four projected corners form a convex quad with the
    source orientation.  It uses only mapped points, not the sign of d."""

    corners = np.array([[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0],
                        [0.0, height - 1.0]])
    mapped = np.c_[corners, np.ones(4)] @ np.asarray(h, dtype=float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        points = mapped[:, :2] / mapped[:, 2:]
    if not np.isfinite(points).all():
        return False

    def turns(q):
        edges = np.roll(q, -1, axis=0) - q
        nxt = np.roll(edges, -1, axis=0)
        return edges[:, 0] * nxt[:, 1] - edges[:, 1] * nxt[:, 0]

    source, target = np.sign(turns(corners)), np.sign(turns(points))
    return bool(np.all(target == source[0]))


def rank_average(*scores) -> np.ndarray:
    """Training-free score fusion: mean of per-score normalized ranks."""

    from scipy.stats import rankdata

    ranks = [rankdata(np.asarray(x, dtype=float)) / len(x) for x in scores]
    return np.mean(ranks, axis=0)


def nested_threshold(
    rho: np.ndarray, failure: np.ndarray, *, target_precision: float, min_flagged: int
) -> float:
    """Largest clearance cutoff whose training precision meets the target.

    Candidates are the observed clearances.  Returns 0.0 (flag crossings only)
    when no positive cutoff qualifies.
    """

    rho = np.asarray(rho, dtype=float)
    failure = np.asarray(failure, dtype=bool)
    finite = np.isfinite(rho)
    order = np.argsort(rho[finite], kind="stable")
    values, labels = rho[finite][order], failure[finite][order]
    if values.size == 0:
        return 0.0
    cumulative_failures = np.cumsum(labels)
    counts = np.arange(1, len(values) + 1)
    # Evaluate each distinct value at its last occurrence so ties are flagged together.
    last = np.r_[values[1:] != values[:-1], True]
    precision = cumulative_failures / counts
    ok = last & (precision >= target_precision) & (counts >= min_flagged) & (values > 0)
    return float(values[ok].max()) if ok.any() else 0.0


def sampled_folding(h: np.ndarray, width: float, height: float, size: int = 32) -> bool:
    """Sign change of det J = det(H)/d^3 on the production pixel-centre lattice."""

    xs = (np.arange(size, dtype=float) + 0.5) * (width / size) - 0.5
    ys = (np.arange(size, dtype=float) + 0.5) * (height / size) - 0.5
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    hh = np.asarray(h, dtype=float)
    den = hh[2, 0] * xx + hh[2, 1] * yy + hh[2, 2]
    det_j = np.linalg.det(hh) / den**3
    finite = np.isfinite(det_j)
    return bool((det_j[finite] > 0).any() and (det_j[finite] < 0).any())


# ---------------------------------------------------------------------------
# Shared case loading
# ---------------------------------------------------------------------------


def load_cases():
    from scripts.conference_extension import assemble, select_arms
    from scripts.pole_guard_ablation import add_pole_guard_features
    from warpaudit.config import load_config

    cfg = load_config(ROOT / "configs/full_study.yaml")
    design = json.loads((ROOT / "configs/paper_study.json").read_text())
    spec = json.loads((ROOT / "configs/conference_extension.json").read_text())
    cases, names = assemble(cfg, ROOT, cfg.full_study.development_datasets)
    selected = select_arms(names, spec)
    cases, guarded_name, indicator_name = add_pole_guard_features(cases, names)
    source_name = guarded_name.removesuffix("_pole_guarded")
    cases["D:log_pole_clearance"] = np.log10(
        np.clip(
            pd.to_numeric(cases.pole_clearance_diagonal_fraction, errors="coerce"),
            CLEARANCE_FLOOR,
            CLEARANCE_CEILING,
        )
    )
    cases["clearance_score"] = clearance_score(
        cases.pole_clearance_diagonal_fraction, cases.status.astype(str).ne("ok")
    )
    cases[source_name + "_log"] = np.log10(
        1.0 + pd.to_numeric(cases[source_name], errors="coerce").clip(lower=0.0)
    )
    cases = add_geometry_columns(cases)
    return cfg, design, cases, names, selected, source_name, guarded_name, indicator_name


def add_geometry_columns(cases: pd.DataFrame) -> pd.DataFrame:
    """Per-transform validity checks and pre-specified training-free baselines."""

    columns = {k: [] for k in (
        "perspective_score", "inverse_pole_crosses_fixed", "orientation_reversing",
        "projected_quad_consistent", "rectangle_crossing_check")}
    for record in cases.to_dict("records"):
        ok = (str(record.get("status")) == "ok"
              and str(record.get("transform_family")) == "homography")
        if not ok:
            for values in columns.values():
                values.append(np.nan)
            continue
        payload = record["transform_params"]
        payload = json.loads(payload) if isinstance(payload, str) else payload
        h = np.asarray(payload["matrix"], dtype=float)
        mh, mw = map(int, record["working_moving_hw"])
        fh, fw = map(int, record["working_fixed_hw"])
        columns["perspective_score"].append(perspective_score(h, mw, mh))
        columns["inverse_pole_crosses_fixed"].append(
            rectangle_crossing(np.linalg.inv(h), fw, fh))
        columns["orientation_reversing"].append(orientation_reversing(h, mw, mh))
        columns["projected_quad_consistent"].append(projected_quad_consistent(h, mw, mh))
        columns["rectangle_crossing_check"].append(rectangle_crossing(h, mw, mh))
    out = cases.copy()
    for name, values in columns.items():
        out[name] = values
    returned = out.status.astype(str).eq("ok")
    matches = pd.to_numeric(out.get("n_matches"), errors="coerce")
    inliers = pd.to_numeric(out.get("n_inliers"), errors="coerce")
    # Pre-specified orientation: fewer inliers / lower ratio / stronger
    # perspective => failure.  No-output cases get the most-failing score.
    out["score_inlier_count"] = np.where(returned, -inliers.fillna(0), 1.0)
    out["score_inlier_ratio"] = np.where(
        returned, -(inliers / matches.replace(0, np.nan)).fillna(0), 1.0)
    out["score_perspective"] = np.where(
        returned, np.log10(pd.to_numeric(out.perspective_score, errors="coerce")
                           .clip(1e-6, 1e6)), 6.0)
    return out


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    from scripts import development_decision as dd

    dd.atomic_write_text(path, frame.to_csv(index=False))


# ---------------------------------------------------------------------------
# Exp. A: clearance as a QC score
# ---------------------------------------------------------------------------


def group_bootstrap_auroc(frame, score_columns, *, resamples, seed):
    """Group (patient/component) bootstrap for AUROCs and paired differences."""

    from warpaudit.evaluation.metrics import auroc
    from warpaudit.evaluation.weights import inverse_group_size_weights

    rng = np.random.default_rng(seed)
    groups = frame.group_id.astype(str).to_numpy()
    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    draws = {name: [] for name in score_columns}
    for _ in range(resamples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in chosen])
        # Relabel duplicated groups so each draw keeps equal total group weight.
        labels = np.concatenate(
            [np.full(len(index_by_group[g]), f"{g}#{k}") for k, g in enumerate(chosen)]
        )
        part = frame.iloc[idx]
        y = part.operational_failure.to_numpy(bool)
        if y.all() or not y.any():
            continue
        w = inverse_group_size_weights(pd.Series(labels))
        for name in score_columns:
            draws[name].append(auroc(part[name].to_numpy(float), y, w))
    return {name: np.asarray(values) for name, values in draws.items()}


HEAVY_COLUMNS = (
    "matches_moving", "matches_fixed", "inlier_mask", "match_scores", "diagnostics",
    "transform_params", "original_homography_json",
)
_STATE: dict | None = None


def _init_state(state: dict) -> None:
    global _STATE
    _STATE = state


def _parallel_map(function, tasks, state, workers):
    """Ordered map over a process pool that receives ``state`` once per worker."""

    if workers <= 1:
        _init_state(state)
        yield from map(function, tasks)
        return
    with ProcessPoolExecutor(
        max_workers=min(workers, len(tasks)), initializer=_init_state, initargs=(state,)
    ) as pool:
        yield from pool.map(function, tasks, chunksize=1)


def training_quantile(reference: np.ndarray, values: pd.Series) -> np.ndarray:
    """Empirical CDF of the training values; outside the range maps to 0 or 1."""

    reference = np.sort(np.asarray(reference, dtype=float))
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    out = np.searchsorted(reference, numeric, side="right") / max(len(reference), 1)
    return np.where(np.isnan(numeric), np.nan, out)


def add_quantile_feature(train, calibration, test, source_name):
    reference = pd.to_numeric(train[source_name], errors="coerce").dropna().to_numpy()
    name = source_name + "_quantile"
    return tuple(
        frame.assign(**{name: training_quantile(reference, frame[source_name])})
        for frame in (train, calibration, test)
    )


def _clearance_cell(task):
    from threadpoolctl import threadpool_limits

    from scripts import development_decision as dd
    from scripts import paper_study as ps
    from scripts import publication_audit as audit
    from scripts.pole_guard_ablation import guarded_columns

    seed, pipeline, fold = task
    state = _STATE
    cases, design = state["cases"], state["design"]
    source_name, guarded_name = state["source_name"], state["guarded_name"]
    local = copy.deepcopy(state["cfg"])
    local.splits.seed = seed
    train, calibration, test = dd.split_cell(
        cases, local, design, "within_pipeline", pipeline, fold
    )
    train, calibration, test = add_quantile_feature(train, calibration, test, source_name)
    development = pd.concat([train, calibration])
    returned = development[development.status.astype(str).eq("ok")]
    threshold_rows, metric_rows, prediction_rows = [], [], []
    for target in (0.90, 0.95, 1.0):
        tau = nested_threshold(
            returned.pole_clearance_diagonal_fraction.to_numpy(float),
            returned.operational_failure.to_numpy(bool),
            target_precision=target,
            min_flagged=5,
        )
        rho = pd.to_numeric(test.pole_clearance_diagonal_fraction, errors="coerce")
        flagged = test.status.astype(str).eq("ok") & (rho <= tau)
        threshold_rows.append({
            "seed": seed, "pipeline": pipeline, "fold": fold,
            "target_precision": target, "tau": tau,
            "test_cases": len(test),
            "test_failures": int(test.operational_failure.sum()),
            "test_flagged": int(flagged.sum()),
            "test_flagged_failures": int(test.operational_failure[flagged].sum()),
            "test_flagged_groups": int(test.group_id[flagged].nunique()),
            "test_crossings": int(test.pole_crosses_image.sum()),
        })
    for arm, base_columns in state["feature_arms"].items():
        guarded = guarded_columns(base_columns, source_name, guarded_name,
                                  state["indicator_name"])
        variants = {
            "pole_guard": guarded,
            "pole_guard_plus_clearance": (*guarded, "D:log_pole_clearance"),
            "no_curvature_plus_clearance": (
                *[c for c in base_columns if c != source_name],
                "D:log_pole_clearance",
            ),
            "clearance_only": ("D:log_pole_clearance",),
            "log_curvature": tuple(
                source_name + "_log" if c == source_name else c for c in base_columns),
            "quantile_curvature": tuple(
                source_name + "_quantile" if c == source_name else c for c in base_columns),
        }
        for variant, columns in variants.items():
            if variant == "clearance_only" and arm != "non_stability":
                continue
            with threadpool_limits(limits=1):
                model, _, _ = ps.fit_arm(train, calibration, columns, local, variant)
                diagnostics = audit.model_diagnostics(model, test, columns)
                raw = model.decision_function(test.loc[:, columns])
                probability = model.predict_probability(test.loc[:, columns])
            metric_rows.append({
                "seed": seed, "pipeline": pipeline, "fold": fold,
                "feature_arm": arm, "variant": variant,
                "ranking_reversed_by_calibration": diagnostics[
                    "ranking_reversed_by_calibration"],
            })
            prediction_rows.append(pd.DataFrame({
                "seed": seed, "pipeline": pipeline, "fold": fold,
                "feature_arm": arm, "variant": variant,
                "job_id": test.job_id.to_numpy(),
                "group_id": test.group_id.astype(str).to_numpy(),
                "operational_failure": test.operational_failure.to_numpy(bool),
                "raw_score": raw, "probability": probability,
            }))
    return threshold_rows, metric_rows, prediction_rows


def run_clearance(output: Path, *, resamples: int, workers: int) -> None:

    from warpaudit.evaluation.metrics import auroc, brier_score
    from warpaudit.evaluation.weights import inverse_group_size_weights

    output.mkdir(parents=True, exist_ok=True)
    cfg, design, cases, names, selected, source_name, guarded_name, indicator_name = (
        load_cases()
    )
    started = time.time()

    # A0. Per-case export for any later analysis.
    keep = [
        "job_id", "pair_id", "group_id", "dataset_id", "pipeline_id", "status",
        "operational_failure", "pole_crosses_image", "pole_crosses_inscribed_circle",
        "pole_clearance_diagonal_fraction", "clearance_score", source_name,
        "D:folding_fraction",
    ]
    keep = [c for c in keep if c in cases.columns]
    write_csv(output / "clearance_cases.csv", cases[keep])

    # A1. Training-free discrimination versus single-feature baselines.
    baseline_features = [
        n for n in names
        if any(key in n for key in ("inlier", "match_count", "folding_fraction",
                                    "log_jacobian_determinant", "bending_energy",
                                    "residual", "condition"))
    ]
    rows = []
    for pipeline in HOMOGRAPHY_PIPELINES:
        for dataset in ("ALL", "FIRE", "COph100"):
            part = cases[cases.pipeline_id.astype(str).eq(pipeline)]
            if dataset != "ALL":
                part = part[part.dataset_id.astype(str).eq(dataset)]
            for population in ("all_attempted", "returned_only"):
                p = part if population == "all_attempted" else part[part.status.astype(str).eq("ok")]
                y = p.operational_failure.to_numpy(bool)
                if p.empty or y.all() or not y.any():
                    continue
                w = inverse_group_size_weights(p.group_id.astype(str))
                row = {
                    "pipeline": pipeline, "dataset": dataset, "population": population,
                    "cases": len(p), "failures": int(y.sum()), "groups": p.group_id.nunique(),
                    "clearance_auroc": auroc(p.clearance_score.to_numpy(float), y, w),
                    "inlier_count_auroc": auroc(p.score_inlier_count.to_numpy(float), y, w),
                    "inlier_ratio_auroc": auroc(p.score_inlier_ratio.to_numpy(float), y, w),
                    "perspective_auroc": auroc(p.score_perspective.to_numpy(float), y, w),
                    # Training-free combination: mean of the two normalized ranks.
                    "clearance_plus_inlier_ratio_auroc": auroc(
                        rank_average(p.clearance_score, p.score_inlier_ratio), y, w),
                }
                for name in baseline_features:
                    values = pd.to_numeric(p[name], errors="coerce")
                    filled = values.fillna(values.median() if values.notna().any() else 0.0)
                    a = auroc(filled.to_numpy(float), y, w)
                    # Orientation chosen post hoc: optimistic for the baseline.
                    row[f"baseline_auroc_posthoc_sign:{name}"] = max(a, 1.0 - a)
                rows.append(row)
    write_csv(output / "clearance_auroc.csv", pd.DataFrame(rows))

    # A1b. Group bootstrap CI on the primary pooled comparison, original seed.
    ablation = ROOT / "reports/pole_guard_ablation_latest/test_predictions.csv"
    boot_rows = []
    if ablation.exists():
        predictions = pd.read_csv(ablation)
        original_seed = cfg.splits.seed
        for pipeline in PRIMARY_PIPELINES:
            for arm in sorted(predictions.feature_arm.unique()):
                pred = predictions[
                    (predictions.seed == original_seed)
                    & (predictions.pipeline == pipeline)
                    & (predictions.feature_arm == arm)
                    & (predictions.treatment == "pole_guard")
                ][["job_id", "raw_score", "probability"]]
                part = cases.merge(pred, on="job_id", validate="one_to_one")
                if part.empty:
                    continue
                draws = group_bootstrap_auroc(
                    part, ["clearance_score", "raw_score"], resamples=resamples,
                    seed=original_seed,
                )
                y = part.operational_failure.to_numpy(bool)
                w = inverse_group_size_weights(part.group_id.astype(str))
                diff = draws["clearance_score"] - draws["raw_score"]
                boot_rows.append({
                    "pipeline": pipeline, "feature_arm": arm, "cases": len(part),
                    "clearance_auroc": auroc(part.clearance_score, y, w),
                    "detector_raw_auroc": auroc(part.raw_score, y, w),
                    "clearance_ci_low": np.quantile(draws["clearance_score"], 0.025),
                    "clearance_ci_high": np.quantile(draws["clearance_score"], 0.975),
                    "difference_ci_low": np.quantile(diff, 0.025),
                    "difference_ci_high": np.quantile(diff, 0.975),
                    "resamples_used": len(diff),
                })
    write_csv(output / "clearance_vs_detector_bootstrap.csv", pd.DataFrame(boot_rows))

    # A2 + A3 over 21 assignments: 210 independent (seed, pipeline, fold) cells.
    feature_arms = {n: selected[n] for n in ("non_stability", "non_stability_e1_e2")}
    original_seed = cfg.splits.seed
    heavy = [c for c in HEAVY_COLUMNS if c in cases.columns]
    state = {
        "cases": cases.drop(columns=heavy), "cfg": cfg, "design": design,
        "feature_arms": feature_arms, "source_name": source_name,
        "guarded_name": guarded_name, "indicator_name": indicator_name,
    }
    tasks = [
        (seed, pipeline, fold)
        for seed in range(original_seed, original_seed + N_ASSIGNMENTS)
        for pipeline in PRIMARY_PIPELINES
        for fold in range(5)
    ]
    threshold_rows, metric_rows, prediction_rows = [], [], []
    for index, (thresholds_part, metrics_part, predictions_part) in enumerate(
        _parallel_map(_clearance_cell, tasks, state, workers), start=1
    ):
        threshold_rows.extend(thresholds_part)
        metric_rows.extend(metrics_part)
        prediction_rows.extend(predictions_part)
        if index % 20 == 0 or index == len(tasks):
            print(f"[clearance] {index}/{len(tasks)} cells ({time.time() - started:.0f}s)",
                  flush=True)

    thresholds = pd.DataFrame(threshold_rows)
    write_csv(output / "nested_threshold_folds.csv", thresholds)
    pooled = (
        thresholds.groupby(["seed", "pipeline", "target_precision"], as_index=False)
        .agg(flagged=("test_flagged", "sum"), flagged_failures=("test_flagged_failures", "sum"),
             failures=("test_failures", "sum"), crossings=("test_crossings", "sum"),
             tau_median=("tau", "median"), tau_min=("tau", "min"), tau_max=("tau", "max"))
    )
    pooled["precision"] = pooled.flagged_failures / pooled.flagged.replace(0, np.nan)
    pooled["recall"] = pooled.flagged_failures / pooled.failures
    write_csv(output / "nested_threshold_assignments.csv", pooled)
    summary = (
        pooled.groupby(["pipeline", "target_precision"], as_index=False)
        .agg(assignments=("seed", "nunique"),
             median_flagged=("flagged", "median"),
             median_precision=("precision", "median"),
             min_precision=("precision", "min"),
             median_recall=("recall", "median"),
             median_tau=("tau_median", "median"),
             min_tau=("tau_min", "min"), max_tau=("tau_max", "max"),
             total_flagged=("flagged", "sum"),
             total_flagged_failures=("flagged_failures", "sum"))
    )
    write_csv(output / "nested_threshold_summary.csv", summary)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    write_csv(output / "conditional_value_predictions.csv", predictions)
    assignment_rows = []
    for keys, part in predictions.groupby(["seed", "pipeline", "feature_arm", "variant"]):
        w = inverse_group_size_weights(part.group_id.astype(str))
        assignment_rows.append({
            "seed": keys[0], "pipeline": keys[1], "feature_arm": keys[2], "variant": keys[3],
            "pooled_raw_auroc": auroc(part.raw_score, part.operational_failure, w),
            "pooled_calibrated_auroc": auroc(part.probability, part.operational_failure, w),
            "pooled_brier": brier_score(part.probability, part.operational_failure, w),
        })
    assignments = pd.DataFrame(assignment_rows)
    reversals = (
        pd.DataFrame(metric_rows)
        .groupby(["pipeline", "feature_arm", "variant"], as_index=False)
        .agg(calibration_reversals=("ranking_reversed_by_calibration", "sum"))
    )
    reference = assignments[assignments.variant.eq("pole_guard")].set_index(
        ["seed", "pipeline", "feature_arm"])
    paired = []
    for keys, part in assignments.groupby(["pipeline", "feature_arm", "variant"]):
        joined = part.set_index(["seed", "pipeline", "feature_arm"]).join(
            reference, rsuffix="_ref", how="inner")
        d_auc = joined.pooled_raw_auroc - joined.pooled_raw_auroc_ref
        d_cal = joined.pooled_calibrated_auroc - joined.pooled_calibrated_auroc_ref
        d_brier = joined.pooled_brier - joined.pooled_brier_ref
        paired.append({
            "pipeline": keys[0], "feature_arm": keys[1], "variant": keys[2],
            "assignments": len(joined),
            "median_raw_auroc": part.pooled_raw_auroc.median(),
            "median_calibrated_auroc": part.pooled_calibrated_auroc.median(),
            "median_brier": part.pooled_brier.median(),
            "median_delta_raw_auroc_vs_guard": d_auc.median(),
            "median_delta_calibrated_auroc_vs_guard": d_cal.median(),
            "median_delta_brier_vs_guard": d_brier.median(),
            "assignments_raw_auroc_improved": int((d_auc > 0).sum()),
            "assignments_brier_improved": int((d_brier < 0).sum()),
        })
    conditional = pd.DataFrame(paired).merge(
        reversals, on=["pipeline", "feature_arm", "variant"], how="left")
    write_csv(output / "conditional_value_assignments.csv", assignments)
    write_csv(output / "conditional_value_summary.csv", conditional)

    # A4. Exact versus lattice-sampled folding.
    fold_rows = []
    for record in cases[cases.status.astype(str).eq("ok")
                        & cases.transform_family.astype(str).eq("homography")].to_dict("records"):
        payload = record["transform_params"]
        payload = json.loads(payload) if isinstance(payload, str) else payload
        h = np.asarray(payload["matrix"], dtype=float)
        height, width = map(int, record["working_moving_hw"])
        fold_rows.append({
            "job_id": record["job_id"], "pipeline_id": record["pipeline_id"],
            "dataset_id": record["dataset_id"],
            "operational_failure": bool(record["operational_failure"]),
            "exact_rectangle_crossing": bool(record["pole_crosses_image"]),
            "exact_fov_crossing": bool(record["pole_crosses_inscribed_circle"]),
            "sampled_detj_sign_change": sampled_folding(h, width, height),
            "cached_folding_fraction": record.get("D:folding_fraction", np.nan),
        })
    folding = pd.DataFrame(fold_rows)
    write_csv(output / "exact_vs_sampled_folding.csv", folding)
    crosstab = (
        folding.assign(cached_folding_positive=pd.to_numeric(
            folding.cached_folding_fraction, errors="coerce").fillna(0) > 0)
        .groupby(["pipeline_id", "exact_rectangle_crossing", "sampled_detj_sign_change",
                  "cached_folding_positive"], as_index=False)
        .agg(cases=("job_id", "size"), failures=("operational_failure", "sum"))
    )
    write_csv(output / "exact_vs_sampled_folding_summary.csv", crosstab)
    returned_cases = cases[cases.status.astype(str).eq("ok")
                           & cases.transform_family.astype(str).eq("homography")]
    validity = []
    for pipeline, part in returned_cases.groupby("pipeline_id"):
        crossing = part.pole_crosses_image.astype(bool)
        inverse = part.inverse_pole_crosses_fixed.astype(bool)
        reflect = part.orientation_reversing.astype(bool)
        quad_ok = part.projected_quad_consistent.astype(bool)
        failed = part.operational_failure.astype(bool)
        validity.append({
            "pipeline_id": pipeline, "returned": len(part),
            "forward_crossings": int(crossing.sum()),
            "inverse_crossings_fixed_image": int(inverse.sum()),
            "either_direction": int((crossing | inverse).sum()),
            "either_direction_failed": int(((crossing | inverse) & failed).sum()),
            "inverse_only": int((inverse & ~crossing).sum()),
            "inverse_only_failed": int((inverse & ~crossing & failed).sum()),
            "global_reflections_noncrossing": int(reflect.sum()),
            "global_reflections_failed": int((reflect & failed).sum()),
            "crossings_passing_projected_quad_heuristic": int((crossing & quad_ok).sum()),
            "noncrossings_failing_projected_quad_heuristic": int((~crossing & ~quad_ok).sum()),
            "corner_test_consistent_with_add_pole_guard": bool(
                (part.rectangle_crossing_check.astype(bool) == crossing).all()),
        })
    write_csv(output / "geometry_validity_summary.csv", pd.DataFrame(validity))
    (output / "run_metadata.json").write_text(json.dumps({
        "experiment": "clearance", "bootstrap_resamples": resamples,
        "assignments": N_ASSIGNMENTS, "original_seed": original_seed,
        "clearance_floor": CLEARANCE_FLOOR, "clearance_ceiling": CLEARANCE_CEILING,
        "seconds": time.time() - started,
    }, indent=2) + "\n")
    print(f"[clearance] done in {time.time() - started:.0f}s -> {output}", flush=True)


# ---------------------------------------------------------------------------
# Exp. B: support-constrained RANSAC
# ---------------------------------------------------------------------------

VARIANTS = ("none", "sample_orientation", "rectangle", "fov")


def _points(value) -> np.ndarray:
    """Decode cached correspondences exactly as the CLI's cache replay does."""

    from warpaudit.cli import _optional_array

    array = _optional_array(value, points=True)
    if array is None or len(array) == 0:
        return np.empty((0, 2), dtype=float)
    return np.asarray(array, dtype=float).reshape(-1, 2)


def _refit_one(record: dict) -> list[dict]:
    from warpaudit.cli import _compute_label_task

    state = _STATE
    cfg, root, pair_rows, policies = state["cfg"], state["root"], state["pairs"], state["policies"]
    policy = policies[record["pipeline_id"]]
    height, width = map(int, record["working_moving_hw"])
    moving = record.get("matches_moving")
    fixed = record.get("matches_fixed")
    src = _points(moving)
    dst = _points(fixed)
    cached = record.get("transform_params")
    cached = json.loads(cached) if isinstance(cached, str) else cached
    out = []
    for variant in VARIANTS:
        start = time.perf_counter()
        fit = fit_homography_constrained(
            src, dst, policy, seed=int(record["seed"]), constraint=variant,
            width=width, height=height,
        )
        elapsed = time.perf_counter() - start
        row = dict(record)
        row["status"] = fit["status"]
        row["transform_params"] = (
            None if fit["matrix"] is None
            else json.dumps({"family": "homography", "matrix": fit["matrix"]})
        )
        label = _compute_label_task((pd.Series(row), pair_rows[record["pair_id"]], cfg, root))
        matrix = None if fit["matrix"] is None else np.asarray(fit["matrix"])
        reproduces = None
        if variant == "none":
            if cached is None or matrix is None:
                reproduces = (cached is None) == (matrix is None)
            else:
                reproduces = bool(np.allclose(matrix, np.asarray(cached["matrix"]),
                                              rtol=1e-9, atol=1e-12))
        out.append({
            "job_id": record["job_id"], "pair_id": record["pair_id"],
            "group_id": record["group_id"], "dataset_id": record["dataset_id"],
            "pipeline_id": record["pipeline_id"], "variant": variant,
            "status": fit["status"], "n_matches": len(src), "n_inliers": fit["n_inliers"],
            "n_iterations": fit["n_iterations"],
            "rejected_hypotheses": fit["rejected_hypotheses"],
            "refit_fallback": fit["refit_fallback"],
            "rectangle_crossing": None if matrix is None else rectangle_crossing(matrix, width, height),
            "fov_crossing": None if matrix is None else circle_crossing(matrix, width, height),
            "tre_px": label["tre_px"], "tre_norm": label["tre_norm"],
            "operational_failure": bool(label["operational_failure"]),
            "reproduces_cache": reproduces, "seconds": elapsed,
            "matrix_json": None if matrix is None else json.dumps(matrix.tolist()),
        })
    return out


def run_constrained_ransac(
    output: Path, *, workers: int, limit: int | None, allow_stale_cache: bool = False
) -> None:
    from warpaudit.cache.store import ShardedTable
    from warpaudit.cli import _current_registration_rows, _pairs_manifest
    from warpaudit.config import load_config
    from warpaudit.registration.fitting import FittingPolicy

    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs/full_study.yaml")
    pairs, _ = _pairs_manifest(cfg, ROOT)
    cache_root = cfg.paths.resolve(ROOT)["cache_root"]
    registrations = ShardedTable(cache_root, "registrations").load()
    if not allow_stale_cache:
        registrations = _current_registration_rows(registrations, pairs, cfg, ROOT)
    registrations = registrations[
        registrations.pipeline_id.astype(str).isin(HOMOGRAPHY_PIPELINES)
        & registrations.dataset_id.astype(str).isin(cfg.full_study.development_datasets)
        & registrations.direction.astype(str).eq("canonical")
        & registrations.condition.astype(str).eq("clean")
    ].drop_duplicates("job_id", keep="last").sort_values(["pipeline_id", "pair_id"])
    if registrations.empty:
        raise SystemExit(
            "no current canonical clean homography registrations; run on the workstation "
            "that holds the full-study cache"
        )
    if limit:
        registrations = registrations.groupby("pipeline_id").head(limit)
    policies = {}
    for pipeline_id in HOMOGRAPHY_PIPELINES:
        p = cfg.pipeline(pipeline_id)
        policies[pipeline_id] = FittingPolicy(
            threshold_px=p.ransac_threshold_px, max_iters=p.max_iters,
            confidence=p.confidence, min_matches=p.min_matches, name=p.fit,
        )
    state = {
        "cfg": cfg, "root": ROOT, "policies": policies,
        "pairs": {str(r["pair_id"]): r for _, r in pairs.iterrows()},
    }
    # Low-consensus fits run the full RANSAC budget; dispatch them first so the
    # pool never waits on a slow straggler at the end.
    ratio = pd.to_numeric(registrations.n_inliers, errors="coerce") / pd.to_numeric(
        registrations.n_matches, errors="coerce").replace(0, np.nan)
    registrations = registrations.assign(_cost=-ratio.fillna(0.0)).sort_values(
        "_cost", ascending=False, kind="stable").drop(columns="_cost")
    keep = [c for c in registrations.columns
            if c not in ("match_scores", "diagnostics", "inlier_mask")]
    records = registrations[keep].to_dict("records")
    started = time.time()
    rows: list[dict] = []
    for index, result in enumerate(_parallel_map(_refit_one, records, state, workers), start=1):
        rows.extend(result)
        if index % 100 == 0 or index == len(records):
            print(f"[ransac] {index}/{len(records)} ({time.time() - started:.0f}s)", flush=True)
    frame = pd.DataFrame(rows).sort_values(["pipeline_id", "pair_id", "variant"], kind="stable")
    write_csv(output / "constrained_ransac_cases.csv", frame)

    base = frame[frame.variant.eq("none")].set_index("job_id")
    summary_rows = []
    scoped = pd.concat([frame, frame.assign(dataset_id="ALL")], ignore_index=True)
    for (pipeline, dataset, variant), part in scoped.groupby(
        ["pipeline_id", "dataset_id", "variant"]
    ):
        joined = part.set_index("job_id").join(base, rsuffix="_base")
        ok = joined.status.eq("ok")
        base_ok = joined.status_base.eq("ok")
        success = ~joined.operational_failure.astype(bool)
        base_success = ~joined.operational_failure_base.astype(bool)
        changed = joined.matrix_json.fillna("") != joined.matrix_json_base.fillna("")
        summary_rows.append({
            "pipeline_id": pipeline, "dataset_id": dataset, "variant": variant,
            "cases": len(joined),
            "returned": int(ok.sum()),
            "explicit_failures": int((~ok).sum()),
            "silent_failures": int((ok & ~success).sum()),
            "successes": int(success.sum()),
            "rectangle_crossings": int(joined.rectangle_crossing.fillna(False).astype(bool).sum()),
            "fov_crossings": int(joined.fov_crossing.fillna(False).astype(bool).sum()),
            "rescued_failure_to_success": int((success & ~base_success).sum()),
            "harmed_success_to_failure": int((~success & base_success).sum()),
            "silent_to_explicit": int((~ok & base_ok & ~base_success).sum()),
            "changed_transform_cases": int(changed.sum()),
            "refit_fallbacks": int(joined.refit_fallback.sum()),
            "median_rejected_hypotheses": float(joined.rejected_hypotheses.median()),
            "reproduces_cache_all": (
                bool(joined.reproduces_cache.fillna(True).astype(bool).all())
                if variant == "none" else None
            ),
            "median_seconds": float(joined.seconds.median()),
        })
    summary = pd.DataFrame(summary_rows)
    write_csv(output / "constrained_ransac_summary.csv", summary)
    # Focus table: baseline pole cases only.
    pole_ids = base.index[base.rectangle_crossing.fillna(False).astype(bool)]
    focus = frame[frame.job_id.isin(pole_ids)]
    write_csv(output / "constrained_ransac_pole_cases.csv", focus)
    (output / "run_metadata.json").write_text(json.dumps({
        "experiment": "constrained_ransac", "variants": list(VARIANTS),
        "pipelines": list(HOMOGRAPHY_PIPELINES), "cases": len(records),
        "limit": limit, "seconds": time.time() - started,
    }, indent=2) + "\n")
    if not summary[summary.variant.eq("none")].reproduces_cache_all.astype(bool).all():
        print("WARNING: the unconstrained re-fit does not reproduce every cached transform; "
              "inspect constrained_ransac_cases.csv (reproduces_cache == False).", flush=True)
    print(f"[ransac] done in {time.time() - started:.0f}s -> {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("clearance", "constrained-ransac", "all"))
    parser.add_argument("--output", default="reports/isbi_submission_latest")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="worker processes (default: logical CPUs - 2)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--allow-stale-cache", action="store_true",
                        help="smoke test only: skip the provenance filter on cached rows")
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke test: first N cases per pipeline (constrained-ransac only)")
    args = parser.parse_args()
    output = ROOT / args.output
    if args.command in ("clearance", "all"):
        run_clearance(output / "clearance", resamples=args.bootstrap, workers=args.workers)
    if args.command in ("constrained-ransac", "all"):
        run_constrained_ransac(output / "constrained_ransac", workers=args.workers,
                               limit=args.limit, allow_stale_cache=args.allow_stale_cache)


if __name__ == "__main__":
    main()
