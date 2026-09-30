#!/usr/bin/env python3
"""Evaluate cached homographies with the standard FIRE success-rate protocol.

The script reads current clean canonical registration and label caches. It does
not launch a matcher, refit a detector, or expose annotations to a predictor.
Undefined/no-output registrations count as unsuccessful at every threshold.
"""

# ruff: noqa: E402 -- add the repository root before local package imports.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from warpaudit.cache.store import ShardedTable, atomic_write_text
from warpaudit.cli import _current_registration_rows, _pairs_manifest
from warpaudit.config import load_config


def exact_normalized_auc(errors_px: np.ndarray, maximum_px: float) -> float:
    """Area under the empirical success curve on [0, maximum_px]."""

    errors = np.asarray(errors_px, dtype=float)
    contributions = np.zeros(len(errors), dtype=float)
    finite = np.isfinite(errors)
    contributions[finite] = np.clip(
        (maximum_px - np.maximum(errors[finite], 0.0)) / maximum_px,
        0.0,
        1.0,
    )
    return float(contributions.mean())


def evaluate(
    config: Path,
    output: Path,
    pipelines: tuple[str, ...],
    maximum_px: float,
    step_px: float,
) -> None:
    cfg = load_config(config)
    pairs, _ = _pairs_manifest(cfg, ROOT)
    fire_pairs = pairs[pairs.dataset_id.astype(str).eq("FIRE")].copy()
    if fire_pairs.empty:
        raise ValueError("FIRE pairs are absent from the manifest")
    categories = tuple(sorted(fire_pairs.dataset_category.astype(str).unique()))
    if set(categories) != {"A", "P", "S"}:
        raise ValueError(f"expected FIRE categories A/P/S, found {categories}")

    cache_root = cfg.paths.resolve(ROOT)["cache_root"]
    registrations = ShardedTable(cache_root, "registrations").load()
    registrations = _current_registration_rows(registrations, pairs, cfg, ROOT)
    registrations = registrations[
        registrations.dataset_id.astype(str).eq("FIRE")
        & registrations.pipeline_id.astype(str).isin(pipelines)
        & registrations.direction.astype(str).eq("canonical")
        & registrations.condition.astype(str).eq("clean")
    ].drop_duplicates("job_id", keep="last")
    labels = ShardedTable(cache_root, "labels").load().drop_duplicates(
        "job_id", keep="last"
    )
    cases = registrations.merge(
        labels[["job_id", "tre_px", "tre_defined"]],
        on="job_id",
        how="left",
        validate="one_to_one",
    ).merge(
        fire_pairs[["pair_id", "dataset_category"]],
        on="pair_id",
        how="left",
        validate="many_to_one",
    )

    thresholds = np.arange(0.0, maximum_px + step_px / 2.0, step_px)
    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    pair_rows: list[pd.DataFrame] = []
    expected_ids = set(fire_pairs.pair_id.astype(str))
    for pipeline in pipelines:
        pipeline_cases = cases[cases.pipeline_id.astype(str).eq(pipeline)].copy()
        got_ids = set(pipeline_cases.pair_id.astype(str))
        if got_ids != expected_ids or len(pipeline_cases) != len(expected_ids):
            missing = sorted(expected_ids - got_ids)
            raise ValueError(
                f"{pipeline}: expected {len(expected_ids)} FIRE cases, got "
                f"{len(pipeline_cases)}; missing examples={missing[:5]}"
            )
        pipeline_cases["tre_px_for_curve"] = pd.to_numeric(
            pipeline_cases.tre_px, errors="coerce"
        ).where(pipeline_cases.tre_defined.fillna(False).astype(bool), np.nan)
        pair_rows.append(
            pipeline_cases[
                [
                    "pipeline_id",
                    "pair_id",
                    "dataset_category",
                    "status",
                    "tre_px_for_curve",
                ]
            ]
        )
        strata = [(category, pipeline_cases.dataset_category.eq(category)) for category in categories]
        strata.append(("ALL", pd.Series(True, index=pipeline_cases.index)))
        for category, selector in strata:
            cell = pipeline_cases[selector]
            errors = cell.tre_px_for_curve.to_numpy(float)
            success = np.array(
                [np.mean(np.isfinite(errors) & (errors <= threshold)) for threshold in thresholds]
            )
            sampled_auc = float(np.trapz(success, thresholds) / maximum_px)
            exact_auc = exact_normalized_auc(errors, maximum_px)
            summary_rows.append(
                {
                    "pipeline_id": pipeline,
                    "category": category,
                    "pairs": len(cell),
                    "defined_tre": int(np.isfinite(errors).sum()),
                    "maximum_threshold_px": maximum_px,
                    "curve_step_px": step_px,
                    "normalized_auc": exact_auc,
                    "sampled_trapezoid_auc": sampled_auc,
                }
            )
            curve_rows.extend(
                {
                    "pipeline_id": pipeline,
                    "category": category,
                    "threshold_px": float(threshold),
                    "success_rate": float(rate),
                }
                for threshold, rate in zip(thresholds, success, strict=True)
            )

    output.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output / "summary.csv", pd.DataFrame(summary_rows).to_csv(index=False))
    atomic_write_text(output / "success_curves.csv", pd.DataFrame(curve_rows).to_csv(index=False))
    atomic_write_text(output / "per_pair_errors.csv", pd.concat(pair_rows).to_csv(index=False))
    diagonal = float(np.hypot(2912, 2912))
    atomic_write_text(
        output / "endpoint_relation.json",
        json.dumps(
            {
                "fire_width_px": 2912,
                "fire_height_px": 2912,
                "diagonal_px": diagonal,
                "paper_endpoint_diagonal_fraction": 0.005,
                "paper_endpoint_px": 0.005 * diagonal,
                "fire_curve_maximum_px": maximum_px,
                "undefined_registration_policy": "unsuccessful at every threshold",
            },
            indent=2,
        )
        + "\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/full_study.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/fire_standard_protocol_latest"))
    parser.add_argument("--pipelines", nargs="+", default=["xfeat_h", "sp_lg_h"])
    parser.add_argument("--maximum-px", type=float, default=25.0)
    parser.add_argument("--step-px", type=float, default=0.01)
    args = parser.parse_args()
    if args.maximum_px <= 0 or args.step_px <= 0:
        parser.error("--maximum-px and --step-px must be positive")
    evaluate(
        args.config,
        args.output,
        tuple(args.pipelines),
        args.maximum_px,
        args.step_px,
    )


if __name__ == "__main__":
    main()
