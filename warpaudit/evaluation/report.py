"""Manuscript tables and figures built from frozen caches (spec §16.2).

Every artefact here is derived, never hand-maintained: the tables are built
from the manifests, caches, and results the run produced, so a number in the
paper can be traced to the row that produced it.

Two constraints shape the module. Figures are optional at import time --
matplotlib is a reporting dependency deliberately kept out of the cache
identity, so its absence must degrade to the underlying CSV rather than fail a
run. And the failure gallery renders source pixels, so it is written only to
the ignored local review directory and never to the manuscript figure
directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "FIGURE_SPECS",
    "FigureSpec",
    "markdown_table",
    "matplotlib_available",
    "provenance_table",
    "capability_table",
    "policy_table",
    "primary_table",
    "per_pipeline_table",
    "signal_family_table",
    "render_figures",
]


def markdown_table(frame: pd.DataFrame, *, float_format: str = "{:.4f}") -> str:
    """Render a frame as a GitHub table without a third-party dependency.

    ``DataFrame.to_markdown`` needs ``tabulate``, which would be another pinned
    dependency for formatting alone; the tables are also written as CSV, which
    stays the machine-readable form.
    """

    def cell(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float | np.floating):
            return "" if not np.isfinite(value) else float_format.format(float(value))
        if isinstance(value, bool | np.bool_):
            return "yes" if value else "no"
        return str(value).replace("|", "\\|")

    columns = [str(c) for c in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body]) + "\n"


def matplotlib_available() -> tuple[bool, str]:
    try:
        import matplotlib  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def provenance_table(pairs: pd.DataFrame, datasets: Sequence[Any]) -> pd.DataFrame:
    """Data provenance, group counts, and release permissions (§16.2)."""
    rows = []
    by_dataset = {str(k): v for k, v in pairs.groupby("dataset_id")}
    for dataset in datasets:
        frame = by_dataset.get(dataset.id)
        if frame is None:
            continue
        development = frame[frame["is_development"].astype(bool)]
        rows.append(
            {
                "dataset": dataset.id,
                "version": dataset.version,
                "pairs": int(len(frame)),
                "images": int(
                    pd.concat(
                        (frame["moving_image_id"], frame["fixed_image_id"])
                    ).nunique()
                ),
                "groups": int(frame["group_id"].nunique()),
                "group_basis": str(frame["group_basis"].iloc[0]),
                "development_groups": int(development["group_id"].nunique()),
                "confirmatory_groups": int(
                    frame["group_id"].nunique() - development["group_id"].nunique()
                ),
                "patients_identified": bool(dataset.patient_ids_available),
                "licence": dataset.licence,
                "redistribute_images": bool(dataset.redistribute_images),
                "redistribute_derived": dataset.redistribute_derived,
                "source": dataset.source_url,
            }
        )
    return pd.DataFrame(rows)


def capability_table(features: pd.DataFrame, pipelines: Sequence[Any]) -> pd.DataFrame:
    """Pipeline/checkpoint/capability matrix with measured availability (§7.3)."""
    rows = []
    availability: dict[tuple[str, str], float] = {}
    if not features.empty and {"family", "available", "job_id"} <= set(features):
        grouped = features.groupby(["pipeline_id", "family"])["available"]
        availability = {
            (str(pipeline), str(family)): float(value)
            for (pipeline, family), value in grouped.mean().items()
        }
    families = sorted(set(features["family"].astype(str))) if not features.empty else []
    for pipeline in pipelines:
        row: dict[str, Any] = {
            "pipeline": pipeline.id,
            "version": pipeline.version,
            "matcher": pipeline.matcher,
            "checkpoint": pipeline.checkpoint,
            "transform_family": pipeline.transform_family,
            "fit": pipeline.fit,
            "in_common_block": bool(pipeline.in_common_block),
        }
        for family in families:
            row[f"{family}_available"] = availability.get((pipeline.id, family), float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def primary_table(results: Mapping[str, Any]) -> pd.DataFrame:
    """The three joint-claim components with bounds and two-sided intervals."""
    frame = pd.DataFrame(list(results.get("joint_claim", ())))
    bootstrap = results.get("bootstrap", {})
    if not frame.empty and bootstrap:
        frame["invalid_resamples"] = [
            bootstrap.get(key, {}).get("n_invalid", float("nan"))
            for key in ("target_auroc_transferred", "gap_auc", "delta_policy")[: len(frame)]
        ]
    return frame


def policy_table(results: Mapping[str, Any]) -> pd.DataFrame:
    """Realised operating points of both policies on the same target test groups."""
    aggregate = results.get("aggregate", {})
    return pd.DataFrame(
        [
            {
                "policy": "frozen source (pi_X)",
                "realised_target_coverage": aggregate.get("realised_coverage_source_on_target"),
                "accepted_groups": aggregate.get("accepted_groups_source_on_target"),
                "target_auroc": aggregate.get("target_auroc_transferred"),
            },
            {
                "policy": "matched-budget target reference (pi_Y,n)",
                "realised_target_coverage": aggregate.get(
                    "realised_coverage_reference_on_target"
                ),
                "accepted_groups": aggregate.get("accepted_groups_reference_on_target"),
                "target_auroc": aggregate.get("target_auroc_reference"),
            },
        ]
    )


def per_pipeline_table(
    cases: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    """Per-pipeline and per-dataset cells, reported before any macro-average (§8.4)."""
    if predictions.empty or cases.empty:
        return pd.DataFrame()
    merged = predictions.merge(
        cases[["job_id", "dataset_id", "pipeline_id", "group_id", "operational_failure"]],
        on="job_id",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    grouped = merged.groupby(["arm", "dataset_id", "pipeline_id"])
    return (
        grouped.agg(
            cases=("job_id", "nunique"),
            groups=("group_id", "nunique"),
            failures=("operational_failure", "sum"),
            prevalence=("operational_failure", "mean"),
            accepted=("accepted", "sum"),
        )
        .reset_index()
        .sort_values(["arm", "dataset_id", "pipeline_id"], kind="stable")
    )


def signal_family_table(results: Mapping[str, Any]) -> pd.DataFrame:
    """Single-score arms beside the composite, with the paired gap (§16.2)."""
    aggregate = results.get("aggregate", {})
    transferred = aggregate.get("target_auroc_transferred", float("nan"))
    rows = [
        {
            "arm": "composite (frozen families)",
            "target_auroc": transferred,
            "gap_to_composite": 0.0,
        }
    ]
    for key, value in sorted(aggregate.items()):
        if key.endswith("__folds_defined"):
            continue
        if key.startswith("single_") and key.endswith("_auroc"):
            name = key[len("single_") : -len("_auroc")]
            rows.append(
                {
                    "arm": f"single: {name}",
                    "target_auroc": value,
                    "gap_to_composite": transferred - value,
                }
            )
        elif key.startswith("control_") and key.endswith("_auroc"):
            name = key[len("control_") : -len("_auroc")]
            rows.append(
                {
                    "arm": f"control: {name}",
                    "target_auroc": value,
                    "gap_to_composite": transferred - value,
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class FigureSpec:
    """A figure and the table it is drawn from, so the data survives without it."""

    name: str
    title: str
    description: str


FIGURE_SPECS: tuple[FigureSpec, ...] = (
    FigureSpec(
        "policy_operating_points",
        "Frozen source policy versus matched-budget target reference",
        "Both policies on the same untouched target test groups: realised coverage "
        "against accepted failure risk, with the source-test point as secondary context.",
    ),
    FigureSpec(
        "signal_family_comparison",
        "Signal families and controls",
        "Absolute target AUROC per arm with the paired gap to the composite.",
    ),
    FigureSpec(
        "calibration",
        "Predicted against observed failure risk",
        "Five fixed equal-width probability bins with counts; tiny bins are not "
        "interpreted.",
    ),
    FigureSpec(
        "cost",
        "Measured cost against incremental utility",
        "Measured wall time per family beside the discrimination it supports.",
    ),
)


def render_figures(
    figure_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    results: Mapping[str, Any],
    *,
    reliability: pd.DataFrame | None = None,
    costs: pd.DataFrame | None = None,
) -> list[str]:
    """Render the figures matplotlib can draw; return the files written."""
    ok, _ = matplotlib_available()
    if not ok:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    policy = tables.get("policy_operating_points")
    if policy is not None and not policy.empty:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        aggregate = results.get("aggregate", {})
        risks = [
            aggregate.get("policy_risk_source_on_target", float("nan")),
            aggregate.get("policy_risk_reference_on_target", float("nan")),
        ]
        source_test_risk = aggregate.get("policy_risk_source_on_source_test", float("nan"))
        if np.isfinite(source_test_risk):
            ax.scatter(
                aggregate.get("source_test_coverage", float("nan")),
                source_test_risk,
                s=60,
                marker="^",
                color="grey",
                label="frozen source policy on source test (secondary)",
            )
        for (_, row), risk in zip(policy.iterrows(), risks, strict=False):
            ax.scatter(
                row["realised_target_coverage"],
                risk,
                s=90,
                label=f"{row['policy']} ({row['accepted_groups']:.0f} groups)",
            )
        ax.set_xlabel("realised coverage on target test")
        ax.set_ylabel("accepted failure risk")
        ax.set_title("Same target test groups, thresholds from own calibration")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        path = figure_dir / "policy_operating_points.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)

    families = tables.get("signal_family_comparison")
    if families is not None and not families.empty:
        frame = families.dropna(subset=["target_auroc"]).sort_values("target_auroc")
        if not frame.empty:
            fig, ax = plt.subplots(figsize=(6.4, max(2.6, 0.28 * len(frame))))
            ax.barh(frame["arm"], frame["target_auroc"], color="#4c72b0")
            ax.axvline(0.5, color="grey", linestyle=":", linewidth=1)
            ax.set_xlabel("target AUROC")
            ax.set_title("Arms on the same target test population")
            ax.tick_params(labelsize=7)
            fig.tight_layout()
            path = figure_dir / "signal_family_comparison.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path.name)

    if reliability is not None and not reliability.empty:
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
        for arm, frame in reliability.groupby("arm"):
            ax.plot(
                frame["mean_predicted"], frame["observed"], marker="o", label=str(arm)
            )
            for _, row in frame.iterrows():
                ax.annotate(
                    f"n={int(row['n'])}",
                    (row["mean_predicted"], row["observed"]),
                    fontsize=6,
                    xytext=(2, 3),
                    textcoords="offset points",
                )
        ax.set_xlabel("mean predicted failure probability")
        ax.set_ylabel("observed failure rate")
        ax.set_title("Reliability, five fixed bins")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = figure_dir / "calibration.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)

    if costs is not None and not costs.empty:
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        ax.bar(costs["family"].astype(str), costs["median_wall_s"], color="#55a868")
        ax.set_ylabel("median wall time per case (s)")
        ax.set_yscale("log")
        ax.set_title("Measured incremental cost per signal family")
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        path = figure_dir / "cost.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)

    return written


def reliability_frame(
    predictions: pd.DataFrame, cases: pd.DataFrame, *, bins: int = 5
) -> pd.DataFrame:
    """Fixed equal-width probability bins with counts (§9.1)."""
    if predictions.empty or cases.empty:
        return pd.DataFrame()
    merged = predictions.merge(
        cases[["job_id", "operational_failure"]], on="job_id", how="inner"
    )
    merged = merged[np.isfinite(merged["probability"].to_numpy(dtype=float))]
    if merged.empty:
        return pd.DataFrame()
    edges = np.linspace(0.0, 1.0, bins + 1)
    merged["bin"] = np.clip(
        np.digitize(merged["probability"].to_numpy(dtype=float), edges[1:-1]), 0, bins - 1
    )
    grouped = merged.groupby(["arm", "bin"])
    frame = grouped.agg(
        mean_predicted=("probability", "mean"),
        observed=("operational_failure", "mean"),
        n=("job_id", "size"),
    ).reset_index()
    frame["bin_low"] = edges[frame["bin"].to_numpy(dtype=int)]
    frame["bin_high"] = edges[frame["bin"].to_numpy(dtype=int) + 1]
    return frame
