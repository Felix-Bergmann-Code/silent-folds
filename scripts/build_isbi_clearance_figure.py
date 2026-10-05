#!/usr/bin/env python3
"""ISBI Fig. 2: why clearance is a QC signal.

(a) Lowest AUROC across the three pipelines (XFeat-H, SPLG-H, SuperRetina)
    as the failure endpoint varies from 0.0025 to 0.02 of the image diagonal,
    for -log rho, inlier ratio, and the cross-fitted fused score
    (``reports/isbi_review_analyses/threshold_sensitivity.csv``).
(b) Distribution of normalized clearance for failed versus successful
    returned XFeat-H and SPLG-H homographies
    (``reports/pole_guard_ablation_latest/pole_clearance.csv``).

Reads committed CSVs only; no registration or detector is run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/isbi_2027/assets"
FAIL, SUCCESS = "#D55E00", "#0072B2"  # validated CVD-safe pair
INK, MUTED = "#222222", "#777777"
# Panel (a) series; validated with the dataviz palette checker (light mode).
# Each entry: column, direct label, color, line style, width, label anchor.
SCORES = (("clearance_auroc", r"$-\log\rho$", "#009E73", (0, (3, 1.5)), 1.2,
           (0.0165, 0.49, "right")),
          ("inlier_ratio_auroc", "inlier", "#B8860B", (0, (1, 1)), 1.2,
           (0.0026, 0.55, "left")),
          ("fused_auroc", "fused", "#7B3294", "-", 1.8, (0.008, 0.9, "center")))

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK,
    "ytick.color": INK, "axes.labelcolor": INK,
})


def main() -> None:
    sweep = pd.read_csv(ROOT / "reports/isbi_review_analyses/threshold_sensitivity.csv")
    worst = sweep.groupby("threshold")[[c for c, *_ in SCORES]].min().sort_index()
    cases = pd.read_csv(ROOT / "reports/pole_guard_ablation_latest/pole_clearance.csv")
    cases = cases[cases.pipeline_id.isin(["xfeat_h", "sp_lg_h"])]
    rho = cases.pole_clearance_diagonal_fraction.astype(float)
    floor = 10 ** -2.5
    log_rho = np.log10(np.clip(rho, floor, None))
    failed = cases.operational_failure.astype(bool)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(3.4, 1.7), layout="constrained", gridspec_kw={"wspace": 0.08}
    )
    for column, label, color, style, width, (x, y, align) in SCORES:
        left.plot(worst.index, worst[column], color=color, ls=style, lw=width)
        left.text(x, y, label, color=INK, va="center", ha=align, fontsize=9)
    left.set_xscale("log")
    left.set_xlim(0.0025, 0.02)
    left.set_xticks([0.005, 0.01, 0.02])
    left.set_xticklabels([".005", ".01", ".02"])
    left.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    left.set_ylim(0.45, 1.0)
    left.set_yticks([0.5, 0.75, 1.0], [".50", ".75", "1"])
    left.axvline(0.005, color=MUTED, lw=0.8, ls="--")
    left.set_xlabel("endpoint (diagonals)")
    left.set_ylabel("lowest AUROC")
    left.set_title("(a) endpoint sweep", fontsize=9, color=INK, pad=2)

    bins = np.arange(-2.5, 2.51, 0.25)
    capped = np.clip(log_rho, -2.4, 2.4)
    for mask, color in ((failed, FAIL), (~failed, SUCCESS)):
        right.hist(capped[mask], bins=bins, histtype="step", color=color, lw=1.4)
    right.axvline(0.0, color=MUTED, lw=0.8, ls="--")
    right.text(-0.15, 150, "failure", color=FAIL, ha="right", va="center")
    right.text(1.45, 125, "success", color=SUCCESS, ha="left", va="center")
    right.set_xlabel(r"$\log_{10}\rho$ (retinal)")
    right.set_ylabel("estimates")
    right.set_xticks([-2, 0, 2])
    right.set_xticklabels([r"$\leq\!-2$", "0", r"$\geq\!2$"])
    right.set_title("(b) XFeat-H + SPLG-H", fontsize=9, color=INK, pad=2)
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "clearance_qc.pdf")
    fig.savefig(OUT / "clearance_qc.png", dpi=300)
    near = cases[rho <= 1.0]
    print(f"rho<=1: {int(near.operational_failure.sum())}/{len(near)} failures; "
          f"crossings/floored: {int((rho <= floor).sum())}")


if __name__ == "__main__":
    main()
