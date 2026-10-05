#!/usr/bin/env python3
"""Build the deterministic near-pole response experiment for the ISBI paper.

The synthetic homography has a vertical projective pole just outside the right
edge of a 640 x 480 support.  Its distance from that edge is prescribed as a
fraction of the image diagonal, which is exactly the paper's continuous
clearance.  No study images, learned parameters, or cached registrations enter
this experiment.
"""

# ruff: noqa: E402 -- the noninteractive backend must be selected before pyplot.

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from warpaudit.geometry.projective import projective_pole_diagnostics

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "paper/isbi_2027/assets"
REPORT = ROOT / "reports/isbi_synthetic_near_pole.csv"
WIDTH = 640
HEIGHT = 480
GRID_SIZES = (16, 32, 64)


def sampled_bending_energy(
    matrix: np.ndarray,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    size: int = 32,
) -> float:
    """Reproduce the dimensionless Family-D bending-energy calculation."""

    xs = (np.arange(size, dtype=float) + 0.5) * (width / size) - 0.5
    ys = (np.arange(size, dtype=float) + 0.5) * (height / size) - 0.5
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.c_[xx.ravel(), yy.ravel(), np.ones(size * size)]
    homogeneous = points @ np.asarray(matrix, dtype=float).T
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mapped = homogeneous[:, :2] / homogeneous[:, 2, None]

    diagonal = float(np.hypot(width, height))
    source_n = (points[:, :2] / diagonal).reshape(size, size, 2)
    mapped_n = (mapped / diagonal).reshape(size, size, 2)
    d_col = float(source_n[0, 1, 0] - source_n[0, 0, 0])
    d_row = float(source_n[1, 0, 1] - source_n[0, 0, 1])
    d_mapped_drow = np.gradient(mapped_n, d_row, axis=0)
    d_mapped_dcol = np.gradient(mapped_n, d_col, axis=1)
    d2_row = np.gradient(d_mapped_drow, d_row, axis=0)
    d2_col = np.gradient(d_mapped_dcol, d_col, axis=1)
    d2_cross = np.gradient(d_mapped_drow, d_col, axis=1)
    energy = d2_row**2 + 2.0 * d2_cross**2 + d2_col**2
    finite = energy[np.isfinite(energy)]
    if not finite.size:
        raise ValueError("synthetic homography produced no finite curvature samples")
    return float(finite.mean())


def synthetic_homography(clearance: float) -> np.ndarray:
    """Return a valid map whose vertical pole has the requested clearance."""

    clearance = float(clearance)
    if not np.isfinite(clearance) or clearance <= 0:
        raise ValueError("clearance must be positive and finite")
    support_diagonal = float(np.hypot(WIDTH - 1.0, HEIGHT - 1.0))
    pole_x = WIDTH - 1.0 + clearance * support_diagonal
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0 / pole_x, 0.0, 1.0)),
        dtype=float,
    )


def build_sweep() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for requested in np.logspace(-3.0, 0.0, 81):
        matrix = synthetic_homography(float(requested))
        diagnostic = projective_pole_diagnostics(
            matrix, width=WIDTH, height=HEIGHT
        )
        if diagnostic.denominator_crosses_image:
            raise AssertionError("positive-clearance synthetic map must not cross support")
        if not np.isclose(
            diagnostic.pole_clearance_diagonal_fraction,
            requested,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise AssertionError("constructed and measured clearances disagree")
        for grid_size in GRID_SIZES:
            rows.append(
                {
                    "requested_clearance": float(requested),
                    "measured_clearance": diagnostic.pole_clearance_diagonal_fraction,
                    "grid_size": grid_size,
                    "bending_energy": sampled_bending_energy(
                        matrix, size=grid_size
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    frame = build_sweep()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORT, index=False, lineterminator="\n")

    colors = {16: "#4C78A8", 32: "#E45756", 64: "#54A24B"}
    markers = {16: "o", 32: "s", 64: "^"}
    fig, axis = plt.subplots(figsize=(3.35, 2.05), layout="constrained")
    for grid_size in GRID_SIZES:
        subset = frame[frame.grid_size.eq(grid_size)]
        axis.plot(
            subset.measured_clearance,
            subset.bending_energy,
            color=colors[grid_size],
            linewidth=1.35,
            marker=markers[grid_size],
            markevery=16,
            markersize=3.0,
            label=rf"${grid_size}\times{grid_size}$",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Normalized pole clearance $\rho_R$")
    axis.set_ylabel(r"Bending energy $\mathcal{B}$")
    axis.grid(True, which="major", color="#d9d9d9", linewidth=0.55)
    axis.grid(True, which="minor", color="#eeeeee", linewidth=0.35)
    axis.legend(frameon=False, ncols=1, loc="upper right", fontsize=9)
    axis.tick_params(labelsize=9)
    axis.xaxis.label.set_size(9)
    axis.yaxis.label.set_size(9)

    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        ASSETS / "synthetic_near_pole.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(ASSETS / "synthetic_near_pole.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
