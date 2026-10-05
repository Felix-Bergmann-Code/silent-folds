"""Render the two deidentified COph100 XFeat pole cases side by side.

The source RIDIRP pixels are CC0. Every overlay is derived from a frozen
homography; registration is not rerun. The paired layout distinguishes the
extreme pole confined to a black corner from the pole crossing the retinal FOV.
"""

# ruff: noqa: E402 -- the noninteractive backend must be selected before pyplot.

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from PIL import Image

from warpaudit.geometry.projective import projective_pole_crosses_circle

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (
        "COph100/055-1/055_M_GA38_BW3245_PA42_DG13_PF0_D1_S03_2__055_M_GA38_BW3245_PA78_DG13_PF0_D1_S09_16",
        "Rectangle-only pole",
        "rectangle crossing;\nFOV/mask clear",
    ),
    (
        "COph100/055/055_M_GA38_BW3245_PA43_DG13_PF0_D1_S04_13__055_M_GA38_BW3245_PA78_DG13_PF0_D1_S09_6",
        "FOV-crossing pole",
        "rectangle and\nFOV crossing",
    ),
)


def map_points(matrix: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    denominator = homogeneous[:, 2]
    mapped = homogeneous[:, :2] / denominator[:, None]
    return mapped, denominator


def curvature_proxy(matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    nx, ny = 240, 180
    x = np.linspace(0, width - 1, nx)
    y = np.linspace(0, height - 1, ny)
    xx, yy = np.meshgrid(x, y)
    mapped, _ = map_points(matrix, np.c_[xx.ravel(), yy.ravel()])
    u = mapped[:, 0].reshape(ny, nx)
    v = mapped[:, 1].reshape(ny, nx)
    spacing_y, spacing_x = y[1] - y[0], x[1] - x[0]
    uyy = np.gradient(np.gradient(u, spacing_y, axis=0), spacing_y, axis=0)
    uxx = np.gradient(np.gradient(u, spacing_x, axis=1), spacing_x, axis=1)
    vyy = np.gradient(np.gradient(v, spacing_y, axis=0), spacing_y, axis=0)
    vxx = np.gradient(np.gradient(v, spacing_x, axis=1), spacing_x, axis=1)
    curvature = uyy**2 + uxx**2 + vyy**2 + vxx**2
    return np.log10(np.clip(curvature, 1e-8, None))


def shared_color_range(case_payloads) -> tuple[float, float]:
    values = np.concatenate(
        [p["curvature"][np.isfinite(p["curvature"])] for p in case_payloads]
    )
    low, high = np.percentile(values, (1.0, 99.5))
    return float(low), float(high)


def render_four_panel(case_payloads, color_min, color_max) -> list[dict]:
    """Four-panel overview of both cases; also returns the per-case geometry report."""

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.35), layout="constrained")
    reports = []
    panel = 0
    curvature_images = []
    for payload in case_payloads:
        row = payload["row"]
        title = payload["title"]
        subtitle = payload["subtitle"]
        moving = payload["moving"]
        width = payload["width"]
        height = payload["height"]
        center = payload["center"]
        radius = payload["radius"]
        line_x = payload["line_x"]
        line_y = payload["line_y"]
        curvature = payload["curvature"]

        case_index = payload["case_index"]
        image_axis, curvature_axis = axes[2 * case_index : 2 * case_index + 2]
        image_axis.imshow(moving)
        image_axis.add_patch(
            Circle(
                center,
                radius,
                fill=False,
                edgecolor="white",
                linewidth=0.8,
                linestyle=":",
            )
        )
        curvature_images.append(
            curvature_axis.imshow(
                curvature,
                extent=[0, width - 1, height - 1, 0],
                cmap="magma",
                vmin=color_min,
                vmax=color_max,
            )
        )
        for axis in (image_axis, curvature_axis):
            axis.plot(line_x, line_y, color="#00e5ff", lw=1.2, ls="--")
            axis.set_xlim(0, width - 1)
            axis.set_ylim(height - 1, 0)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.text(
                0.02,
                0.96,
                chr(ord("a") + panel),
                transform=axis.transAxes,
                va="top",
                color="white",
                weight="bold",
            )
            panel += 1
        # The ISBI figure is placed at 92% scale; 10 pt here remains above the
        # template's 9 pt minimum after inclusion in the manuscript.
        image_axis.set_title(f"{title}\n{subtitle}", fontsize=10)
        curvature_axis.set_title("Log curvature proxy", fontsize=10)
        reports.append(
            {
                "job_id": row.job_id,
                "pair_id": row.pair_id,
                "denominator_crosses_inscribed_circular_fov": payload[
                    "crosses_circle"
                ],
                "image_width": width,
                "image_height": height,
                "circle_radius_px": radius,
                "line_distance_from_circle_center_px": payload["line_distance"],
                "line_chord_length_in_circle_px": payload["chord_length"],
                "chord_fraction_of_circle_diameter": payload["chord_length"]
                / (2 * radius),
            }
        )

    colorbar = fig.colorbar(
        curvature_images[0],
        ax=[axes[1], axes[3]],
        orientation="horizontal",
        fraction=0.08,
        pad=0.04,
        aspect=28,
    )
    colorbar.set_label(r"Shared $\log_{10}$ squared-curvature scale", fontsize=10)
    colorbar.ax.tick_params(labelsize=9.5)
    for asset_directory in (ROOT / "reports/feedback_revision_latest",):
        asset_directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            asset_directory / "qualitative_pole.pdf",
            metadata={"CreationDate": None, "ModDate": None},
        )
        fig.savefig(asset_directory / "qualitative_pole.png", dpi=220)
    plt.close(fig)
    return reports


def render_isbi(case_payloads, color_min, color_max) -> None:
    """Full-text-width ISBI layout: one header per case, slim vertical colorbar.

    Drawn at the 178 mm ISBI text width and included at 100%, so 9 pt text
    stays 9 pt in print.
    """

    from matplotlib.gridspec import GridSpec

    ink, muted, pole = "#1a1a1a", "#6b6b6b", "#00e5ff"
    headers = ("Case 1: pole clips a corner, misses the FOV",
               "Case 2: pole crosses the FOV")
    with plt.rc_context({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "axes.edgecolor": muted, "axes.linewidth": 0.5}):
        fig = plt.figure(figsize=(7.0, 1.62))
        grid = GridSpec(
            1, 6, figure=fig, width_ratios=[1, 1, 0.08, 1, 1, 0.045],
            left=0.004, right=0.925, bottom=0.02, top=0.80, wspace=0.04,
        )
        columns = ((0, 1), (3, 4))
        image = None
        case_axes = []
        for payload, (ci, cc), header, letters in zip(
            case_payloads, columns, headers, (("a", "b"), ("c", "d")), strict=True
        ):
            width, height = payload["width"], payload["height"]
            image_axis = fig.add_subplot(grid[0, ci])
            curvature_axis = fig.add_subplot(grid[0, cc])
            image_axis.imshow(payload["moving"])
            image_axis.add_patch(Circle(payload["center"], payload["radius"], fill=False,
                                        edgecolor="white", linewidth=0.8, linestyle=":"))
            image = curvature_axis.imshow(
                payload["curvature"], extent=[0, width - 1, height - 1, 0],
                cmap="magma", vmin=color_min, vmax=color_max, interpolation="bilinear",
            )
            for axis, letter in ((image_axis, letters[0]), (curvature_axis, letters[1])):
                axis.plot(payload["line_x"], payload["line_y"], color=pole, lw=1.1, ls="--")
                axis.set_xlim(0, width - 1)
                axis.set_ylim(height - 1, 0)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
                axis.text(0.03, 0.95, letter, transform=axis.transAxes, va="top",
                          ha="left", color="white", weight="bold")
            case_axes.append((image_axis, curvature_axis, header))
        # Aspect-locked images are shorter than their grid cells: anchor the
        # headers and the colorbar to the drawn image boxes, not the grid.
        fig.canvas.draw()
        box = case_axes[0][0].get_position()
        for image_axis, curvature_axis, header in case_axes:
            left = image_axis.get_position().x0
            right = curvature_axis.get_position().x1
            fig.text((left + right) / 2, box.y1 + 0.13, header, ha="center",
                     va="center", color=ink)
            fig.add_artist(plt.Line2D([left, right], [box.y1 + 0.045] * 2, color=muted,
                                      lw=0.6, transform=fig.transFigure))
        colorbar_axis = fig.add_subplot(grid[0, 5])
        cell = colorbar_axis.get_position()
        colorbar_axis.set_position([cell.x0, box.y0, cell.width, box.height])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_ticks([-5, 0, 5])
        colorbar.outline.set_linewidth(0.4)
        colorbar.ax.tick_params(length=2, width=0.4, pad=1.5)
        colorbar.set_label(r"$\log_{10}$ bending energy", labelpad=2)
        asset_directory = ROOT / "paper/isbi_2027/assets"
        asset_directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(asset_directory / "qualitative_pole.pdf",
                    metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(asset_directory / "qualitative_pole.png", dpi=300)
        plt.close(fig)


def main() -> None:
    rows = pd.read_csv(ROOT / "reports/pole_guard_ablation_latest/pole_cases.csv")
    required = {"pair_id", "moving_image_id", "original_homography_json"}
    if not required <= set(rows):
        raise RuntimeError(
            "pole_cases.csv predates the paired-figure handoff; rerun pole_guard_ablation.py"
        )
    case_payloads = []
    for case_index, (pair_id, title, subtitle) in enumerate(CASES):
        match = rows[
            (rows.pair_id.astype(str) == pair_id)
            & (rows.pipeline_id.astype(str) == "xfeat_h")
        ]
        if len(match) != 1:
            raise RuntimeError(f"expected one returned row for {pair_id}, found {len(match)}")
        row = match.iloc[0]
        h_original = np.asarray(json.loads(row.original_homography_json), dtype=float)
        moving = np.asarray(
            Image.open(ROOT / "data" / (row.moving_image_id + ".jpg")).convert("RGB")
        )
        height, width = moving.shape[:2]
        center = ((width - 1) / 2, (height - 1) / 2)
        radius = min(width - 1, height - 1) / 2
        h31, h32, h33 = h_original[2]
        line_x = np.array([0, width - 1])
        line_y = -(h31 * line_x + h33) / h32
        crosses_circle = projective_pole_crosses_circle(
            h_original, center_x=center[0], center_y=center[1], radius=radius
        )
        line_distance = abs(h31 * center[0] + h32 * center[1] + h33) / np.hypot(
            h31, h32
        )
        chord_length = 2 * np.sqrt(max(0.0, radius**2 - line_distance**2))
        curvature = curvature_proxy(h_original, width, height)
        case_payloads.append(
            {
                "case_index": case_index,
                "row": row,
                "title": title,
                "subtitle": subtitle,
                "moving": moving,
                "width": width,
                "height": height,
                "center": center,
                "radius": radius,
                "line_x": line_x,
                "line_y": line_y,
                "curvature": curvature,
                "crosses_circle": crosses_circle,
                "line_distance": line_distance,
                "chord_length": chord_length,
            }
        )

    color_min, color_max = shared_color_range(case_payloads)
    reports = render_four_panel(case_payloads, color_min, color_max)
    render_isbi(case_payloads, color_min, color_max)
    report = ROOT / "reports/feedback_revision_latest/fov_pole_cases.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "cases": reports,
                "note": "The circle is an explicit conservative geometric proxy for the fundus field of view.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
