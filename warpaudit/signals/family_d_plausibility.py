"""Family D: warp plausibility (specification §7.1, §7.3).

Jacobian quantiles, folding fraction, normalised bending energy, and
displacement/scale/shear summaries of the estimated warp.

Two conventions matter enough to state rather than imply, because §7.3 warns
that "diagonal division alone is insufficient for Jacobians, bending energy,
or design-matrix conditioning":

*Domain.* The features describe the estimated forward map, so they are
evaluated on a deterministic lattice in the *moving* working frame, built with
the same pixel-centre convention as :func:`prespecified_grid`. The fixed-frame
evaluation grid handed to signals is the domain of the inverse map, not of the
map being characterised.

*Dimensions.* Moving coordinates are divided by the moving working diagonal
and fixed coordinates by the fixed working diagonal before any derivative is
taken. The Jacobian of the resulting map is therefore dimensionless, and so
are the second derivatives entering the bending energy; neither would be under
a single shared divisor when the two frames differ in size.

A physically plausible warp can still be wrong. These features describe the
geometry of the estimate, never its correctness.
"""

from __future__ import annotations

import numpy as np

from ..geometry.grids import _uniform_centres
from ..types import FeatureBundle, SignalContext
from .registry import SignalSpec, register_signal

__all__ = ["compute_warp_plausibility", "moving_frame_lattice", "normalised_warp_field"]

#: Central differences need an interior, and the singular-value summaries are
#: meaningless on a handful of points.
_MIN_LATTICE = 4


def moving_frame_lattice(ctx: SignalContext, size: int) -> np.ndarray:
    """A deterministic ``size x size`` pixel-centre lattice in the moving frame.

    Shared with family F so that two pipelines' maps are always compared on one
    identical sample set: the lattice depends only on the pair's moving frame,
    never on any pipeline's output.
    """
    moving = ctx.pair.coordinates.moving
    xs = _uniform_centres(size, moving.working_width)
    ys = _uniform_centres(size, moving.working_height)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def normalised_warp_field(ctx: SignalContext, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the dimensionless source lattice and its image under the warp.

    Both returned arrays have shape ``(size, size, 2)`` in ``(row, column, xy)``
    order, so ``numpy.gradient`` differentiates along the lattice axes directly.
    """
    transform = ctx.result.forward_moving_to_fixed
    if transform is None:  # pragma: no cover - guarded by declared requirements
        raise ValueError("family D requires a forward transform")
    moving = ctx.pair.coordinates.moving
    fixed = ctx.pair.coordinates.fixed

    source = moving_frame_lattice(ctx, size)
    mapped = np.asarray(transform.apply(source), dtype=np.float64)

    source_n = (source / moving.working_diagonal).reshape(size, size, 2)
    mapped_n = (mapped / fixed.working_diagonal).reshape(size, size, 2)
    return source_n, mapped_n


def _axis_steps(source_n: np.ndarray) -> tuple[float, float]:
    """Normalised spacing along the lattice row and column axes."""
    d_col = float(source_n[0, 1, 0] - source_n[0, 0, 0])
    d_row = float(source_n[1, 0, 1] - source_n[0, 0, 1])
    return d_row, d_col


def _quantiles(bundle: FeatureBundle, prefix: str, values: np.ndarray, unit: str) -> None:
    finite = values[np.isfinite(values)]
    reason = f"no finite {prefix} sample"
    for name, q in (("p05", 5.0), ("median", 50.0), ("p95", 95.0)):
        bundle.add(
            f"{prefix}_{name}",
            float(np.percentile(finite, q)) if finite.size else None,
            unit=unit,
            reason=reason,
        )


def compute_warp_plausibility(ctx: SignalContext) -> FeatureBundle:
    bundle = FeatureBundle(family="D")
    size = max(int(ctx.config.grid_size), 0)
    if size < _MIN_LATTICE:
        bundle.add(
            "family_available",
            None,
            reason=f"grid_size {size} is below the {_MIN_LATTICE}-point lattice minimum",
        )
        return bundle

    source_n, mapped_n = normalised_warp_field(ctx, size)
    finite_points = np.isfinite(mapped_n).all(axis=2)
    bundle.add(
        "mapped_invalid_fraction", float(1.0 - finite_points.mean()), unit="fraction"
    )
    if finite_points.sum() < _MIN_LATTICE:
        for name in (
            "displacement_mean",
            "displacement_p95",
            "log_jacobian_determinant_p05",
            "log_jacobian_determinant_median",
            "log_jacobian_determinant_p95",
            "folding_fraction",
            "anisotropy_median",
            "anisotropy_p95",
            "scale_median",
            "bending_energy",
        ):
            bundle.add(name, None, reason="too few finite mapped lattice points")
        return bundle

    displacement = np.linalg.norm(mapped_n - source_n, axis=2)
    finite_displacement = displacement[np.isfinite(displacement)]
    bundle.add(
        "displacement_mean",
        float(finite_displacement.mean()) if finite_displacement.size else None,
        unit="frame diagonal",
        reason="no finite displacement",
    )
    bundle.add(
        "displacement_p95",
        float(np.percentile(finite_displacement, 95)) if finite_displacement.size else None,
        unit="frame diagonal",
        reason="no finite displacement",
    )

    d_row, d_col = _axis_steps(source_n)
    # d(mapped)/d(row) and d(mapped)/d(column) of the dimensionless map.
    dv_drow = np.gradient(mapped_n, d_row, axis=0)
    dv_dcol = np.gradient(mapped_n, d_col, axis=1)
    # Columns advance x, rows advance y, so the Jacobian columns are
    # [d/dx, d/dy] = [dv_dcol, dv_drow].
    jacobian = np.stack((dv_dcol, dv_drow), axis=-1)  # (size, size, 2, 2)
    usable = np.isfinite(jacobian).all(axis=(2, 3))

    determinant = np.full((size, size), np.nan)
    determinant[usable] = np.linalg.det(jacobian[usable])
    finite_det = determinant[np.isfinite(determinant)]
    bundle.add(
        "folding_fraction",
        float(np.mean(finite_det <= 0.0)) if finite_det.size else None,
        unit="fraction",
        reason="no finite Jacobian determinant",
    )
    # Log determinant is the scale-symmetric summary: a halving and a doubling
    # of local area are then equal and opposite rather than 0.5 against 2.
    positive = finite_det[finite_det > 0]
    _quantiles(
        bundle,
        "log_jacobian_determinant",
        np.log(positive) if positive.size else np.array([]),
        "log area ratio",
    )

    singular = np.full((size, size, 2), np.nan)
    if usable.any():
        singular[usable] = np.linalg.svd(jacobian[usable], compute_uv=False)
    s1, s2 = singular[..., 0], singular[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        anisotropy = np.where(s2 > 0, s1 / s2, np.nan)
    finite_anisotropy = anisotropy[np.isfinite(anisotropy)]
    for name, q in (("anisotropy_median", 50.0), ("anisotropy_p95", 95.0)):
        bundle.add(
            name,
            float(np.percentile(finite_anisotropy, q)) if finite_anisotropy.size else None,
            unit="ratio",
            reason="no finite anisotropy sample",
        )
    scale = np.sqrt(np.clip(s1 * s2, 0.0, None))
    finite_scale = scale[np.isfinite(scale)]
    bundle.add(
        "scale_median",
        float(np.median(finite_scale)) if finite_scale.size else None,
        unit="ratio",
        reason="no finite scale sample",
    )

    # Normalised bending energy: mean squared second derivative of the
    # dimensionless map, the discrete analogue of the thin-plate energy.
    d2_row = np.gradient(dv_drow, d_row, axis=0)
    d2_col = np.gradient(dv_dcol, d_col, axis=1)
    d2_cross = np.gradient(dv_drow, d_col, axis=1)
    energy = d2_row**2 + 2.0 * d2_cross**2 + d2_col**2
    finite_energy = energy[np.isfinite(energy)]
    bundle.add(
        "bending_energy",
        float(finite_energy.mean()) if finite_energy.size else None,
        unit="squared curvature",
        reason="no finite curvature sample",
    )
    return bundle


register_signal(
    SignalSpec(
        family="D",
        description="warp plausibility: Jacobian, folding, bending energy, scale and shear",
        requires=("forward_transform",),
        compute=compute_warp_plausibility,
        definition_version="1",
        cost_class="cheap",
        notes=(
            "Evaluated on a moving-frame lattice with both frames normalised by their "
            "own working diagonal, so Jacobians and curvatures are dimensionless. "
            "Plausible geometry is not evidence of correctness."
        ),
    )
)
