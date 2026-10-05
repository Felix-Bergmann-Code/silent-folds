"""Addressable robust transform fitting for the study pipeline blocks.

Both common-block pipelines use *this* fitter, so ``xfeat_h`` and ``sp_lg_h``
differ only in correspondence extraction while the transform family and the
fitting procedure are held fixed. It is a **WarpAudit adapter**, not a
published system, and it is implemented here rather than delegated to OpenCV
for one scientific reason: E1 must be able to hold the correspondence set
fixed while varying only the RANSAC seed, and vice versa (§7.2a). That
decomposition is impossible with a fitter whose randomness cannot be
addressed. The full study additionally fits a regularised thin-plate spline to
a seeded robust homography consensus, separating transform-family adequacy
from correspondence extraction.

Numerical conventions:

* correspondences are Hartley-normalised before the DLT solve, so the design
  matrix conditioning that Family B reports is a property of the geometry
  rather than of the pixel scale;
* a fit is rejected as ``degenerate_fit`` when fewer than four distinct
  correspondences survive, when the normalised design matrix is
  rank-deficient, or when the resulting matrix is not invertible;
* every rejection is an explicit status, never a silently returned identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.transforms import (
    AffineTransform,
    HomographyTransform,
    ThinPlateSplineTransform,
    Transform,
)
from ..types import RegistrationStatus

__all__ = [
    "FitResult",
    "FittingPolicy",
    "condition_number",
    "dlt_homography",
    "fit_homography",
    "fit_correspondences",
    "fit_tps_on_consensus",
    "normalise_points",
]

MIN_CORRESPONDENCES = 4


@dataclass(frozen=True)
class FittingPolicy:
    """Frozen fitting hyperparameters, hashed into the pipeline version."""

    threshold_px: float = 3.0
    max_iters: int = 10_000
    confidence: float = 0.9999
    min_matches: int = MIN_CORRESPONDENCES
    refine_on_inliers: bool = True
    name: str = "warpaudit_ransac_dlt"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "threshold_px": self.threshold_px,
            "max_iters": self.max_iters,
            "confidence": self.confidence,
            "min_matches": self.min_matches,
            "refine_on_inliers": self.refine_on_inliers,
        }


@dataclass
class FitResult:
    status: RegistrationStatus
    transform: Transform | None
    inlier_mask: np.ndarray | None
    n_iterations: int = 0
    residuals_px: np.ndarray | None = None
    design_condition: float = float("nan")
    diagnostics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is RegistrationStatus.OK


def normalise_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: centroid at the origin, mean distance sqrt(2).

    Returns ``(normalised_points, T)`` with ``normalised = T @ homogeneous``.
    """
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    mean_dist = float(np.mean(np.linalg.norm(centred, axis=1)))
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    T = np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]]
    )
    return centred * scale, T


def dlt_homography(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Normalised DLT solve. Returns ``(H, condition_number)``.

    ``H`` is ``None`` when the normalised design matrix is rank-deficient,
    which is the degenerate case §7.2 requires to be handled explicitly rather
    than producing a near-singular matrix.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < MIN_CORRESPONDENCES:
        return None, float("nan")

    ns, Ts = normalise_points(src)
    nd, Td = normalise_points(dst)

    n = len(ns)
    A = np.zeros((2 * n, 9), dtype=np.float64)
    x, y = ns[:, 0], ns[:, 1]
    u, v = nd[:, 0], nd[:, 1]
    A[0::2, 0] = -x
    A[0::2, 1] = -y
    A[0::2, 2] = -1.0
    A[0::2, 6] = u * x
    A[0::2, 7] = u * y
    A[0::2, 8] = u
    A[1::2, 3] = -x
    A[1::2, 4] = -y
    A[1::2, 5] = -1.0
    A[1::2, 6] = v * x
    A[1::2, 7] = v * y
    A[1::2, 8] = v

    try:
        # One full decomposition provides both the singular values used for
        # rank checks and the right-null vector used by DLT. ``full_matrices``
        # must remain true for the minimal 8x9 system: its ninth right-singular
        # vector spans the one-dimensional null space.
        _, s, Vt = np.linalg.svd(A, full_matrices=True)
    except np.linalg.LinAlgError:
        return None, float("nan")

    # A is (2n, 9) and its null space must be exactly one-dimensional, i.e.
    # rank 8. With n = 4 the matrix has only 8 rows, so `s` holds 8 values and
    # the last of them is the rank-8 witness; with n > 4 it holds 9 and the
    # witness is the second to last. Indexing from the front avoids that
    # asymmetry entirely.
    if len(s) < 8:
        return None, float("inf")
    cond = float(s[0] / s[7]) if s[7] > 0 else float("inf")
    if s[7] <= 1e-12 * s[0]:
        return None, cond

    H_norm = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H_norm @ Ts
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        return None, cond
    return H / H[2, 2], cond


def condition_number(points: np.ndarray) -> float:
    """Conditioning of the coordinate-normalised design (Family B feature)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return float("nan")
    normalised, _ = normalise_points(pts)
    design = np.concatenate([normalised, np.ones((len(normalised), 1))], axis=1)
    try:
        s = np.linalg.svd(design, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")


def _symmetric_residuals(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Forward transfer error in destination pixels."""
    T = HomographyTransform(H)
    mapped = T.apply(src)
    err = np.linalg.norm(mapped - dst, axis=1)
    return np.where(np.isfinite(err), err, np.inf)


def fit_homography(
    src: np.ndarray,
    dst: np.ndarray,
    policy: FittingPolicy | None = None,
    *,
    seed: int = 0,
) -> FitResult:
    """Seeded RANSAC + normalised DLT, refined on the inlier set.

    ``seed`` addresses the sampling randomness alone. Calling this twice with
    the same correspondences and the same seed returns bitwise-identical
    parameters on CPU, which is what makes the E1 seed-only control of §7.2a
    measurable.
    """
    policy = policy or FittingPolicy()
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("correspondences must be matching (N, 2) arrays")

    n = len(src)
    if n == 0:
        return FitResult(
            RegistrationStatus.NO_MATCHES, None, None, diagnostics={"reason": "no correspondences"}
        )
    if n < policy.min_matches:
        return FitResult(
            RegistrationStatus.NO_MATCHES,
            None,
            None,
            diagnostics={"reason": f"{n} correspondences < min_matches {policy.min_matches}"},
        )
    if len(np.unique(src, axis=0)) < MIN_CORRESPONDENCES:
        return FitResult(
            RegistrationStatus.DEGENERATE_FIT,
            None,
            None,
            diagnostics={"reason": "fewer than four distinct source points"},
        )

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    iterations = 0
    max_iters = int(policy.max_iters)

    while iterations < max_iters:
        iterations += 1
        sample = rng.choice(n, size=MIN_CORRESPONDENCES, replace=False)
        H, _ = dlt_homography(src[sample], dst[sample])
        if H is None:
            continue
        residuals = _symmetric_residuals(H, src, dst)
        inliers = residuals <= policy.threshold_px
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
            # Adaptive stopping at the declared confidence.
            w = max(count / n, 1e-9)
            denom = np.log1p(-min(w**MIN_CORRESPONDENCES, 1 - 1e-12))
            if denom < 0:
                needed = np.log1p(-policy.confidence) / denom
                max_iters = int(min(max_iters, max(1, np.ceil(needed))))

    if best_count < MIN_CORRESPONDENCES:
        return FitResult(
            RegistrationStatus.DEGENERATE_FIT,
            None,
            None,
            n_iterations=iterations,
            diagnostics={"reason": f"largest consensus set has {best_count} correspondences"},
        )

    fit_src, fit_dst = (
        (src[best_inliers], dst[best_inliers]) if policy.refine_on_inliers else (src, dst)
    )
    H, cond = dlt_homography(fit_src, fit_dst)
    if H is None:
        return FitResult(
            RegistrationStatus.DEGENERATE_FIT,
            None,
            None,
            n_iterations=iterations,
            design_condition=cond,
            diagnostics={"reason": "rank-deficient refit on the consensus set"},
        )

    transform = HomographyTransform(H)
    if not transform.is_valid():
        return FitResult(
            RegistrationStatus.INVALID_TRANSFORM,
            None,
            None,
            n_iterations=iterations,
            design_condition=cond,
            diagnostics={"reason": "fitted matrix is singular or non-finite"},
        )

    residuals = _symmetric_residuals(H, src, dst)
    return FitResult(
        status=RegistrationStatus.OK,
        transform=transform,
        inlier_mask=best_inliers,
        n_iterations=iterations,
        residuals_px=residuals,
        design_condition=cond,
        diagnostics={
            "n_inliers": int(best_inliers.sum()),
            "inlier_ratio": float(best_inliers.mean()),
            "policy": policy.as_dict(),
            "seed": int(seed),
        },
    )


def _normalisation_transform(points: np.ndarray) -> AffineTransform:
    normalised, matrix = normalise_points(points)
    del normalised
    return AffineTransform(matrix[:2])


def fit_tps_on_consensus(
    src: np.ndarray,
    dst: np.ndarray,
    policy: FittingPolicy | None = None,
    *,
    seed: int = 0,
    regularisation: float = 1e-3,
) -> FitResult:
    """Robust homography consensus followed by a regularised TPS refit."""
    if regularisation < 0:
        raise ValueError("TPS regularisation must be non-negative")
    consensus = fit_homography(src, dst, policy, seed=seed)
    if not consensus.ok or consensus.inlier_mask is None:
        return consensus
    keep = consensus.inlier_mask
    source = np.asarray(src, dtype=float)[keep]
    target = np.asarray(dst, dtype=float)[keep]
    if len(source) < 6:
        return FitResult(
            RegistrationStatus.DEGENERATE_FIT,
            None,
            keep,
            n_iterations=consensus.n_iterations,
            diagnostics={"reason": f"TPS consensus has {len(source)} points; need at least six"},
        )
    source_norm = _normalisation_transform(source)
    target_norm = _normalisation_transform(target)
    x = source_norm.apply(source)
    y = target_norm.apply(target)
    diff = x[:, None, :] - x[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", diff, diff)
    kernel = ThinPlateSplineTransform._kernel(r2)
    P = np.column_stack((np.ones(len(x)), x))
    system = np.block(
        [[kernel + regularisation * np.eye(len(x)), P], [P.T, np.zeros((3, 3))]]
    )
    rhs = np.vstack((y, np.zeros((3, 2))))
    try:
        solution = np.linalg.solve(system, rhs)
        denormalisation = AffineTransform(np.linalg.inv(target_norm.as_homography().matrix)[:2])
        transform = ThinPlateSplineTransform(
            control_points=x,
            weights=solution[: len(x)],
            affine=solution[len(x) :],
            normalisation=source_norm,
            denormalisation=denormalisation,
            regularisation=regularisation,
        )
    except (np.linalg.LinAlgError, ValueError):
        return FitResult(
            RegistrationStatus.DEGENERATE_FIT,
            None,
            keep,
            n_iterations=consensus.n_iterations,
            diagnostics={"reason": "singular TPS consensus system"},
        )
    if not transform.is_valid():
        return FitResult(RegistrationStatus.INVALID_TRANSFORM, None, keep)
    residuals = np.linalg.norm(transform.apply(np.asarray(src, dtype=float)) - dst, axis=1)
    return FitResult(
        RegistrationStatus.OK,
        transform,
        keep,
        n_iterations=consensus.n_iterations,
        residuals_px=residuals,
        design_condition=float(np.linalg.cond(system)),
        diagnostics={
            "n_inliers": int(keep.sum()),
            "inlier_ratio": float(keep.mean()),
            "regularisation": regularisation,
            "seed": int(seed),
        },
    )


def fit_correspondences(
    src: np.ndarray,
    dst: np.ndarray,
    policy: FittingPolicy | None = None,
    *,
    seed: int = 0,
    transform_family: str = "homography",
    tps_regularisation: float = 1e-3,
) -> FitResult:
    if transform_family == "homography":
        return fit_homography(src, dst, policy, seed=seed)
    if transform_family == "tps":
        return fit_tps_on_consensus(
            src, dst, policy, seed=seed, regularisation=tps_regularisation
        )
    raise ValueError(f"unsupported correspondence transform family {transform_family!r}")
