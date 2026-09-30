"""Transform families and their composition (specification §5.2).

Conventions, fixed once here and relied on everywhere else:

* A transform maps **moving** pixel-centre coordinates to **fixed** pixel-centre
  coordinates. This is the *forward* direction ``T_m2f`` used for point
  evaluation. Backward sampling for raster warping is a separate operation
  (see ``warpaudit.geometry.resample``) and never reuses this object silently.
* Points are ``(N, 2)`` float arrays of ``(x, y)``.
* A composition ``g.after(f)`` means "apply ``f`` first, then ``g``", i.e.
  ``(g . f)(x)``.
* A transform is *valid* only when every parameter is finite and, for
  projective maps, the fit is not rank-deficient. An invalid map is a status,
  not a large number (spec §6.2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = [
    "AffineTransform",
    "ComposedTransform",
    "HomographyTransform",
    "ThinPlateSplineTransform",
    "Transform",
    "as_points",
    "identity",
]


def as_points(points: np.ndarray) -> np.ndarray:
    """Validate and normalise an ``(N, 2)`` point array."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1 and pts.shape[0] == 2:
        pts = pts[None, :]
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (N, 2), got shape {pts.shape}")
    return pts


class Transform(ABC):
    """A point map from moving to fixed coordinates."""

    family: str = "abstract"

    @abstractmethod
    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map ``(N, 2)`` moving points to ``(N, 2)`` fixed points.

        Points that map to a non-finite location (e.g. a projective point at
        infinity) return ``nan``; the caller decides whether that is an
        invalid transform or a single unmappable landmark. An out-of-image but
        *finite* mapping is a geometric error, not a reason to omit a landmark
        (spec §6.2).
        """

    @abstractmethod
    def is_valid(self) -> bool:
        """True when the parameters are finite and non-degenerate."""

    def after(self, other: Transform) -> Transform:
        """Return ``self . other`` (``other`` applied first)."""
        return ComposedTransform((other, self))

    def jacobian(self, points: np.ndarray, *, eps: float = 1e-3) -> np.ndarray:
        """Central-difference Jacobians, shape ``(N, 2, 2)``.

        Analytic forms override this. ``eps`` is in the transform's own input
        units, so callers working in normalised coordinates must pass a
        normalised step (spec §7.3: diagonal division alone is insufficient
        for Jacobians).
        """
        pts = as_points(points)
        jac = np.empty((pts.shape[0], 2, 2), dtype=np.float64)
        for axis in (0, 1):
            step = np.zeros((1, 2))
            step[0, axis] = eps
            plus = self.apply(pts + step)
            minus = self.apply(pts - step)
            jac[:, :, axis] = (plus - minus) / (2.0 * eps)
        return jac

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.apply(points)


@dataclass(frozen=True)
class HomographyTransform(Transform):
    """Projective map given by a 3x3 matrix in the transform's own frame."""

    matrix: np.ndarray
    family: str = "homography"

    def __post_init__(self) -> None:
        H = np.asarray(self.matrix, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"homography must be 3x3, got {H.shape}")
        object.__setattr__(self, "matrix", H)

    def is_valid(self) -> bool:
        H = self.matrix
        if not np.isfinite(H).all():
            return False
        det = float(np.linalg.det(H))
        if not np.isfinite(det) or det == 0.0:
            return False
        # Reject numerically rank-deficient fits: a projective matrix whose
        # smallest singular value is negligible relative to the largest cannot
        # be inverted meaningfully and is reported as a degenerate fit.
        s = np.linalg.svd(H, compute_uv=False)
        return bool(s[-1] > 1e-10 * s[0])

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = as_points(points)
        homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        out = homo @ self.matrix.T
        w = out[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = out[:, :2] / w[:, None]
        xy[~np.isfinite(xy).all(axis=1)] = np.nan
        return xy

    def inverse(self) -> HomographyTransform:
        """Analytic inverse.

        This is the *inverse map*. It is never a substitute for an
        independently estimated reverse registration (spec §5.2); the two are
        kept in different fields throughout the cache schema.
        """
        if not self.is_valid():
            raise ValueError("cannot invert a degenerate homography")
        return HomographyTransform(np.linalg.inv(self.matrix))

    def jacobian(self, points: np.ndarray, *, eps: float = 1e-3) -> np.ndarray:
        pts = as_points(points)
        H = self.matrix
        homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        out = homo @ H.T
        w = out[:, 2]
        u, v = out[:, 0], out[:, 1]
        jac = np.empty((pts.shape[0], 2, 2), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            for axis in (0, 1):
                du, dv, dw = H[0, axis], H[1, axis], H[2, axis]
                jac[:, 0, axis] = (du * w - u * dw) / w**2
                jac[:, 1, axis] = (dv * w - v * dw) / w**2
        jac[~np.isfinite(jac).all(axis=(1, 2))] = np.nan
        return jac


@dataclass(frozen=True)
class AffineTransform(Transform):
    """Affine map stored as a 2x3 matrix ``[A | t]``."""

    matrix: np.ndarray
    family: str = "affine"

    def __post_init__(self) -> None:
        M = np.asarray(self.matrix, dtype=np.float64)
        if M.shape == (3, 3):
            if not np.allclose(M[2], [0.0, 0.0, 1.0]):
                raise ValueError("3x3 affine must have bottom row [0, 0, 1]")
            M = M[:2]
        if M.shape != (2, 3):
            raise ValueError(f"affine must be 2x3 or 3x3, got {M.shape}")
        object.__setattr__(self, "matrix", M)

    def is_valid(self) -> bool:
        M = self.matrix
        return bool(np.isfinite(M).all() and abs(float(np.linalg.det(M[:, :2]))) > 1e-12)

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = as_points(points)
        return pts @ self.matrix[:, :2].T + self.matrix[:, 2]

    def as_homography(self) -> HomographyTransform:
        H = np.eye(3)
        H[:2] = self.matrix
        return HomographyTransform(H)

    def jacobian(self, points: np.ndarray, *, eps: float = 1e-3) -> np.ndarray:
        pts = as_points(points)
        return np.repeat(self.matrix[None, :, :2], pts.shape[0], axis=0)


@dataclass(frozen=True)
class ComposedTransform(Transform):
    """Ordered composition. ``steps[0]`` is applied first."""

    steps: tuple[Transform, ...]
    family: str = "composed"

    def is_valid(self) -> bool:
        return all(s.is_valid() for s in self.steps)

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = as_points(points)
        for step in self.steps:
            pts = step.apply(pts)
        return pts

    def collapse(self) -> Transform:
        """Multiply out a chain of purely projective/affine steps.

        Returns ``self`` unchanged when any step is non-parametric, so a TPS
        in the chain is never silently linearised.
        """
        mats: list[np.ndarray] = []
        for step in self.steps:
            if isinstance(step, HomographyTransform):
                mats.append(step.matrix)
            elif isinstance(step, AffineTransform):
                mats.append(step.as_homography().matrix)
            else:
                return self
        out = np.eye(3)
        for M in mats:  # steps[0] first => left-multiply successively
            out = M @ out
        return HomographyTransform(out)


@dataclass(frozen=True)
class ThinPlateSplineTransform(Transform):
    """Regularised TPS fitted in NORMALISED coordinates (spec §5.1).

    ``normalisation`` is the affine that maps the transform's input frame to
    the normalised frame in which the spline was fitted; ``denormalisation``
    maps spline output back. Keeping them explicit prevents the scale-mixing
    error that a bare "divide by the diagonal" would introduce.

    This class is the *WarpAudit adapter* transform used for the candidate
    non-rigid block; it is not a published system.
    """

    control_points: np.ndarray  # (K, 2) normalised source points
    weights: np.ndarray  # (K, 2) TPS weights
    affine: np.ndarray  # (3, 2) affine part [a0; ax; ay]
    normalisation: AffineTransform
    denormalisation: AffineTransform
    regularisation: float = 0.0
    family: str = "tps"

    def __post_init__(self) -> None:
        cp = np.asarray(self.control_points, dtype=np.float64)
        w = np.asarray(self.weights, dtype=np.float64)
        a = np.asarray(self.affine, dtype=np.float64)
        if cp.ndim != 2 or cp.shape[1] != 2:
            raise ValueError("control_points must be (K, 2)")
        if w.shape != cp.shape:
            raise ValueError("weights must match control_points")
        if a.shape != (3, 2):
            raise ValueError("affine part must be (3, 2)")
        object.__setattr__(self, "control_points", cp)
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "affine", a)

    def is_valid(self) -> bool:
        return bool(
            np.isfinite(self.control_points).all()
            and np.isfinite(self.weights).all()
            and np.isfinite(self.affine).all()
            and len(np.unique(self.control_points, axis=0)) >= 3
        )

    @staticmethod
    def _kernel(r2: np.ndarray) -> np.ndarray:
        # U(r) = r^2 log r^2, continuous at r = 0.
        out = np.zeros_like(r2)
        nz = r2 > 0
        out[nz] = r2[nz] * np.log(r2[nz])
        return out

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = self.normalisation.apply(as_points(points))
        diff = pts[:, None, :] - self.control_points[None, :, :]
        r2 = np.einsum("nkd,nkd->nk", diff, diff)
        U = self._kernel(r2)
        out = self.affine[0] + pts @ self.affine[1:] + U @ self.weights
        return self.denormalisation.apply(out)


def identity() -> HomographyTransform:
    return HomographyTransform(np.eye(3))
