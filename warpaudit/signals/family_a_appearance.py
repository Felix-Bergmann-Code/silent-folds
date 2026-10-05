"""Family A: appearance agreement on documented valid support (spec §7.1).

NCC, SSIM, normalised mutual information, and gradient NCC, each computed over
the intersection of fixed support and warped moving support with padding
excluded (§5.3). Every score travels with the overlap fraction, because a
similarity computed on 3% of the image is not the same measurement as one
computed on 80%.

Multimodal failure of appearance metrics is a hypothesis to test here, not an
assumption to build in. The optional frozen-DINO patch agreement is not
implemented in the pilot: it adds compute and requires a fixed checkpoint and
aggregation, and §7.1 places it behind the pilot core.
"""

from __future__ import annotations

import numpy as np

from ..geometry.resample import warp_moving_to_fixed
from ..geometry.support import frame_content_mask
from ..types import FeatureBundle, SignalContext
from .registry import SignalSpec, register_signal

__all__ = ["compute_appearance", "gradient_ncc", "ncc", "normalised_mutual_information", "ssim"]

_MIN_SUPPORT_PIXELS = 64


def _to_grey(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        # Rec. 601 luma; recorded here so the colour convention is not implicit.
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation over the supplied samples."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else float("nan")


def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray, *, data_range: float = 1.0) -> float:
    """Global SSIM restricted to ``mask``.

    A masked *global* SSIM is used rather than the usual windowed mean,
    because a sliding window straddling the support boundary would mix real
    and padded pixels. The definition version records this choice.
    """
    a = np.asarray(a, dtype=np.float64)[mask]
    b = np.asarray(b, dtype=np.float64)[mask]
    if a.size < 2:
        return float("nan")
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    cov = float(np.cov(a, b, ddof=1)[0, 1])
    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (va + vb + C2)
    return float(num / den) if den > 0 else float("nan")


def normalised_mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """NMI = (H(A) + H(B)) / H(A, B), the standard registration convention."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2:
        return float("nan")
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist / hist.sum() if hist.sum() > 0 else hist
    px, py = pxy.sum(axis=1), pxy.sum(axis=0)

    def entropy(p: np.ndarray) -> float:
        nz = p[p > 0]
        return float(-np.sum(nz * np.log(nz)))

    h_joint = entropy(pxy)
    return float((entropy(px) + entropy(py)) / h_joint) if h_joint > 0 else float("nan")


def gradient_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """NCC of gradient magnitude -- the one generic structure definition (§7.1 G).

    Deliberately identical across all domains. Domain-specific vessel or
    tissue-mask agreement is a *separate* feature with a separate name; a
    shared column name cannot make vessel, tissue, and edge Dice equivalent.
    """
    ga = np.hypot(*np.gradient(np.asarray(a, dtype=np.float64)))
    gb = np.hypot(*np.gradient(np.asarray(b, dtype=np.float64)))
    return ncc(ga[mask], gb[mask])


def compute_appearance(ctx: SignalContext) -> FeatureBundle:
    bundle = FeatureBundle(family="A")
    transform = ctx.result.forward_moving_to_fixed
    moving = _to_grey(ctx.images["moving"])
    fixed = _to_grey(ctx.images["fixed"])
    coords = ctx.pair.coordinates

    warped = warp_moving_to_fixed(moving, transform, coords.moving, coords.fixed)
    support = warped.valid & frame_content_mask(coords.fixed)
    n_support = int(support.sum())
    total = support.size

    bundle.add("overlap_fraction", n_support / total if total else None, unit="fraction")
    bundle.add("support_pixels", float(n_support), unit="pixels")

    if n_support < _MIN_SUPPORT_PIXELS:
        reason = f"valid support has {n_support} pixel(s) (< {_MIN_SUPPORT_PIXELS})"
        for name in ("ncc", "ssim", "nmi", "gradient_ncc"):
            bundle.add(name, None, reason=reason)
        return bundle

    warped_grey = _to_grey(warped.image)
    a, b = warped_grey[support], fixed[support]
    data_range = float(max(np.ptp(fixed[support]), 1e-9))

    bundle.add("ncc", ncc(a, b), unit="correlation")
    bundle.add("ssim", ssim(warped_grey, fixed, support, data_range=data_range), unit="ssim")
    bundle.add("nmi", normalised_mutual_information(a, b), unit="ratio")
    bundle.add("gradient_ncc", gradient_ncc(warped_grey, fixed, support), unit="correlation")
    return bundle


register_signal(
    SignalSpec(
        family="A",
        description="appearance agreement on documented valid support",
        requires=("moving_image", "fixed_image", "forward_transform"),
        compute=compute_appearance,
        definition_version="1",
        cost_class="moderate",
        notes=(
            "SSIM is a masked global statistic, not a windowed mean, so no window "
            "straddles the support boundary. Optional frozen-DINO agreement is out of "
            "the pilot core (§7.1)."
        ),
    )
)
