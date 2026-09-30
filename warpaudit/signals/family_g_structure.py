"""Family G: structure agreement (specification §7.1).

One identical gradient/edge-agreement definition across every domain. The
spec is explicit that generic and semantic structure must stay separate: a
shared column name cannot make vessel, tissue, and edge Dice equivalent, so
nothing here is tuned per dataset, per modality, or per anatomy.

The edge set is defined by a fixed *quantile* of gradient magnitude inside the
common valid support rather than an absolute threshold. A fixed absolute
threshold would silently become a brightness and contrast detector across
domains with different intensity ranges, which is exactly the domain-specific
behaviour this family must not have.

Domain-specific vessel or tissue-mask agreement is deliberately absent. It
would need predicted masks available at deployment plus recorded segmentation
provenance (§7.1), and it belongs in its own separately named family.
"""

from __future__ import annotations

import numpy as np

from ..geometry.resample import warp_moving_to_fixed
from ..geometry.support import frame_content_mask
from ..types import FeatureBundle, SignalContext
from .family_a_appearance import _to_grey, gradient_ncc
from .registry import SignalSpec, register_signal

__all__ = ["compute_structure", "edge_dice", "gradient_orientation_agreement"]

#: Shared with family A: below this, a similarity statistic is noise.
_MIN_SUPPORT_PIXELS = 64
#: Frozen, identical for every domain (§7.1). Two quantiles, not a search.
_EDGE_QUANTILES: tuple[float, ...] = (0.80, 0.90)


def edge_dice(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray, quantile: float
) -> tuple[float, float, float]:
    """Dice of the two edge sets, plus each image's edge fraction.

    Both edge sets are defined by the same within-support quantile of gradient
    magnitude, so each image contributes the same number of edge pixels by
    construction and Dice measures *where* the edges are, not how many.
    """
    ga = np.hypot(*np.gradient(np.asarray(a, dtype=np.float64)))
    gb = np.hypot(*np.gradient(np.asarray(b, dtype=np.float64)))
    va, vb = ga[mask], gb[mask]
    if va.size < 2:
        return float("nan"), float("nan"), float("nan")
    ea = va >= np.quantile(va, quantile)
    eb = vb >= np.quantile(vb, quantile)
    total = int(ea.sum()) + int(eb.sum())
    dice = float(2.0 * np.sum(ea & eb) / total) if total else float("nan")
    return dice, float(ea.mean()), float(eb.mean())


def gradient_orientation_agreement(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean ``|cos(theta)|`` between gradient directions inside the mask.

    The absolute value is deliberate: a bright-on-dark structure in one
    modality can be dark-on-bright in the other, and an edge that agrees in
    orientation but flips in polarity is still the same structure.
    """
    gay, gax = np.gradient(np.asarray(a, dtype=np.float64))
    gby, gbx = np.gradient(np.asarray(b, dtype=np.float64))
    ax, ay = gax[mask], gay[mask]
    bx, by = gbx[mask], gby[mask]
    na = np.hypot(ax, ay)
    nb = np.hypot(bx, by)
    usable = (na > 0) & (nb > 0)
    if usable.sum() < 2:
        return float("nan")
    cosine = (ax[usable] * bx[usable] + ay[usable] * by[usable]) / (na[usable] * nb[usable])
    return float(np.mean(np.abs(np.clip(cosine, -1.0, 1.0))))


def compute_structure(ctx: SignalContext) -> FeatureBundle:
    bundle = FeatureBundle(family="G")
    transform = ctx.result.forward_moving_to_fixed
    moving = _to_grey(ctx.images["moving"])
    fixed = _to_grey(ctx.images["fixed"])
    coords = ctx.pair.coordinates

    warped = warp_moving_to_fixed(moving, transform, coords.moving, coords.fixed)
    support = warped.valid & frame_content_mask(coords.fixed)
    n_support = int(support.sum())
    bundle.add("structure_support_pixels", float(n_support), unit="pixels")

    names = ["gradient_magnitude_ncc", "gradient_orientation_agreement"]
    for quantile in _EDGE_QUANTILES:
        tag = f"{int(round(quantile * 100))}"
        names += [f"edge_dice_q{tag}", f"edge_fraction_warped_q{tag}", f"edge_fraction_fixed_q{tag}"]
    if n_support < _MIN_SUPPORT_PIXELS:
        reason = f"valid support has {n_support} pixel(s) (< {_MIN_SUPPORT_PIXELS})"
        for name in names:
            bundle.add(name, None, reason=reason)
        return bundle

    warped_grey = _to_grey(warped.image)
    bundle.add(
        "gradient_magnitude_ncc",
        gradient_ncc(warped_grey, fixed, support),
        unit="correlation",
    )
    bundle.add(
        "gradient_orientation_agreement",
        gradient_orientation_agreement(warped_grey, fixed, support),
        unit="mean |cos|",
    )
    for quantile in _EDGE_QUANTILES:
        tag = f"{int(round(quantile * 100))}"
        dice, fraction_warped, fraction_fixed = edge_dice(warped_grey, fixed, support, quantile)
        bundle.add(f"edge_dice_q{tag}", dice, unit="dice")
        bundle.add(f"edge_fraction_warped_q{tag}", fraction_warped, unit="fraction")
        bundle.add(f"edge_fraction_fixed_q{tag}", fraction_fixed, unit="fraction")
    return bundle


register_signal(
    SignalSpec(
        family="G",
        description="generic gradient and edge agreement, identical across domains",
        requires=("moving_image", "fixed_image", "forward_transform"),
        compute=compute_structure,
        definition_version="1",
        cost_class="moderate",
        notes=(
            "Edge sets use frozen within-support gradient quantiles, never an absolute "
            "threshold, so the definition does not become a contrast detector across "
            "domains. Semantic vessel/tissue agreement is a separate family by design."
        ),
    )
)
