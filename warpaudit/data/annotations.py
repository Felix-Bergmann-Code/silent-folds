"""Restricted annotation loading (specification §7.3, §12.3).

This module is the *only* legitimate route to ground truth, and nothing in
:mod:`warpaudit.signals` or :mod:`warpaudit.registration` may import it. That
rule is enforced by ``tests/test_label_access.py``, which walks the import
graph rather than trusting review.

Annotations are returned in ORIGINAL image coordinates. Conversion into a
working frame happens inside the evaluation process only, using the pair's own
recorded ``A_m``/``A_f``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..types import AnnotationPayload, EvaluationAnnotation

__all__ = [
    "AnnotationStore",
    "load_coph100_control_points",
    "load_fire_control_points",
    "load_labelme_points",
]


def load_fire_control_points(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one FIRE ``control_points_*.txt`` file.

    Format confirmed against the official archive: whitespace-separated rows
    of four numbers ``x1 y1 x2 y2`` in ORIGINAL pixel coordinates, where
    ``(x1, y1)`` is in the first-named image of the pair and ``(x2, y2)`` in
    the second. Which of the two is the moving image is decided by the loader's
    declared canonical direction, not guessed here.
    """
    raw = np.loadtxt(str(path), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] != 4:
        raise ValueError(f"{path}: expected four columns per landmark, got {raw.shape[1]}")
    return raw[:, 0:2].copy(), raw[:, 2:4].copy()


def load_labelme_points(path: str | Path) -> dict[str, np.ndarray]:
    """Read point shapes from one COph100 LabelMe JSON in original coordinates.

    Three official-v1 files repeat one numeric label while omitting another.
    Occurrence suffixes retain this defect explicitly (``"1#2"``) rather than
    overwriting a point.  Pair construction then uses only common keys, which
    yields nine points for seven affected pairs and ten for the other 484.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid LabelMe JSON: {exc}") from exc
    shapes = payload.get("shapes") if isinstance(payload, dict) else None
    if not isinstance(shapes, list):
        raise ValueError(f"{path}: LabelMe shapes must be a list")
    seen: Counter[str] = Counter()
    points: dict[str, np.ndarray] = {}
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict) or shape.get("shape_type") != "point":
            continue
        label = str(shape.get("label", "")).strip()
        raw_points = shape.get("points")
        if not label or not isinstance(raw_points, list) or len(raw_points) != 1:
            raise ValueError(f"{path}: malformed point shape at index {index}")
        point = np.asarray(raw_points[0], dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError(f"{path}: point {label!r} must be one finite (x, y) coordinate")
        seen[label] += 1
        key = label if seen[label] == 1 else f"{label}#{seen[label]}"
        points[key] = point
    if not points:
        raise ValueError(f"{path}: no point annotations found")
    return points


def _label_sort_key(label: str) -> tuple[int, int, str]:
    base, _, occurrence = label.partition("#")
    if base.isdigit():
        return int(base), int(occurrence or 1), ""
    return 10**9, int(occurrence or 1), base


def load_coph100_control_points(
    moving_path: str | Path, fixed_path: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Pair common COph100 labels in canonical moving-to-fixed order."""
    moving = load_labelme_points(moving_path)
    fixed = load_labelme_points(fixed_path)
    labels = sorted(set(moving) & set(fixed), key=_label_sort_key)
    if len(labels) < 4:
        raise ValueError(
            f"COph100 pair has only {len(labels)} common labelled points: "
            f"{moving_path}, {fixed_path}"
        )
    return (
        np.stack([moving[label] for label in labels]),
        np.stack([fixed[label] for label in labels]),
    )


class AnnotationStore:
    """In-memory annotation table keyed by ``(dataset_id, pair_id)``."""

    def __init__(self, annotations: Iterable[EvaluationAnnotation] = ()) -> None:
        self._by_key: dict[tuple[str, str], EvaluationAnnotation] = {}
        for ann in annotations:
            self.add(ann)

    def add(self, annotation: EvaluationAnnotation) -> None:
        key = (annotation.dataset_id, annotation.pair_id)
        if key in self._by_key:
            raise ValueError(f"duplicate annotation for {key}")
        self._by_key[key] = annotation

    def get(self, dataset_id: str, pair_id: str) -> EvaluationAnnotation | None:
        return self._by_key.get((dataset_id, pair_id))

    def require(self, dataset_id: str, pair_id: str) -> EvaluationAnnotation:
        ann = self.get(dataset_id, pair_id)
        if ann is None:
            raise KeyError(f"no annotation for {(dataset_id, pair_id)}")
        return ann

    def __len__(self) -> int:
        return len(self._by_key)

    def keys(self) -> list[tuple[str, str]]:
        return sorted(self._by_key)

    @staticmethod
    def landmark_annotation(
        dataset_id: str,
        pair_id: str,
        points_moving: np.ndarray,
        points_fixed: np.ndarray,
        provenance: dict,
    ) -> EvaluationAnnotation:
        return EvaluationAnnotation(
            dataset_id=dataset_id,
            pair_id=pair_id,
            kind="landmarks",
            payload=AnnotationPayload(points_moving=points_moving, points_fixed=points_fixed),
            provenance=provenance,
        )

    @staticmethod
    def homography_annotation(
        dataset_id: str, pair_id: str, homography: np.ndarray, provenance: dict
    ) -> EvaluationAnnotation:
        return EvaluationAnnotation(
            dataset_id=dataset_id,
            pair_id=pair_id,
            kind="homography",
            payload=AnnotationPayload(homography=homography),
            provenance=provenance,
        )
