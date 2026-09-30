"""Dataset loaders: images and metadata only (specification §4).

A loader returns :class:`PairListing` objects -- ids, paths, original
dimensions, grouping evidence, and *pointers* to annotation files. It never
returns landmark arrays. Annotations are read separately by
:mod:`warpaudit.data.annotations` inside the evaluation process.

Archive layouts are encoded only after inspection of the official archives. A
loader that cannot find its data raises
:class:`DatasetUnavailable` with the exact expected layout, because a silent
empty listing would let a downstream command report "0 pairs, all checks
passed".
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from PIL import Image

__all__ = [
    "DatasetUnavailable",
    "PairListing",
    "get_loader",
    "list_coph100_pairs",
    "list_fire_pairs",
    "list_indexed_pairs",
    "register_loader",
]

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
_COPH_STAGE_RE = re.compile(r"_S(?P<stage>\d+)_")


class DatasetUnavailable(FileNotFoundError):
    """Raised when a dataset root does not contain the expected layout."""


@dataclass(frozen=True)
class PairListing:
    """One registration pair before grouping and before annotations are read."""

    dataset_id: str
    pair_id: str
    moving_image_id: str
    fixed_image_id: str
    moving_path: Path
    fixed_path: Path
    moving_hw: tuple[int, int]
    fixed_hw: tuple[int, int]
    annotation_paths: tuple[Path, ...]
    annotation_kind: str = "landmarks"
    annotation_provenance: str = ""
    dataset_category: str = ""
    subject_id: str | None = None
    subject_basis: str = "patient"
    subject_evidence: str = ""
    extra: dict = field(default_factory=dict)

    def as_graph_edge(self) -> tuple[str, str, str]:
        return (self.pair_id, self.moving_image_id, self.fixed_image_id)


def _image_size(path: Path) -> tuple[int, int]:
    """(height, width) without decoding pixel data."""
    with Image.open(path) as im:
        width, height = im.size
    return int(height), int(width)


def _coph_exam_key(path: Path) -> tuple[int, str]:
    match = _COPH_STAGE_RE.search(path.stem)
    if match is None:
        raise DatasetUnavailable(f"COph100 examination has no '_S<stage>_' token: {path}")
    return int(match.group("stage")), path.name


# --------------------------------------------------------------------------
# FIRE [R10]
# --------------------------------------------------------------------------

#: FIRE pair names are ``<category><nn>`` with category in S (same), P
#: (partial overlap), A (anatomical difference). The category is a descriptive
#: stratum, never a label or a grouping unit.
_FIRE_PAIR_RE = re.compile(r"^(?P<category>[SPA])(?P<number>\d+)$", re.IGNORECASE)


def list_fire_pairs(root: str | Path, dataset_id: str = "FIRE") -> list[PairListing]:
    """List FIRE pairs.

    Expected layout in the official archive::

        <root>/Images/<pair>_1.jpg          # first image of the pair
        <root>/Images/<pair>_2.jpg          # second image of the pair
        <root>/Ground Truth/control_points_<pair>_1_2.txt

    Canonical direction: image ``_1`` is the MOVING image and ``_2`` is the
    FIXED image, matching column order ``x1 y1 x2 y2`` in the control-point
    file. The reverse direction is generated later as a derivative row that
    inherits this pair's group; it never becomes an extra independent sample
    (spec §8.2).

    Patient identity is NOT derivable from the pair name. ``subject_id``
    therefore stays ``None`` and grouping falls back to shared-image connected
    components (spec §4.1). If a patient mapping is obtained from the dataset
    authors, supply it through ``audit-data --subject-map`` rather than
    inventing one here.
    """
    root = Path(root)
    images_dir = next(
        (root / name for name in ("Images", "images") if (root / name).is_dir()), None
    )
    gt_dir = next(
        (
            root / name
            for name in ("Ground Truth", "Ground truth", "ground_truth", "GroundTruth")
            if (root / name).is_dir()
        ),
        None,
    )
    if images_dir is None or gt_dir is None:
        raise DatasetUnavailable(
            f"FIRE layout not found under {root}. Expected '<root>/Images' and "
            "'<root>/Ground Truth'. Download the archive from "
            "https://www.ics.forth.gr/cvrl/fire/ under its stated terms and record "
            "the checksum in DATA_LICENCES.md."
        )

    by_pair: dict[str, dict[str, Path]] = {}
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        stem = path.stem
        if "_" not in stem:
            continue
        pair, _, index = stem.rpartition("_")
        if index in ("1", "2") and _FIRE_PAIR_RE.match(pair):
            by_pair.setdefault(pair.upper(), {})[index] = path

    listings: list[PairListing] = []
    for pair_id in sorted(by_pair):
        sides = by_pair[pair_id]
        if set(sides) != {"1", "2"}:
            continue  # an incomplete pair is reported by audit-data, not guessed
        moving, fixed = sides["1"], sides["2"]
        gt = gt_dir / f"control_points_{pair_id}_1_2.txt"
        listings.append(
            PairListing(
                dataset_id=dataset_id,
                pair_id=f"{dataset_id}/{pair_id}",
                moving_image_id=f"{dataset_id}/{moving.stem}",
                fixed_image_id=f"{dataset_id}/{fixed.stem}",
                moving_path=moving,
                fixed_path=fixed,
                moving_hw=_image_size(moving),
                fixed_hw=_image_size(fixed),
                annotation_paths=(gt,) if gt.exists() else (),
                annotation_kind="landmarks",
                annotation_provenance="FIRE control_points, 10 correspondences per pair [R10]",
                dataset_category=pair_id[0].upper(),
                subject_id=None,  # never fabricated from the pair id (§4.1)
                subject_basis="patient",
                subject_evidence="",
            )
        )
    if not listings:
        raise DatasetUnavailable(f"no complete FIRE pairs found under {images_dir}")
    return listings


# --------------------------------------------------------------------------
# COph100 [R11]
# --------------------------------------------------------------------------


def list_coph100_pairs(root: str | Path, dataset_id: str = "COph100") -> list[PairListing]:
    """List all within-eye examination combinations from COph100 v1.

    The official deposit contains 100 eye directories, 324 examination JSONs,
    and ten LabelMe points per examination.  Its bundled extraction script
    maps directories such as ``002`` and ``002-1`` back to patient ``002``;
    this is deposited identity evidence, not an inference from pair names.
    Sorted examination order defines the canonical moving-to-fixed direction.
    """
    root = Path(root)
    eye_dirs = sorted(
        path
        for path in root.iterdir() if path.is_dir() and re.fullmatch(r"\d{3}(?:-1)?", path.name)
    ) if root.is_dir() else []
    if not eye_dirs:
        raise DatasetUnavailable(
            f"COph100 layout not found under {root}. Run `python -m warpaudit prepare-data "
            "--dataset COph100` using the official Figshare v1 archive."
        )

    listings: list[PairListing] = []
    for eye_dir in eye_dirs:
        annotations = sorted(eye_dir.glob("*.json"), key=_coph_exam_key)
        stages = [_coph_exam_key(path)[0] for path in annotations]
        if len(stages) != len(set(stages)):
            raise DatasetUnavailable(f"COph100 eye {eye_dir.name} repeats an acquisition stage")
        patient_id = eye_dir.name.split("-", 1)[0]
        for moving_ann, fixed_ann in combinations(annotations, 2):
            moving = moving_ann.with_suffix(".jpg")
            fixed = fixed_ann.with_suffix(".jpg")
            if not moving.is_file() or not fixed.is_file():
                raise DatasetUnavailable(
                    f"COph100 reconstructed JPEG absent for {moving_ann if not moving.is_file() else fixed_ann}"
                )
            local_pair_id = f"{eye_dir.name}/{moving.stem}__{fixed.stem}"
            listings.append(
                PairListing(
                    dataset_id=dataset_id,
                    pair_id=f"{dataset_id}/{local_pair_id}",
                    moving_image_id=f"{dataset_id}/{eye_dir.name}/{moving.stem}",
                    fixed_image_id=f"{dataset_id}/{eye_dir.name}/{fixed.stem}",
                    moving_path=moving,
                    fixed_path=fixed,
                    moving_hw=_image_size(moving),
                    fixed_hw=_image_size(fixed),
                    annotation_paths=(moving_ann, fixed_ann),
                    annotation_kind="landmarks",
                    annotation_provenance=(
                        "COph100 v1 LabelMe point annotations; common labels paired in "
                        "canonical examination order [R11]"
                    ),
                    subject_id=patient_id,
                    subject_basis="patient",
                    subject_evidence=(
                        "patient prefix and eye suffix supplied by the deposited "
                        "Copy_COph100_from_ROP.py destination mapping"
                    ),
                    extra={"eye_id": eye_dir.name, "patient_id": patient_id},
                )
            )
    if not listings:
        raise DatasetUnavailable(f"no COph100 within-eye examination pairs found under {root}")
    return listings


def list_indexed_pairs(root: str | Path, dataset_id: str) -> list[PairListing]:
    """Load an externally acquired dataset through a reviewed pair index.

    Access-controlled archives have changing layouts and must not be guessed.
    Place ``warpaudit_pairs.csv`` at the dataset root with columns
    ``pair_id,moving_path,fixed_path,annotation_path,subject_id``. Optional
    columns are ``category`` and ``annotation_provenance``. Paths are relative
    to the root; annotations use four numeric columns ``x_m y_m x_f y_f``.
    """
    root = Path(root).resolve()
    index = root / "warpaudit_pairs.csv"
    if not index.is_file():
        raise DatasetUnavailable(
            f"{dataset_id}: {index} is absent. Create the reviewed external-data index "
            "from the acquired archive; WarpAudit refuses to infer subjects or pairings."
        )
    required = {"pair_id", "moving_path", "fixed_path", "annotation_path", "subject_id"}
    listings: list[PairListing] = []
    with index.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise DatasetUnavailable(f"{index}: missing required columns {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            local_id = str(row["pair_id"]).strip()
            subject = str(row["subject_id"]).strip()
            if not local_id or not subject:
                raise DatasetUnavailable(f"{index}:{line}: pair_id and subject_id are required")
            indexed_paths = {}
            for column in ("moving_path", "fixed_path", "annotation_path"):
                relative = Path(str(row[column]).strip())
                if relative.is_absolute():
                    raise DatasetUnavailable(
                        f"{index}:{line}: {column} must be relative to the dataset root"
                    )
                candidate = (root / relative).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError as exc:
                    raise DatasetUnavailable(
                        f"{index}:{line}: {column} escapes the dataset root"
                    ) from exc
                indexed_paths[column] = candidate
            moving = indexed_paths["moving_path"]
            fixed = indexed_paths["fixed_path"]
            annotation = indexed_paths["annotation_path"]
            for path in (moving, fixed, annotation):
                if not path.is_file():
                    raise DatasetUnavailable(f"{index}:{line}: indexed file is absent: {path}")
            listings.append(
                PairListing(
                    dataset_id=dataset_id,
                    pair_id=f"{dataset_id}/{local_id}",
                    moving_image_id=f"{dataset_id}/{Path(row['moving_path']).as_posix()}",
                    fixed_image_id=f"{dataset_id}/{Path(row['fixed_path']).as_posix()}",
                    moving_path=moving,
                    fixed_path=fixed,
                    moving_hw=_image_size(moving),
                    fixed_hw=_image_size(fixed),
                    annotation_paths=(annotation,),
                    annotation_kind="landmarks",
                    annotation_provenance=str(row.get("annotation_provenance") or "reviewed external landmark index"),
                    dataset_category=str(row.get("category") or ""),
                    subject_id=subject,
                    subject_basis=str(row.get("subject_basis") or (
                        "specimen" if dataset_id == "ANHIR" else "patient"
                    )),
                    subject_evidence="reviewed warpaudit_pairs.csv",
                )
            )
    if not listings:
        raise DatasetUnavailable(f"{index}: contains no data rows")
    if len({item.pair_id for item in listings}) != len(listings):
        raise DatasetUnavailable(f"{index}: pair_id values are not unique")
    return listings


# --------------------------------------------------------------------------

_LOADERS: dict[str, Callable[[str | Path, str], list[PairListing]]] = {
    "FIRE": list_fire_pairs,
    "COph100": list_coph100_pairs,
    "ANHIR": list_indexed_pairs,
    "AN200": list_indexed_pairs,
    "MEMO": list_indexed_pairs,
    "MultiRegHistology": list_indexed_pairs,
    "MultiRegCytology": list_indexed_pairs,
}


def register_loader(
    dataset_id: str, loader: Callable[[str | Path, str], list[PairListing]]
) -> None:
    _LOADERS[dataset_id] = loader


def get_loader(dataset_id: str) -> Callable[[str | Path, str], list[PairListing]]:
    try:
        return _LOADERS[dataset_id]
    except KeyError as exc:
        raise KeyError(
            f"no loader registered for dataset {dataset_id!r}; " f"available: {sorted(_LOADERS)}"
        ) from exc
