"""Checksum-verified reconstruction of the two pilot datasets.

The repository never redistributes dataset bytes.  This module turns official
archives obtained by the user into the on-disk layouts consumed by the data
loaders.  Preparation is deliberately separate from auditing: ``audit-data``
is read-only and can therefore be rerun without modifying restricted data.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image

from ..cache.hashing import file_digest
from ..cache.store import atomic_write_bytes

__all__ = [
    "ArchiveSpec",
    "PreparationError",
    "download_archive",
    "prepare_coph100",
    "prepare_fire",
    "verify_archive",
]


class PreparationError(RuntimeError):
    """Raised when source bytes do not satisfy the recorded archive contract."""


@dataclass(frozen=True)
class ArchiveSpec:
    dataset_id: str
    url: str
    sha256: str
    size_bytes: int = 0


def verify_archive(path: str | Path, spec: ArchiveSpec) -> None:
    """Require exact size (when known) and SHA-256 before extraction."""
    path = Path(path)
    if not path.is_file():
        raise PreparationError(f"{spec.dataset_id}: archive is absent: {path}")
    actual_size = path.stat().st_size
    if spec.size_bytes and actual_size != spec.size_bytes:
        raise PreparationError(
            f"{spec.dataset_id}: archive size mismatch: expected {spec.size_bytes}, "
            f"got {actual_size}"
        )
    actual_digest = file_digest(path)
    if actual_digest.lower() != spec.sha256.lower():
        raise PreparationError(
            f"{spec.dataset_id}: archive SHA-256 mismatch: expected {spec.sha256}, "
            f"got {actual_digest}"
        )


def _partial_path(path: Path) -> Path:
    """Where an interrupted download of ``path`` keeps the bytes it already has."""
    return path.with_name(path.name + ".partial")


def download_archive(
    path: str | Path,
    spec: ArchiveSpec,
    *,
    attempts: int = 5,
    timeout: float = 60.0,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Download an official archive, resumably, and verify it before publication.

    The multi-gigabyte parent archive is the step most likely to fail during an
    unattended run, so a transient network error must not cost the whole
    download. Bytes accumulate in a persistent ``.partial`` file and a retry
    asks the server to continue from that offset with an HTTP Range request;
    only a server that ignores the range restarts from zero. The file is
    published under its real name solely after the recorded size and SHA-256
    both match, so a resumed or corrupted transfer can never be mistaken for a
    verified archive.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_path(path)
    last_error: Exception | None = None

    for attempt in range(1, max(1, attempts) + 1):
        have = partial.stat().st_size if partial.is_file() else 0
        if spec.size_bytes and have > spec.size_bytes:
            # More bytes than the archive can contain: the partial file is not
            # a prefix of this archive, so restart rather than resume into it.
            partial.unlink(missing_ok=True)
            have = 0
        try:
            headers = {"User-Agent": "WarpAudit/0.1"}
            if have:
                headers["Range"] = f"bytes={have}-"
            request = urllib.request.Request(spec.url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                resumed = response.status == 206
                if have and not resumed:
                    partial.unlink(missing_ok=True)
                    have = 0
                declared = response.headers.get("Content-Length")
                total = have + int(declared) if declared else spec.size_bytes
                mode = "ab" if have else "wb"
                with open(partial, mode) as out:
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
                        have += len(block)
                        if progress is not None:
                            progress(have, total)
                    out.flush()
                    os.fsync(out.fileno())
            verify_archive(partial, spec)
            os.replace(partial, path)
            return path
        except PreparationError:
            # The bytes are complete but wrong: resuming would only re-verify
            # the same corrupt file, so start over on the next attempt.
            partial.unlink(missing_ok=True)
            raise
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(min(2.0 ** (attempt - 1), 30.0))
        except BaseException:
            raise

    raise PreparationError(
        f"{spec.dataset_id}: download failed after {attempts} attempt(s): {last_error}. "
        f"Partial bytes are kept at {partial}; rerunning resumes from there."
    )


def _safe_zip_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or "\\" in name:
        raise PreparationError(f"unsafe ZIP member path: {name!r}")
    return member


def _validated_labelme(raw: bytes, member: str) -> tuple[dict, bytes, str]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"{member}: invalid LabelMe JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PreparationError(f"{member}: LabelMe payload is not an object")
    image_name = Path(str(payload.get("imagePath", ""))).name
    encoded = payload.get("imageData")
    if not image_name.lower().endswith((".jpg", ".jpeg")) or not isinstance(encoded, str):
        raise PreparationError(f"{member}: missing embedded JPEG imageData/imagePath")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise PreparationError(f"{member}: invalid base64 imageData") from exc
    return payload, image_bytes, image_name


def prepare_coph100(
    archive: str | Path,
    parent_archive: str | Path,
    root: str | Path,
    spec: ArchiveSpec,
    parent_spec: ArchiveSpec,
    *,
    reuse_existing: bool = False,
) -> dict[str, int]:
    """Reconstruct COph100 from its annotations and official RIDIRP parent.

    COph100 v1's LabelMe ``imageData`` fields are stale for at least some
    records and must not be treated as the named examination. The deposit's
    bundled copy helper identifies the authoritative source as
    ``<patient>/<stage>/<imagePath>`` inside RIDIRP. Only those 324 named JPEGs
    are read from the 6,004-image parent archive; masks and unrelated parent
    images are not extracted.
    """
    archive, parent_archive, root = Path(archive), Path(parent_archive), Path(root)
    verify_archive(archive, spec)
    verify_archive(parent_archive, parent_spec)
    if root.exists() and reuse_existing:
        # Both archives were just verified byte for byte; what is reused is the
        # extraction, which audit-data re-checksums file by file straight after.
        # Re-reading the 2.7 GB parent on every resumed run buys nothing.
        images = sorted(root.glob("*/*.jpg"))
        annotations = sorted(root.glob("*/*.json"))
        if not images or not annotations:
            raise PreparationError(
                f"COph100 destination {root} exists but has no extracted examinations; "
                "move it aside and prepare again"
            )
        return {
            "images": len(images),
            "annotations": len(annotations),
            "eyes": len({path.parent.name for path in images}),
            "reused_existing": 1,
        }
    root.mkdir(parents=True, exist_ok=True)

    records: list[tuple[PurePosixPath, bytes, bytes, str, str, str]] = []
    eye_counts: dict[str, int] = {}
    with ZipFile(archive) as bundle:
        for info in bundle.infolist():
            member = _safe_zip_member(info.filename)
            if info.is_dir() or member.suffix.lower() != ".json":
                continue
            if len(member.parts) != 2:
                raise PreparationError(
                    f"COph100: expected '<eye>/<exam>.json', got {info.filename!r}"
                )
            eye_id = member.parts[0]
            if not (eye_id.isdigit() or (eye_id.endswith("-1") and eye_id[:-2].isdigit())):
                raise PreparationError(f"COph100: invalid eye directory {eye_id!r}")
            raw = bundle.read(info)
            payload, image_bytes, image_name = _validated_labelme(raw, info.filename)
            if Path(image_name).stem != member.stem:
                raise PreparationError(
                    f"{info.filename}: imagePath stem {Path(image_name).stem!r} "
                    f"does not match annotation stem {member.stem!r}"
                )
            stage_match = re.search(r"_S(\d+)_", image_name)
            if stage_match is None:
                raise PreparationError(f"{info.filename}: image name has no acquisition stage")
            patient_id = image_name[:3]
            if not patient_id.isdigit():
                raise PreparationError(f"{info.filename}: image name has no patient prefix")
            records.append(
                (
                    member,
                    raw,
                    image_bytes,
                    image_name,
                    patient_id,
                    f"{int(stage_match.group(1)):02d}",
                )
            )
            eye_counts[eye_id] = eye_counts.get(eye_id, 0) + 1

    n_pairs = sum(n * (n - 1) // 2 for n in eye_counts.values())
    if len(records) != 324 or len(eye_counts) != 100 or n_pairs != 491:
        raise PreparationError(
            "COph100: official-v1 structure mismatch; expected 324 examinations, "
            f"100 eyes and 491 pairs, got {len(records)}, {len(eye_counts)}, {n_pairs}"
        )
    embedded_mismatches = 0
    with ZipFile(parent_archive) as parent:
        parent_members: dict[tuple[str, str, str], str] = {}
        for info in parent.infolist():
            member = _safe_zip_member(info.filename)
            if info.is_dir() or len(member.parts) < 3:
                continue
            key = (member.parts[-3], member.parts[-2], member.parts[-1])
            if key in parent_members:
                raise PreparationError(f"RIDIRP archive repeats source path suffix {key!r}")
            parent_members[key] = info.filename

        for member, raw, embedded, image_name, patient_id, stage in records:
            key = (patient_id, stage, image_name)
            source_member = parent_members.get(key)
            if source_member is None:
                raise PreparationError(
                    f"COph100 source image absent from RIDIRP at */{'/'.join(key)}"
                )
            image_bytes = parent.read(source_member)
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
                image.verify()
            payload = json.loads(raw)
            if (height, width) != (
                int(payload.get("imageHeight", -1)),
                int(payload.get("imageWidth", -1)),
            ):
                raise PreparationError(
                    f"{source_member}: parent image dimensions disagree with COph100 annotation"
                )
            embedded_mismatches += sha256(embedded).digest() != sha256(image_bytes).digest()
            folder = root / member.parts[0]
            atomic_write_bytes(folder / member.name, raw)
            atomic_write_bytes(folder / image_name, image_bytes)
    return {
        "images": len(records),
        "annotations": len(records),
        "eyes": len(eye_counts),
        "pairs": n_pairs,
        "stale_embedded_images": embedded_mismatches,
    }


def prepare_fire(
    archive: str | Path,
    root: str | Path,
    spec: ArchiveSpec,
    *,
    reuse_existing: bool = False,
) -> dict[str, int]:
    """Extract the checksum-verified official FIRE archive with 7-Zip.

    ``reuse_existing`` keeps a resumable run from failing on its second pass.
    The archive is still verified byte for byte; what is reused is the
    extraction, and ``audit-data`` immediately re-checksums every extracted
    file against the manifest, so reuse cannot smuggle in altered pixels.
    Without the flag an existing destination is refused rather than
    overwritten, because restricted data is not something to clobber silently.
    """
    archive, root = Path(archive), Path(root)
    verify_archive(archive, spec)
    if root.exists() and reuse_existing:
        images = sorted((root / "Images").glob("*.jpg"))
        annotations = sorted((root / "Ground Truth").glob("control_points_*.txt"))
        if not images or not annotations:
            raise PreparationError(
                f"FIRE destination {root} exists but has no extracted images or "
                "annotations; move it aside and prepare again"
            )
        return {
            "images": len(images),
            "annotations": len(annotations),
            "reused_existing": 1,
        }
    executable = shutil.which("7z") or shutil.which("7zz")
    if executable is None:
        raise PreparationError("FIRE preparation requires a 7z or 7zz executable")
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise PreparationError(
            f"FIRE destination already exists: {root}; pass --reuse-existing to keep it "
            "(the archive is still verified), or move it aside explicitly"
        )
    with tempfile.TemporaryDirectory(prefix=".prepare-fire-", dir=root.parent) as temporary:
        destination = Path(temporary)
        proc = subprocess.run(
            [executable, "x", "-y", f"-o{destination}", str(archive.resolve())],
            text=True,
            capture_output=True,
            check=False,
        )
        extracted = destination / "FIRE"
        if proc.returncode != 0 or not extracted.is_dir():
            detail = (proc.stderr or proc.stdout)[-2000:]
            raise PreparationError(f"FIRE extraction failed: {detail}")
        n_images = len(list((extracted / "Images").glob("*.jpg")))
        n_annotations = len(list((extracted / "Ground Truth").glob("*.txt")))
        if n_images != 268 or n_annotations != 134:
            raise PreparationError(
                "FIRE: archive structure mismatch; expected 268 pair-side image files and "
                f"134 annotations, got {n_images} and {n_annotations}"
            )
        os.replace(extracted, root)
    return {"pair_side_files": n_images, "annotations": n_annotations, "pairs": 134}
