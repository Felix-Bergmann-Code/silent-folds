from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from warpaudit.data.annotations import (
    load_coph100_control_points,
    load_labelme_points,
)
from warpaudit.data.loaders import list_coph100_pairs


def _write_exam(folder: Path, stem: str, labels: list[str], offset: float) -> None:
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(folder / f"{stem}.jpg")
    shapes = [
        {
            "label": label,
            "points": [[float(index + offset), float(2 * index + offset)]],
            "shape_type": "point",
        }
        for index, label in enumerate(labels)
    ]
    (folder / f"{stem}.json").write_text(
        json.dumps({"shapes": shapes, "imageWidth": 16, "imageHeight": 12}),
        encoding="utf-8",
    )


def test_coph100_layout_builds_all_within_eye_pairs_and_patient_groups(tmp_path: Path) -> None:
    eye = tmp_path / "002-1"
    eye.mkdir()
    for session in range(1, 4):
        _write_exam(
            eye,
            f"002_F_GA40_BW3000_PA4{session}_DG0_PF0_D1_S0{session}_{session}",
            ["1", "2", "3", "4"],
            float(session),
        )

    pairs = list_coph100_pairs(tmp_path)
    assert len(pairs) == 3
    assert {pair.subject_id for pair in pairs} == {"002"}
    assert {pair.subject_basis for pair in pairs} == {"patient"}
    assert {pair.extra["eye_id"] for pair in pairs} == {"002-1"}
    assert all(len(pair.annotation_paths) == 2 for pair in pairs)


def test_coph100_duplicate_labels_are_retained_but_only_common_keys_are_paired(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_exam(tmp_path, "first", ["1", "1", "3", "4", "5"], 0.0)
    _write_exam(tmp_path, "second", ["1", "2", "3", "4", "5"], 1.0)

    decoded = load_labelme_points(first)
    assert set(decoded) == {"1", "1#2", "3", "4", "5"}
    moving, fixed = load_coph100_control_points(first, second)
    assert moving.shape == fixed.shape == (4, 2)
    assert np.allclose(fixed - moving, 1.0)


def test_an_interrupted_download_resumes_instead_of_restarting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped connection must not cost a multi-gigabyte transfer."""
    from warpaudit.data import prepare

    payload = bytes(range(256)) * 40
    spec = prepare.ArchiveSpec(
        dataset_id="fixture",
        url="https://example.invalid/archive.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    ranges: list[int] = []
    calls = {"n": 0}

    class _Response:
        def __init__(self, start: int, fail_after: int | None) -> None:
            self._data = payload[start:]
            self._offset = 0
            self._fail_after = fail_after
            self.status = 206 if start else 200
            self.headers = {"Content-Length": str(len(self._data))}

        def read(self, size: int) -> bytes:
            if self._fail_after is not None and self._offset >= self._fail_after:
                raise OSError("connection reset")
            block = self._data[self._offset : self._offset + size]
            self._offset += len(block)
            return block

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        start = 0
        header = request.headers.get("Range")
        if header:
            start = int(header.split("=")[1].split("-")[0])
        ranges.append(start)
        # Drop the first connection part-way through, then serve the rest.
        return _Response(start, fail_after=1000 if calls["n"] == 1 else None)

    monkeypatch.setattr(prepare.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(prepare.time, "sleep", lambda *_: None)

    destination = tmp_path / "archive.bin"
    prepare.download_archive(destination, spec)

    assert destination.read_bytes() == payload
    assert calls["n"] == 2
    # The retry asked to continue from the bytes already on disk.
    assert ranges[0] == 0 and ranges[1] > 0
    assert not prepare._partial_path(destination).exists()


def test_a_corrupt_download_is_discarded_rather_than_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from warpaudit.data import prepare

    payload = b"wrong bytes entirely"
    spec = prepare.ArchiveSpec(
        dataset_id="fixture",
        url="https://example.invalid/archive.bin",
        sha256=hashlib.sha256(b"expected").hexdigest(),
        size_bytes=len(payload),
    )

    class _Response:
        status = 200
        headers = {"Content-Length": str(len(payload))}

        def __init__(self) -> None:
            self._sent = False

        def read(self, size: int) -> bytes:
            if self._sent:
                return b""
            self._sent = True
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **k: _Response())
    destination = tmp_path / "archive.bin"
    with pytest.raises(prepare.PreparationError, match="SHA-256 mismatch"):
        prepare.download_archive(destination, spec)
    assert not destination.exists()
    # Complete-but-wrong bytes are removed: resuming would re-verify the same file.
    assert not prepare._partial_path(destination).exists()
