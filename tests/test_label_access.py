from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest

from warpaudit.types import LabelAccessError, SignalContext, sanitise_acquisition_meta


def test_signal_context_has_no_annotation_field() -> None:
    names = {field.name for field in fields(SignalContext)}
    assert not ({"annotation", "label", "error", "landmarks"} & names)


def test_forbidden_metadata_is_rejected() -> None:
    with pytest.raises(LabelAccessError):
        sanitise_acquisition_meta({"modality": "fundus", "patient_id": "7"})


def test_signal_and_registration_modules_do_not_import_label_modules() -> None:
    root = Path(__file__).parents[1] / "warpaudit"
    violations = []
    for folder in (root / "signals", root / "registration"):
        for path in folder.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    resolved = node.module.lstrip(".")
                    if "labels" in resolved or "annotations" in resolved:
                        violations.append((path.name, node.lineno, node.module))
    assert violations == []
