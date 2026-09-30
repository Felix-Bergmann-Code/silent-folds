from __future__ import annotations

from warpaudit.data.grouping import assign_groups, connected_components, grouping_claim
from warpaudit.protocols.leakage import check_derivative_groups


def test_shared_images_form_one_component() -> None:
    pairs = [("p1", "a", "b"), ("p2", "b", "c"), ("p3", "d", "e")]
    components = connected_components(pairs)
    assert components["p1"] == components["p2"]
    assert components["p3"] != components["p1"]


def test_missing_patient_ids_never_become_pair_groups() -> None:
    pairs = [("p1", "a", "b"), ("p2", "c", "d")]
    assignments, report = assign_groups("FIRE", pairs)
    assert all("image_component" in a.group_id for a in assignments)
    assert not report.patient_disjoint_claimable
    assert "MUST NOT" in grouping_claim(report)


def test_mixed_patient_and_eye_evidence_preserves_per_pair_basis() -> None:
    pairs = [("p1", "a", "b"), ("p2", "c", "d")]
    assignments, report = assign_groups(
        "D",
        pairs,
        subject_ids={"p1": "person", "p2": "eye"},
        subject_bases={"p1": "patient", "p2": "eye"},
    )
    assert {a.group_basis for a in assignments} == {"patient", "eye"}
    assert report.basis == "eye"


def test_derivative_group_change_is_detected() -> None:
    rows = [{"pair_id": "p", "group_id": "g1"}, {"pair_id": "p", "group_id": "g2"}]
    assert len(check_derivative_groups(rows)) == 1
