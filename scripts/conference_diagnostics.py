"""Development-only diagnostics for the extension; never used as detector inputs."""

from __future__ import annotations

import numpy as np


def availability(frame, columns):
    finite = np.isfinite(frame.loc[:, columns].to_numpy(float))
    counts = finite.sum(axis=0)
    missing = [c for c, n in zip(columns, counts, strict=True) if n == 0]
    return {
        "cases": len(frame),
        "finite_fraction": float(finite.mean()) if finite.size else None,
        "finite_counts": dict(zip(columns, counts.tolist(), strict=True)),
        "all_missing_columns": missing,
        "all_missing_measurements": [c for c in missing if not c.endswith(":family_available")],
    }


def support_warnings(row, minimum_groups=10):
    """A disclosed review threshold, not a power calculation or fitting gate."""
    warnings = []
    for dataset, counts in row["datasets"].items():
        if not counts["failures"] or not counts["successes"]:
            warnings.append(
                {
                    "code": "single_class_dataset",
                    "location": dataset,
                    "message": "One development dataset has only one outcome class.",
                }
            )
    subsets = [(k, row[k]) for k in ("train", "calibration") if k in row]
    subsets += [
        (f"inner/{r['fold']}/{part}", r[part])
        for r in row.get("inner_folds", [])
        for part in ("train", "validation")
    ]
    for location, counts in subsets:
        if min(counts["failure_groups"], counts["success_groups"]) < minimum_groups:
            warnings.append(
                {
                    "code": "sparse_class_support",
                    "location": location,
                    "failure_groups": counts["failure_groups"],
                    "success_groups": counts["success_groups"],
                    "message": f"Fewer than {minimum_groups} groups bear one class; "
                    "computable does not imply reliable calibration/selection.",
                }
            )
    return warnings


def coverage_contrasts(reference, candidate):
    """Never describe tied, unequal achieved coverage as a matched-risk gain."""
    rows = []
    for a, b in zip(reference, candidate, strict=True):
        if a["requested_coverage"] != b["requested_coverage"]:
            raise ValueError("coverage grids differ")
        ca, cb = a["achieved_coverage"], b["achieved_coverage"]
        matched = bool(
            a["attainable"]
            and b["attainable"]
            and ca is not None
            and cb is not None
            and np.isclose(ca, cb, rtol=0, atol=1e-12)
        )
        rows.append(
            {
                "requested_coverage": a["requested_coverage"],
                "reference_achieved_coverage": ca,
                "candidate_achieved_coverage": cb,
                "reference_risk": a["risk"],
                "candidate_risk": b["risk"],
                "matched_achieved_coverage": matched,
                "risk_improvement": a["risk"] - b["risk"] if matched else None,
            }
        )
    return rows


def render_preflight(report):
    lines = [
        "# Development preflight review",
        "",
        f"Executable fits passed: **{report['all_passed']}**. Scientific review is separate.",
        "No held-out detector performance is measured by this preflight.",
        "",
        "Sparse-support warnings use 10 class-bearing groups as a review flag, "
        "not a power guarantee or an instruction to change the fixed split.",
        "",
    ]
    for name, row in report["pipelines"].items():
        lines += [
            f"## {name}",
            "",
            f"Training pipelines: {', '.join(row['training_pipelines'])}.",
            "",
        ]
        for part in ("train", "calibration"):
            if part in row:
                r = row[part]
                lines.append(
                    f"- {part}: {r['failures']} failures / {r['successes']} successes; "
                    f"{r['failure_groups']} failure-bearing / {r['success_groups']} "
                    "success-bearing groups."
                )
        lines += [
            f"- WARNING [{w['code']}]: {w.get('location', '')} {w['message']}"
            for w in row.get("warnings", [])
        ]
        lines += [f"- BLOCKER: {issue}" for issue in row["issues"]]
        lines += [
            "",
            "| Arm | Fit | Finite entries | Entirely missing measurements |",
            "|---|---|---:|---:|",
        ]
        for arm, result in row["arms"].items():
            lines.append(
                f"| {arm} | {result['fit_passed']} | "
                f"{result['finite_fraction']:.1%} | "
                f"{len(result['availability']['all_missing_measurements'])} |"
            )
        lines += ["", "Target-pipeline development feature contract (outcomes not used):", ""]
        for arm, contract in row.get("target_feature_contract", {}).items():
            missing = contract["all_missing_measurements"]
            if missing:
                lines.append(f"- {arm}: {', '.join(missing)}")
        lines.append("")
    lines += [
        "## Before external evaluation",
        "",
        "Review the SIFT gallery and label/coordinate audit in `development_review.json`. "
        "A successful audit verifies cache consistency, not anatomical correctness.",
        "",
        "TPS remains a forward-map registration adapter without an inverse sampler. "
        "Unavailable appearance and common-support measurements remain missing. "
        "The extension evaluates available-signal bundles; held-out TPS includes "
        "a feature-availability shift. No numerical inverse or new registration "
        "method is introduced by this revision.",
        "",
    ]
    return "\n".join(lines)


def select_gallery(cases, per_stratum=3):
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    selected = cases.copy()
    selected["review_category"] = np.where(
        selected.explicit_failure,
        "explicit_failure",
        np.where(selected.operational_failure == 1, "silent_failure", "success"),
    )
    # One case per group first: the gallery illustrates breadth, not prevalence.
    return (
        selected.sort_values("job_id", kind="stable")
        .drop_duplicates(["dataset_id", "review_category", "group_id"])
        .groupby(["dataset_id", "review_category"], sort=True)
        .head(per_stratum)
    )


def audit_sift(cases, cfg, root, out, per_stratum=3):
    """Recompute all current development SIFT labels; render deterministic examples."""
    from PIL import Image, ImageDraw

    from warpaudit.cache.hashing import file_digest
    from warpaudit.cli import (
        _annotation_points,
        _compute_label_task,
        _pair_from_row,
        _pairs_manifest,
        _result_from_row,
        _verify_pair_images,
    )
    from warpaudit.evaluation.overlays import contact_sheet, render_registration_panel
    from warpaudit.geometry.transforms import HomographyTransform

    # Filter before touching annotations, including if a caller supplies external cases.
    sift = cases[
        (cases.pipeline_id == "sift_h") & cases.dataset_id.isin(cfg.full_study.development_datasets)
    ]
    pairs, _ = _pairs_manifest(cfg, root)
    pairs = pairs[pairs.dataset_id.isin(cfg.full_study.development_datasets)].set_index(
        "pair_id", drop=False
    )
    selected = select_gallery(sift, per_stratum).set_index("job_id")
    gallery = out / "sift_review"
    gallery.mkdir(parents=True, exist_ok=True)
    records, checks, panels = [], [], []
    for _, row in sift.sort_values("job_id", kind="stable").iterrows():
        pair_row = pairs.loc[row.pair_id]
        recomputed = _compute_label_task((row, pair_row, cfg, root))
        mismatches = []
        for key in ("operational_failure", "eligible_for_acceptance", "bounded_loss"):
            if not np.isclose(
                float(row[key]), float(recomputed[key]), rtol=1e-9, atol=1e-10, equal_nan=True
            ):
                mismatches.append(key)
        if row.eligible_for_acceptance and not recomputed["tre_defined"]:
            mismatches.append("undefined_ground_truth")
        check = {
            "job_id": row.job_id,
            "dataset": row.dataset_id,
            "mismatches": mismatches,
            "tre_px": recomputed["tre_px"],
            "tre_norm": recomputed["tre_norm"],
            "tre_reason": recomputed["tre_reason"],
            "tau": recomputed["tau"],
            "fixed_diagonal_px": recomputed["fixed_diagonal_px"],
        }
        checks.append(check)
        if row.job_id not in selected.index:
            continue
        pair = _pair_from_row(pair_row, cfg, root, direction="canonical")
        result = _result_from_row(row)
        record = {
            **check,
            "pair_id": row.pair_id,
            "group_id": row.group_id,
            "category": selected.loc[row.job_id, "review_category"],
            "status": str(row.status),
            "diagnostics": result.diagnostics,
            "A_m": pair.coordinates.A_m,
            "A_f": pair.coordinates.A_f,
            "moving_original_hw": [
                pair.coordinates.moving.original_height,
                pair.coordinates.moving.original_width,
            ],
            "fixed_original_hw": [
                pair.coordinates.fixed.original_height,
                pair.coordinates.fixed.original_width,
            ],
            "match_count": 0 if result.matches_moving is None else len(result.matches_moving),
            "inlier_count": 0 if result.inlier_mask is None else int(result.inlier_mask.sum()),
            "file": None,
            "note": "No transform returned.",
        }
        transform = result.forward_moving_to_fixed
        if transform is not None:
            _verify_pair_images(pair_row, root)
            with Image.open(pair.moving_path) as image:
                moving = np.asarray(image.convert("RGB"))
            with Image.open(pair.fixed_path) as image:
                fixed = np.asarray(image.convert("RGB"))
            pm, pf = _annotation_points(pair_row, root, direction="canonical")
            am, af = (
                HomographyTransform(pair.coordinates.A_m),
                HomographyTransform(pair.coordinates.A_f),
            )
            predicted = transform.apply(am.apply(pm))
            expected = af.apply(pf)
            panel, overlap = render_registration_panel(
                moving,
                fixed,
                transform,
                pair.coordinates,
                title=f"{record['category']}; {row.pair_id}; TRE={recomputed['tre_px']:.2f} original px",
            )
            image = Image.fromarray(panel)
            draw = ImageDraw.Draw(image)
            width = pair.coordinates.fixed.working_width
            height = pair.coordinates.fixed.working_height
            # Annotated fixed (cyan) versus transformed moving (magenta), in working pixels.
            for target, estimate in zip(expected, predicted, strict=True):
                for point, colour in ((target, "cyan"), (estimate, "magenta")):
                    if (
                        np.isfinite(point).all()
                        and 0 <= point[0] < width
                        and 0 <= point[1] < height
                    ):
                        x, y = point + [2 * width, 32]
                        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=colour, width=2)
            filename = f"sift_review/{row.job_id}.png"
            image.save(out / filename)
            panels.append(np.asarray(image))
            record.update(
                file=filename,
                sha256=file_digest(out / filename),
                overlap_fraction=overlap,
                landmarks_fixed_working=expected,
                landmarks_predicted_working=predicted,
                note="Cyan: fixed landmarks; magenta: mapped moving landmarks. "
                "Out-of-frame points retained in numerical error.",
            )
        records.append(record)
    if panels:
        Image.fromarray(contact_sheet(panels, columns=2)).save(gallery / "contact_sheet.png")
    return {
        "passed": bool(len(sift)) and not any(r["mismatches"] for r in checks),
        "audited_cases": len(sift),
        "label_checks": checks,
        "gallery": records,
        "selection_rule": "Per development dataset/outcome class: sort job_id, "
        "one case per group, first three groups (or requested per_stratum).",
        "interpretation": "Cache/label consistency only; actual visual review still required.",
    }
