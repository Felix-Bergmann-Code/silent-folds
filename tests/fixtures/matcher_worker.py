"""Tiny deterministic worker used to exercise the real subprocess boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    moving_image = np.load(request["moving"]["working_array_path"], allow_pickle=False)
    fixed_image = np.load(request["fixed"]["working_array_path"], allow_pickle=False)
    points = np.array(
        [[5, 5], [30, 5], [55, 5], [5, 30], [30, 30], [55, 30], [5, 55], [55, 55]],
        dtype=float,
    )
    offset = np.asarray(request["options"].get("offset", [0.0, 0.0]), dtype=float)
    response = {
        "protocol_version": request["protocol_version"],
        "pipeline_id": request["pipeline_id"],
        "coordinate_frame": request["options"].get(
            "coordinate_frame", "working_pixel_centres"
        ),
        "matches_moving": points.tolist(),
        "matches_fixed": (points + offset).tolist(),
        "match_scores": np.linspace(0.8, 1.0, len(points)).tolist(),
        "provenance": {"fixture_commit": "fixed"},
        "runtime_s": 0.001,
        "cpu_time_s": 0.001,
        "peak_vram_bytes": 0,
        "diagnostics": {
            "moving_shape": list(moving_image.shape),
            "fixed_shape": list(fixed_image.shape),
        },
    }
    args.response.write_text(json.dumps(response), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
