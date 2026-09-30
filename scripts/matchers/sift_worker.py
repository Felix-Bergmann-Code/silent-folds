#!/usr/bin/env python3
"""Classical SIFT baseline using the same WarpAudit robust fitter."""

from __future__ import annotations

import platform
import sys
import time

import cv2
import numpy as np
from _common import (
    arguments,
    base_response,
    load_working_image,
    read_request,
    write_response,
)


def main() -> int:
    args = arguments()
    request = read_request(args.request)
    options = request["options"]
    seed = int(request["seed"])
    cv2.setRNGSeed(seed & 0x7FFFFFFF)
    moving_image = load_working_image(request["moving"])
    fixed_image = load_working_image(request["fixed"])
    moving_gray = cv2.cvtColor((moving_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    fixed_gray = cv2.cvtColor((fixed_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    started, cpu_started = time.perf_counter(), time.process_time()
    sift = cv2.SIFT_create(nfeatures=int(options.get("nfeatures", 4096)))
    key_m, desc_m = sift.detectAndCompute(moving_gray, None)
    key_f, desc_f = sift.detectAndCompute(fixed_gray, None)
    ratio = float(options.get("ratio", 0.8))
    accepted = []
    if desc_m is not None and desc_f is not None and len(desc_f) >= 2:
        for nearest in cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_m, desc_f, k=2):
            if len(nearest) == 2 and nearest[0].distance < ratio * nearest[1].distance:
                accepted.append(nearest[0])
    moving = np.asarray([key_m[m.queryIdx].pt for m in accepted], dtype=float).reshape(-1, 2)
    fixed = np.asarray([key_f[m.trainIdx].pt for m in accepted], dtype=float).reshape(-1, 2)
    scores = np.asarray([1.0 / (1.0 + m.distance) for m in accepted], dtype=float)
    response = base_response(
        request,
        moving=moving,
        fixed=fixed,
        scores=scores,
        provenance={
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
        },
        runtime_s=time.perf_counter() - started,
        cpu_time_s=time.process_time() - cpu_started,
        peak_vram=0,
        diagnostics={
            "backend": "opencv_sift_ratio",
            "python_executable": sys.executable,
            "nfeatures": int(options.get("nfeatures", 4096)),
            "ratio": ratio,
            "n_keypoints_moving": len(key_m),
            "n_keypoints_fixed": len(key_f),
        },
    )
    write_response(args.response, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
