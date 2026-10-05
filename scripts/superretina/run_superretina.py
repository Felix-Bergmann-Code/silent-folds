#!/usr/bin/env python3
"""Stage 1 of the SuperRetina pole audit: run the official released pipeline.

Runs inside the isolated SuperRetina environment (see
``scripts/run_isbi_submission.ps1``, which builds it).  It reproduces the official
``test_on_FIRE.py`` registration exactly -- green channel, CLAHE/gamma
pre-processing, 768x768 network input, NMS 10 / 0.01, 0.9 ratio test,
``cv2.findHomography(..., cv2.LMEDS)`` and the two-pass "matching trick" --
and additionally records the stage-1 correspondences so the evaluation
environment can re-fit them.

This file is deliberately outside ``scripts/matchers/`` so that it does not
change the registration-cache code identity.

Input: ``jobs.json`` from ``scripts/isbi_superretina_audit.py export-jobs``.
Output: one JSON line per job in ``--output`` (resumable; finished jobs are
skipped).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def homography_with_matches(predictor, query, refer_path, *, query_is_image, cv2, np):
    """``Predictor.compute_homography`` verbatim, also returning the matches."""

    good, kq, kr, raw_query, raw_refer = predictor.match(
        query, refer_path, query_is_image=query_is_image
    )
    h = None
    rate = 0.0
    src = np.empty((0, 2))
    dst = np.empty((0, 2))
    mask = np.empty(0, dtype=bool)
    if len(good) >= 4:
        src = np.float32([kq[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst = np.float32([kr[m.trainIdx].pt for m in good]).reshape(-1, 2)
        h, lmeds_mask = cv2.findHomography(
            src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), cv2.LMEDS
        )
        if lmeds_mask is not None:
            mask = lmeds_mask.ravel().astype(bool)
            rate = float(mask.sum() / len(mask))
    return h, rate, raw_query, src, dst, mask, len(kq), len(kr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--superretina-root", type=Path, default=Path("vendor/SuperRetina"))
    parser.add_argument("--weights", type=Path,
                        default=Path("checkpoints/superretina/SuperRetina.pth"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-matching-trick", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0,
                        help="this process's shard index (several shards share one GPU)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--cv-threads", type=int, default=4)
    args = parser.parse_args()

    project = Path.cwd().resolve()
    root = args.superretina_root.resolve()
    weights = args.weights.resolve()
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))["jobs"]
    if args.limit:
        jobs = jobs[: args.limit]
    if not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be in [0, --num-shards)")
    # Seeds are per job, so results do not depend on the sharding.
    jobs = jobs[args.shard :: args.num_shards]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["job_key"])

    sys.path.insert(0, str(root))
    os.chdir(root)  # upstream imports the namespace package ``config``
    import cv2
    import numpy as np
    import torch
    from predictor import Predictor

    cv2.setNumThreads(args.cv_threads)
    torch.set_num_threads(args.cv_threads)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False

    config = {
        "PREDICT": {
            "device": args.device,
            "model_save_path": str(weights),
            "model_image_width": 768,
            "model_image_height": 768,
            "use_matching_trick": not args.no_matching_trick,
            "nms_size": 10,
            "nms_thresh": 0.01,
            "knn_thresh": 0.9,
        }
    }
    predictor = Predictor(config)
    provenance = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "weights_sha256": sha256(weights),
        "device": str(predictor.device),
        "matching_trick": not args.no_matching_trick,
    }
    print(json.dumps(provenance), flush=True)
    started = time.time()
    with output.open("a", encoding="utf-8") as stream:
        for index, job in enumerate(jobs, start=1):
            if job["job_key"] in done:
                continue
            seed = int(job["seed"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            cv2.setRNGSeed(seed)
            query_path = str((project / job["query_path"]).resolve())
            refer_path = str((project / job["refer_path"]).resolve())
            record = {"job_key": job["job_key"], "provenance": provenance}
            t0 = time.perf_counter()
            try:
                h1, rate1, query_image, src, dst, mask, nq, nr = homography_with_matches(
                    predictor, query_path, refer_path, query_is_image=False, cv2=cv2, np=np
                )
                h2, rate2 = None, None
                if not args.no_matching_trick and h1 is not None:
                    hh, ww = predictor.image_height, predictor.image_width
                    aligned = cv2.warpPerspective(
                        query_image, h1, (ww, hh), borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0),
                    ).astype(float) / 255.0
                    h2, rate2, *_ = homography_with_matches(
                        predictor, aligned, refer_path, query_is_image=True, cv2=cv2, np=np
                    )
                record.update(
                    status="ok",
                    query_hw=[int(predictor.image_height), int(predictor.image_width)],
                    h1=None if h1 is None else np.asarray(h1, float).tolist(),
                    h2=None if h2 is None else np.asarray(h2, float).tolist(),
                    inlier_rate_stage1=rate1,
                    inlier_rate_stage2=rate2,
                    # Official failure rule uses the rate of the last executed pass.
                    official_inlier_rate=rate2 if rate2 is not None else rate1,
                    n_keypoints_query=nq,
                    n_keypoints_refer=nr,
                    stage1_query_points=np.asarray(src, float).tolist(),
                    stage1_refer_points=np.asarray(dst, float).tolist(),
                    stage1_lmeds_inliers=np.asarray(mask, bool).tolist(),
                )
            except Exception as exc:  # recorded, never silently dropped
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            record["runtime_s"] = time.perf_counter() - t0
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if index % 25 == 0 or index == len(jobs):
                print(f"[superretina shard {args.shard}] {index}/{len(jobs)} ({time.time() - started:.0f}s)",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
