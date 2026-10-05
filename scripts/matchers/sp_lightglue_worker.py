#!/usr/bin/env python3
"""Official SuperPoint + LightGlue worker for the isolated adapter protocol."""

from __future__ import annotations

import os
import platform
import sys
import time

from _common import (
    add_upstream_to_path,
    arguments,
    base_response,
    choose_device,
    git_commit,
    load_working_image,
    peak_vram_bytes,
    read_request,
    resolve_project_path,
    seed_everything,
    sha256,
    write_response,
)


def main() -> int:
    args = arguments()
    request = read_request(args.request)
    options = request["options"]
    upstream = resolve_project_path(options["upstream_root"])
    torch_home = resolve_project_path(options["torch_home"])
    os.environ["TORCH_HOME"] = str(torch_home)
    add_upstream_to_path(upstream)

    import cv2
    import kornia
    import numpy as np
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import numpy_image_to_torch, rbd

    seed = int(request["seed"])
    seed_everything(seed, torch)
    device = choose_device(str(options.get("device", "auto")), torch)
    moving_image = load_working_image(request["moving"])
    fixed_image = load_working_image(request["fixed"])
    moving_tensor = numpy_image_to_torch(moving_image).to(device)
    fixed_tensor = numpy_image_to_torch(fixed_image).to(device)

    wall_start, cpu_start = time.perf_counter(), time.process_time()
    extractor = SuperPoint(
        max_num_keypoints=int(options.get("max_num_keypoints", 2048)),
        detection_threshold=float(options.get("detection_threshold", 0.0005)),
        nms_radius=int(options.get("nms_radius", 4)),
    ).eval().to(device)
    matcher = LightGlue(
        features="superpoint",
        flash=bool(options.get("flash", True)),
        mp=bool(options.get("mixed_precision", False)),
        depth_confidence=float(options.get("depth_confidence", 0.95)),
        width_confidence=float(options.get("width_confidence", 0.99)),
        filter_threshold=float(options.get("filter_threshold", 0.1)),
    ).eval().to(device)
    first = extractor.extract(moving_tensor, resize=None)
    second = extractor.extract(fixed_tensor, resize=None)
    output = matcher({"image0": first, "image1": second})
    first, second, output = (rbd(value) for value in (first, second, output))
    indices = output["matches"]
    moving = first["keypoints"][indices[:, 0]].detach().cpu().numpy()
    fixed = second["keypoints"][indices[:, 1]].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    runtime_s = time.perf_counter() - wall_start
    cpu_time_s = time.process_time() - cpu_start

    checkpoint_dir = torch_home / "hub" / "checkpoints"
    superpoint_weights = checkpoint_dir / "superpoint_v1.pth"
    lightglue_weights = checkpoint_dir / "superpoint_lightglue_v0-1_arxiv.pth"
    provenance = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "kornia_version": kornia.__version__,
        "opencv_version": cv2.__version__,
        "upstream_commit": git_commit(upstream),
        "superpoint_weights_sha256": sha256(superpoint_weights),
        "lightglue_weights_sha256": sha256(lightglue_weights),
    }
    response = base_response(
        request,
        moving=moving,
        fixed=fixed,
        scores=scores,
        provenance=provenance,
        runtime_s=runtime_s,
        cpu_time_s=cpu_time_s,
        peak_vram=peak_vram_bytes(device, torch),
        diagnostics={
            "backend": "official_superpoint_lightglue",
            "device": device,
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_file": torch.__file__,
            "torch_version": torch.__version__,
            "max_num_keypoints": int(options.get("max_num_keypoints", 2048)),
            "filter_threshold": float(options.get("filter_threshold", 0.1)),
            "n_keypoints_moving": int(len(first["keypoints"])),
            "n_keypoints_fixed": int(len(second["keypoints"])),
            "lightglue_stop_layer": int(output["stop"]),
        },
    )
    write_response(args.response, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
