#!/usr/bin/env python3
"""Official XFeat sparse matcher worker for the isolated adapter protocol."""

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
    weights = resolve_project_path(options["weights_path"])
    add_upstream_to_path(upstream)
    requested_device = str(options.get("device", "auto"))
    if requested_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import numpy as np
    import torch
    from modules.xfeat import XFeat

    seed = int(request["seed"])
    seed_everything(seed, torch)
    # The official XFeat class selects CUDA when available and CPU otherwise;
    # it does not currently expose an MPS selector.
    device = (
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else requested_device
    )
    device = choose_device("cpu" if device == "auto" else device, torch)
    if requested_device == "mps" or device == "mps":
        raise RuntimeError("official XFeat adapter does not expose deterministic MPS selection")

    moving_image = load_working_image(request["moving"])
    fixed_image = load_working_image(request["fixed"])
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    model = XFeat(
        weights=str(weights),
        top_k=int(options.get("top_k", 4096)),
        detection_threshold=float(options.get("detection_threshold", 0.05)),
    )
    first = model.detectAndCompute(moving_image)[0]
    second = model.detectAndCompute(fixed_image)[0]
    indices_moving, indices_fixed = model.match(
        first["descriptors"],
        second["descriptors"],
        min_cossim=float(options.get("min_cossim", 0.82)),
    )
    moving = first["keypoints"][indices_moving].detach().cpu().numpy()
    fixed = second["keypoints"][indices_fixed].detach().cpu().numpy()
    scores = (
        first["descriptors"][indices_moving]
        * second["descriptors"][indices_fixed]
    ).sum(dim=1).detach().cpu().numpy()
    runtime_s = time.perf_counter() - wall_start
    cpu_time_s = time.process_time() - cpu_start
    provenance = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "upstream_commit": git_commit(upstream),
        "xfeat_weights_sha256": sha256(weights),
    }
    response = base_response(
        request,
        moving=np.asarray(moving),
        fixed=np.asarray(fixed),
        scores=np.asarray(scores),
        provenance=provenance,
        runtime_s=runtime_s,
        cpu_time_s=cpu_time_s,
        peak_vram=peak_vram_bytes(device, torch),
        diagnostics={
            "backend": "official_xfeat_sparse_mnn",
            "device": device,
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_file": torch.__file__,
            "torch_version": torch.__version__,
            "top_k": int(options.get("top_k", 4096)),
            "detection_threshold": float(options.get("detection_threshold", 0.05)),
            "min_cossim": float(options.get("min_cossim", 0.82)),
            "n_keypoints_moving": int(len(first["keypoints"])),
            "n_keypoints_fixed": int(len(second["keypoints"])),
        },
    )
    write_response(args.response, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
