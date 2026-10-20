"""Profiles each variant on real Snapdragon hardware through Qualcomm AI Hub.

For every variant: compile (ONNX Runtime + QNN target), profile on device, and run an
inference job whose output is compared against local ONNX Runtime CPU output.
The per-layer compute-unit breakdown from the profile is what justifies any NPU claim.

Requires `qai-hub configure --api_token <TOKEN>` (or QAI_HUB_API_TOKEN). Exits cleanly without it.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from common import BENCH_DIR, all_variants, coco128_images, preprocess, write_json

DEFAULT_DEVICE = "Samsung Galaxy S24 (Family)"


def has_token() -> bool:
    return bool(os.environ.get("QAI_HUB_API_TOKEN")) or (Path.home() / ".qai_hub" / "client.ini").exists()


def compute_units(profile: dict) -> dict:
    layers = profile.get("execution_detail", [])
    return dict(Counter(layer.get("compute_unit", "UNKNOWN") for layer in layers))


def psnr(ref: np.ndarray, out: np.ndarray) -> float:
    mse = float(np.mean((ref.astype(np.float64) - out.astype(np.float64)) ** 2))
    peak = float(np.abs(ref).max())
    return float("inf") if mse == 0 else 10 * np.log10(peak**2 / mse)


def profile_variant(hub, variant, device) -> dict:
    sample, _ = preprocess(Image.open(coco128_images()[0]), variant.resolution)
    compile_job = hub.submit_compile_job(
        model=str(variant.path),
        device=device,
        options="--target_runtime onnx --compute_unit npu",
        name=f"{variant.name}-compile",
    )
    target = compile_job.get_target_model()
    if target is None:
        return {"variant": variant.name, "status": "compile_failed", "job": compile_job.url}

    profile_job = hub.submit_profile_job(model=target, device=device, name=f"{variant.name}-profile")
    profile = profile_job.download_profile()
    summary = profile.get("execution_summary", {})

    inference_job = hub.submit_inference_job(
        model=target, device=device, inputs={"images": [sample]}, name=f"{variant.name}-inference"
    )
    device_out = list(inference_job.download_output_data().values())[0][0]
    ref = ort.InferenceSession(str(variant.path), providers=["CPUExecutionProvider"]).run(None, {"images": sample})[0]

    units = compute_units(profile)
    return {
        "variant": variant.name,
        "device": device.name,
        "status": "ok",
        "inference_time_ms": round(summary.get("estimated_inference_time", 0) / 1000, 3),
        "peak_memory_mb": round(summary.get("inference_memory_peak_range", [0, 0])[1] / 1e6, 2),
        "compute_units": units,
        "npu_verified": units.get("NPU", 0) > 0 and set(units) <= {"NPU"},
        "psnr_vs_host_db": round(psnr(ref, device_out), 2),
        "jobs": {"compile": compile_job.url, "profile": profile_job.url, "inference": inference_job.url},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--out", default=str(BENCH_DIR / "aihub_profile.json"))
    args = parser.parse_args()
    if not has_token():
        print("No Qualcomm AI Hub token configured; skipping on-device profiling.")
        return
    import qai_hub as hub

    device = hub.Device(args.device)
    results = []
    for v in all_variants():
        if v.path.exists():
            r = profile_variant(hub, v, device)
            print(r)
            results.append(r)
    write_json(Path(args.out), {"device": args.device, "results": results})


if __name__ == "__main__":
    main()
