"""Reproducible host latency benchmark (ONNX Runtime CPU EP) for every model variant.

Fixed seed, fixed thread count, warmup iterations discarded, percentiles over the rest.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common import BENCH_DIR, Variant, all_variants, environment, write_json

SEED = 0


def percentiles(samples_ms: list[float]) -> dict:
    arr = np.asarray(samples_ms)
    return {
        "mean_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
    }


def bench(variant: Variant, warmup: int, iterations: int, threads: int) -> dict:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.perf_counter()
    session = ort.InferenceSession(str(variant.path), opts, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t0) * 1000
    inp = session.get_inputs()[0]
    x = np.random.default_rng(SEED).random((1, 3, variant.resolution, variant.resolution), dtype=np.float32)
    for _ in range(warmup):
        session.run(None, {inp.name: x})
    samples = []
    for _ in range(iterations):
        t = time.perf_counter_ns()
        session.run(None, {inp.name: x})
        samples.append((time.perf_counter_ns() - t) / 1e6)
    return {
        "variant": variant.name,
        "resolution": variant.resolution,
        "precision": variant.precision,
        "backend": "CPU (ORT CPUExecutionProvider)",
        "model_size_mb": round(variant.path.stat().st_size / 1e6, 2),
        "session_load_ms": round(load_ms, 1),
        **percentiles(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=str(BENCH_DIR / "host_cpu.json"))
    args = parser.parse_args()
    results = []
    for v in all_variants():
        if not v.path.exists():
            print(f"skip {v.name}: not built")
            continue
        r = bench(v, args.warmup, args.iterations, args.threads)
        print(f"{r['variant']:<20} p50={r['p50_ms']:7.2f} p95={r['p95_ms']:7.2f} p99={r['p99_ms']:7.2f} ms")
        results.append(r)
    write_json(
        Path(args.out),
        {
            "config": {"warmup": args.warmup, "iterations": args.iterations, "threads": args.threads, "seed": SEED},
            "environment": environment(),
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
