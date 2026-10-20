"""Merges benchmark JSON (host, AI Hub, on-device app exports) into benchmarks/RESULTS.md + charts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import BENCH_DIR  # noqa: E402

PENDING = "_Pending: no measurements yet. See the README for the one-command reproduction steps._\n"


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def host_section(host: dict | None, acc: dict | None) -> str:
    if not host:
        return PENDING
    accuracy = {r["variant"]: r for r in (acc or {}).get("results", [])}
    base = {r["resolution"]: r["p50_ms"] for r in host["results"] if r["precision"] == "fp32"}
    base_map = {k.rsplit("_", 1)[0]: v["map50_95"] for k, v in accuracy.items() if k.endswith("fp32")}
    rows = []
    for r in host["results"]:
        a = accuracy.get(r["variant"], {})
        key = r["variant"].rsplit("_", 1)[0]
        speedup = base.get(r["resolution"], r["p50_ms"]) / r["p50_ms"]
        delta = a.get("map50_95", 0) - base_map.get(key, 0) if a else None
        rows.append([
            r["variant"], r["model_size_mb"], r["p50_ms"], r["p95_ms"], r["p99_ms"], f"{speedup:.2f}x",
            a.get("map50", "-"), a.get("map50_95", "-"), f"{delta:+.4f}" if delta is not None else "-",
        ])
    env, cfg = host["environment"], host["config"]
    return (
        f"Environment: {env['system']} ({env['machine']}), ONNX Runtime {env['onnxruntime']}, "
        f"{cfg['threads']} threads, {cfg['warmup']} warmup + {cfg['iterations']} timed runs, seed {cfg['seed']}.\n\n"
        + table(
            ["Variant", "Size (MB)", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Speedup vs FP32", "mAP50", "mAP50-95", "Delta mAP50-95"],
            rows,
        )
        + "\n![Host latency](host_latency.png)\n"
    )


def aihub_section(hub: dict | None) -> str:
    if not hub:
        return PENDING
    rows = [
        [r["variant"], r.get("inference_time_ms", "-"), r.get("peak_memory_mb", "-"),
         r.get("compute_units", "-"), "yes" if r.get("npu_verified") else "no", r.get("psnr_vs_host_db", "-")]
        for r in hub["results"]
    ]
    return f"Device: {hub['device']}\n\n" + table(
        ["Variant", "Inference (ms)", "Peak mem (MB)", "Layers per compute unit", "Fully on NPU", "PSNR vs host (dB)"], rows
    )


def device_section(files: list[Path]) -> str:
    if not files:
        return PENDING
    parts = []
    for f in files:
        d = json.loads(f.read_text())
        dev = d["device"]
        rows = [
            [r["variant"], r["backend"], "yes" if r["npu_verified"] else "no", r["p50_ms"], r["p95_ms"], r["p99_ms"],
             r.get("preprocess_p50_ms", "-"), r.get("fps", "-")]
            for r in d["results"]
        ]
        parts.append(
            f"### {dev['manufacturer']} {dev['model']} ({dev.get('soc', 'unknown SoC')}, Android {dev['android']})\n\n"
            + table(["Variant", "Backend", "NPU verified", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Preprocess p50 (ms)", "FPS"], rows)
        )
        if d.get("sustained", {}).get("samples"):
            chart = f"{f.stem}_sustained.png"
            sustained_chart(d["sustained"]["samples"], BENCH_DIR / chart)
            parts.append(f"\n![Sustained load]({chart})\n")
    return "\n".join(parts)


def host_chart(host: dict) -> None:
    names = [r["variant"].replace("yolo11n_", "") for r in host["results"]]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.27
    for i, key in enumerate(["p50_ms", "p95_ms", "p99_ms"]):
        ax.bar([x + (i - 1) * width for x in range(len(names))], [r[key] for r in host["results"]], width, label=key[:3])
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("YOLO11n host CPU inference latency (ONNX Runtime)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(BENCH_DIR / "host_latency.png", dpi=120)
    plt.close(fig)


def sustained_chart(samples: list[dict], path: Path) -> None:
    samples = [{k: (float("nan") if v is None else v) for k, v in s.items()} for s in samples]
    t = [s["t_s"] for s in samples]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    a1.plot(t, [s["fps"] for s in samples], label="FPS")
    a1.plot(t, [s["p95_ms"] for s in samples], label="p95 latency (ms)")
    a1.legend()
    a2.plot(t, [s["battery_temp_c"] if s["battery_temp_c"] is not None else float("nan") for s in samples],
            label="Battery temp (C)")
    a2.step(t, [s["thermal_status"] for s in samples], label="Thermal status", where="post")
    a2.set_xlabel("Time (s)")
    a2.legend()
    fig.suptitle("Sustained camera-rate load")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    host = load(BENCH_DIR / "host_cpu.json")
    acc = load(BENCH_DIR / "accuracy_host.json")
    hub = load(BENCH_DIR / "aihub_profile.json")
    device_files = sorted((BENCH_DIR / "device").glob("*.json"))
    if host:
        host_chart(host)
    md = (
        "# Benchmark Results\n\nGenerated by `ml/report.py`. Do not edit by hand.\n\n"
        "## Host CPU (ONNX Runtime)\n\n" + host_section(host, acc)
        + "\n## Snapdragon via Qualcomm AI Hub\n\n" + aihub_section(hub)
        + "\n## On-device (app benchmark mode)\n\n" + device_section(device_files)
    )
    (BENCH_DIR / "RESULTS.md").write_text(md)
    print(f"Wrote {BENCH_DIR / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
