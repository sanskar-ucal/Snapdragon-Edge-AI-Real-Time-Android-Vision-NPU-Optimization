"""Measures mAP50 / mAP50-95 of every ONNX variant on COCO128 with ONNX Runtime CPU.

COCO128 is a 128-image proxy for COCO val2017; absolute numbers are higher than the
official benchmark, but relative FP32 -> FP16 -> INT8 deltas are what this tracks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from common import (
    BENCH_DIR,
    Variant,
    all_variants,
    average_precision,
    coco128_images,
    decode,
    environment,
    load_labels,
    match_predictions,
    preprocess,
    write_json,
)


def evaluate(variant: Variant, limit: int | None = None) -> dict:
    session = ort.InferenceSession(str(variant.path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    tps, confs, pred_cls, gt_cls = [], [], [], []
    for path in coco128_images()[:limit]:
        image = Image.open(path).convert("RGB")
        tensor, lb = preprocess(image, variant.resolution)
        output = session.run(None, {input_name: tensor})[0]
        preds = decode(output, lb, conf_threshold=0.001, iou_threshold=0.7)
        gts = load_labels(path, image.width, image.height)
        tps.append(match_predictions(preds, gts))
        confs.append(preds[:, 4])
        pred_cls.append(preds[:, 5])
        gt_cls.append(gts[:, 0])
    ap = average_precision(np.concatenate(tps), np.concatenate(confs), np.concatenate(pred_cls), np.concatenate(gt_cls))
    return {"variant": variant.name, "map50": round(float(ap[0]), 4), "map50_95": round(float(ap.mean()), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="evaluate on the first N images only")
    parser.add_argument("--out", default=str(BENCH_DIR / "accuracy_host.json"))
    args = parser.parse_args()
    results = []
    for v in all_variants():
        if not v.path.exists():
            print(f"skip {v.name}: not built")
            continue
        r = evaluate(v, args.limit)
        print(f"{r['variant']:<20} mAP50={r['map50']:.4f} mAP50-95={r['map50_95']:.4f}")
        results.append(r)
    write_json(
        Path(args.out),
        {"dataset": "coco128", "images": args.limit or 128, "environment": environment(), "results": results},
    )


if __name__ == "__main__":
    main()
