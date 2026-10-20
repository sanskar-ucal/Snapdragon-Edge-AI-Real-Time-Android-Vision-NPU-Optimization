"""Exports YOLO11n from PyTorch to static-shape FP32 ONNX at each target resolution."""
from __future__ import annotations

import argparse
import shutil

from common import ARTIFACTS, RESOLUTIONS, fp32_path

OPSET = 17


def export(resolution: int, weights: str = "yolo11n.pt") -> None:
    from ultralytics import YOLO

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ARTIFACTS / weights) if (ARTIFACTS / weights).exists() else weights)
    # Static shapes and no NMS in-graph: QNN HTP requires fixed shapes, and NMS runs in Kotlin.
    out = model.export(format="onnx", imgsz=resolution, opset=OPSET, simplify=True, dynamic=False, nms=False)
    shutil.move(out, fp32_path(resolution))
    print(f"Exported {fp32_path(resolution)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(RESOLUTIONS))
    args = parser.parse_args()
    for r in args.resolutions:
        export(r)
    pt = ARTIFACTS.parent / "yolo11n.pt"
    if pt.exists():
        shutil.move(pt, ARTIFACTS / "yolo11n.pt")


if __name__ == "__main__":
    main()
