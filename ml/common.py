"""Shared paths, preprocessing, decoding and metrics for the model pipeline."""
from __future__ import annotations

import json
import platform
import shutil
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import certifi
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "ml"
ARTIFACTS = ML_DIR / "artifacts"
DATA_DIR = ML_DIR / "data"
BENCH_DIR = ROOT / "benchmarks"
ANDROID_ASSETS = ROOT / "android" / "core" / "inference" / "src" / "main" / "assets" / "models"

RESOLUTIONS = (320, 640)
PRECISIONS = ("fp32", "fp16", "int8")
PAD_VALUE = 114
NUM_CLASSES = 80

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


@dataclass(frozen=True)
class Variant:
    resolution: int
    precision: str

    @property
    def name(self) -> str:
        return f"yolo11n_{self.resolution}_{self.precision}"

    @property
    def path(self) -> Path:
        return ARTIFACTS / f"{self.name}.onnx"


def all_variants() -> list[Variant]:
    return [Variant(r, p) for r in RESOLUTIONS for p in PRECISIONS]


def fp32_path(resolution: int) -> Path:
    return Variant(resolution, "fp32").path


def ensure_coco128() -> Path:
    """Downloads COCO128 (128 images, YOLO-format labels) if missing."""
    target = DATA_DIR / "coco128"
    if (target / "images" / "train2017").exists():
        return target
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / "coco128.zip"
    print(f"Downloading {COCO128_URL}")
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(COCO128_URL, context=context) as resp, open(archive, "wb") as f:
        shutil.copyfileobj(resp, f)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA_DIR)
    archive.unlink()
    return target


def coco128_images() -> list[Path]:
    return sorted((ensure_coco128() / "images" / "train2017").glob("*.jpg"))


def load_labels(image_path: Path, width: int, height: int) -> np.ndarray:
    """Returns ground truth as [N, 5] = (cls, x1, y1, x2, y2) in pixels."""
    label_path = image_path.parents[2] / "labels" / "train2017" / (image_path.stem + ".txt")
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = np.loadtxt(label_path, ndmin=2, dtype=np.float32)
    cls, cx, cy, w, h = rows.T
    out = np.stack(
        [cls, (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height],
        axis=1,
    )
    return out.astype(np.float32)


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int


def letterbox_params(width: int, height: int, size: int) -> Letterbox:
    """Same math as the C++ preprocessor (native/preprocess.cpp)."""
    scale = min(size / width, size / height)
    new_w = int(width * scale + 0.5)
    new_h = int(height * scale + 0.5)
    return Letterbox(scale, (size - new_w) // 2, (size - new_h) // 2)


def preprocess(image: Image.Image, size: int) -> tuple[np.ndarray, Letterbox]:
    """RGB image -> [1, 3, size, size] float32 in [0, 1], letterboxed with gray padding."""
    image = image.convert("RGB")
    lb = letterbox_params(image.width, image.height, size)
    new_w = int(image.width * lb.scale + 0.5)
    new_h = int(image.height * lb.scale + 0.5)
    canvas = Image.new("RGB", (size, size), (PAD_VALUE,) * 3)
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (lb.pad_x, lb.pad_y))
    chw = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return chw[None], lb


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between [N, 4] and [M, 4] xyxy boxes."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def decode(
    output: np.ndarray,
    lb: Letterbox,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
) -> np.ndarray:
    """YOLO11 head [1, 84, N] -> [K, 6] = (x1, y1, x2, y2, score, cls) in original image pixels."""
    pred = output[0].astype(np.float32).T  # [N, 84]
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(len(cls)), cls]
    mask = scores > conf_threshold
    if not mask.any():
        return np.zeros((0, 6), dtype=np.float32)
    pred, cls, scores = pred[mask], cls[mask], scores[mask]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    offset = cls[:, None].astype(np.float32) * 4096.0  # class-aware NMS
    keep = nms(boxes + offset, scores, iou_threshold)[:max_det]
    boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - lb.pad_x) / lb.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - lb.pad_y) / lb.scale
    return np.concatenate([boxes, scores[:, None], cls[:, None].astype(np.float32)], axis=1)


IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def match_predictions(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Returns [K, 10] bool true-positive matrix over IOU_THRESHOLDS (greedy, class-aware)."""
    tp = np.zeros((len(preds), len(IOU_THRESHOLDS)), dtype=bool)
    if len(preds) == 0 or len(gts) == 0:
        return tp
    iou = box_iou(preds[:, :4], gts[:, 1:5])
    iou = iou * (preds[:, 5:6] == gts[None, :, 0])
    order = preds[:, 4].argsort()[::-1]
    for t, thr in enumerate(IOU_THRESHOLDS):
        taken = np.zeros(len(gts), dtype=bool)
        for i in order:
            candidates = np.where((iou[i] >= thr) & ~taken)[0]
            if candidates.size:
                j = candidates[iou[i, candidates].argmax()]
                taken[j] = True
                tp[i, t] = True
    return tp


def average_precision(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, gt_cls: np.ndarray) -> np.ndarray:
    """COCO-style 101-point AP per IoU threshold, averaged over classes present in GT. Returns [10]."""
    classes = np.unique(gt_cls)
    if classes.size == 0:
        return np.zeros(len(IOU_THRESHOLDS))
    order = conf.argsort()[::-1]
    tp, pred_cls = tp[order], pred_cls[order]
    recall_points = np.linspace(0, 1, 101)
    ap = np.zeros((len(classes), len(IOU_THRESHOLDS)))
    for ci, c in enumerate(classes):
        sel = pred_cls == c
        n_gt = int((gt_cls == c).sum())
        if not sel.any():
            continue
        tpc = np.cumsum(tp[sel], axis=0)
        fpc = np.cumsum(~tp[sel], axis=0)
        recall = tpc / n_gt
        precision = tpc / (tpc + fpc)
        for t in range(len(IOU_THRESHOLDS)):
            p = np.maximum.accumulate(precision[::-1, t])[::-1]
            idx = np.searchsorted(recall[:, t], recall_points, side="left")
            ap[ci, t] = np.where(idx < len(p), p[np.minimum(idx, len(p) - 1)], 0.0).mean()
    return ap.mean(axis=0)


def environment() -> dict:
    import onnxruntime as ort

    return {
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
