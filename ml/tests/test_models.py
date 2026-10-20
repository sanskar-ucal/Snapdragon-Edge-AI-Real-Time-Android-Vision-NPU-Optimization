"""Numerical-correctness and accuracy regression gates for exported models."""
import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

from common import ARTIFACTS, Variant, coco128_images, preprocess
from conftest import require
from evaluate import evaluate

RES = 320
FP32, FP16, INT8 = (Variant(RES, p) for p in ("fp32", "fp16", "int8"))
EVAL_IMAGES = 32
MAX_FP16_MAP_DROP = 0.01
MAX_INT8_MAP_DROP = 0.05


def run(variant: Variant, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(variant.path), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: x})[0]


@pytest.fixture(scope="module")
def sample() -> np.ndarray:
    return preprocess(Image.open(coco128_images()[0]), RES)[0]


def test_fp32_onnx_matches_pytorch(sample):
    require(FP32)
    torch = pytest.importorskip("torch")
    from ultralytics import YOLO

    model = YOLO(str(ARTIFACTS / "yolo11n.pt")).model.float().eval()
    with torch.no_grad():
        ref = model(torch.from_numpy(sample))[0].numpy()
    out = run(FP32, sample)
    assert out.shape == ref.shape == (1, 84, 2100)
    np.testing.assert_allclose(out[:, 4:], ref[:, 4:], atol=1e-3)  # class scores
    np.testing.assert_allclose(out[:, :4], ref[:, :4], atol=5e-2)  # pixel-space boxes


def test_fp16_output_close_to_fp32(sample):
    require(FP32, FP16)
    np.testing.assert_allclose(run(FP16, sample)[:, 4:], run(FP32, sample)[:, 4:], atol=2e-2)


@pytest.mark.parametrize("variant, max_drop", [(FP16, MAX_FP16_MAP_DROP), (INT8, MAX_INT8_MAP_DROP)])
def test_quantized_map_drop_within_budget(variant, max_drop):
    require(FP32, variant)
    base = evaluate(FP32, EVAL_IMAGES)["map50_95"]
    quant = evaluate(variant, EVAL_IMAGES)["map50_95"]
    assert base - quant <= max_drop, f"{variant.name}: mAP50-95 {quant:.4f} vs FP32 {base:.4f}"
