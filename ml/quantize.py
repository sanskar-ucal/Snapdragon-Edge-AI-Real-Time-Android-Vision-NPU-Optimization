"""Creates FP16 and INT8 (static QDQ) variants from each FP32 ONNX model."""
from __future__ import annotations

import argparse
import random

import numpy as np
import onnx
from PIL import Image

from common import RESOLUTIONS, Variant, coco128_images, fp32_path, preprocess

CALIBRATION_IMAGES = 64
SEED = 0


def to_fp16(resolution: int) -> None:
    from onnxconverter_common import float16

    model = onnx.load(fp32_path(resolution))
    # Stale value_info makes the converter mistype outputs of blocked ops (Resize).
    del model.graph.value_info[:]
    # Keep float32 graph I/O so the app can feed the same buffer to every variant.
    model16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model16, Variant(resolution, "fp16").path)
    print(f"Wrote {Variant(resolution, 'fp16').path}")


class CalibrationReader:
    def __init__(self, resolution: int, input_name: str):
        images = coco128_images()
        random.Random(SEED).shuffle(images)
        self._batches = iter(
            [{input_name: preprocess(Image.open(p), resolution)[0]} for p in images[:CALIBRATION_IMAGES]]
        )

    def get_next(self):
        return next(self._batches, None)

    def rewind(self):  # required by newer onnxruntime calibrators
        pass


def head_decode_nodes(model: onnx.ModelProto) -> list[str]:
    """Detect-head decode ops (DFL, sigmoid, box math, concat).

    The final concat mixes pixel-scale boxes (0..640) with 0..1 class scores; a single
    8-bit scale for both destroys score precision, so this tail stays in float.
    """
    head = "/model.23/"
    return [
        n.name
        for n in model.graph.node
        if n.name.startswith(head) and "/cv2." not in n.name and "/cv3." not in n.name
    ]


def to_int8(resolution: int) -> None:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    src = fp32_path(resolution)
    prepped = src.with_name(src.stem + "_prep.onnx")
    quant_pre_process(str(src), str(prepped), skip_symbolic_shape=True)
    model = onnx.load(prepped)
    input_name = model.graph.input[0].name
    exclude = head_decode_nodes(model)

    # QDQ with uint8 activations / int8 per-channel weights is the format the QNN EP consumes.
    quantize_static(
        str(prepped),
        str(Variant(resolution, "int8").path),
        CalibrationReader(resolution, input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=exclude,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    prepped.unlink()
    print(f"Wrote {Variant(resolution, 'int8').path} (kept {len(exclude)} head nodes in float)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(RESOLUTIONS))
    args = parser.parse_args()
    np.random.seed(SEED)
    for r in args.resolutions:
        to_fp16(r)
        to_int8(r)


if __name__ == "__main__":
    main()
