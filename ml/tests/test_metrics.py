import numpy as np
from PIL import Image

from common import IOU_THRESHOLDS, average_precision, box_iou, letterbox_params, match_predictions, nms, preprocess


def test_box_iou_identity_and_disjoint():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]], dtype=np.float32)
    np.testing.assert_allclose(box_iou(a, b)[0], [1.0, 0.0, 1 / 3], atol=1e-6)


def test_nms_suppresses_overlaps_keeps_highest():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    keep = nms(boxes, np.array([0.6, 0.9, 0.5]), iou_threshold=0.5)
    assert keep.tolist() == [1, 2]


def test_letterbox_matches_native_math():
    lb = letterbox_params(640, 480, 320)
    assert (lb.scale, lb.pad_x, lb.pad_y) == (0.5, 0, 40)
    tensor, _ = preprocess(Image.new("RGB", (640, 480), (255, 0, 0)), 320)
    assert tensor.shape == (1, 3, 320, 320)
    assert np.isclose(tensor[0, 0, 0, 0], 114 / 255) and np.isclose(tensor[0, 0, 160, 160], 1.0)


def test_perfect_predictions_give_ap_one():
    gts = np.array([[0, 0, 0, 10, 10], [1, 20, 20, 40, 40]], dtype=np.float32)
    preds = np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 40, 40, 0.8, 1]], dtype=np.float32)
    tp = match_predictions(preds, gts)
    assert tp.all() and tp.shape == (2, len(IOU_THRESHOLDS))
    ap = average_precision(tp, preds[:, 4], preds[:, 5], gts[:, 0])
    np.testing.assert_allclose(ap, 1.0)
