package com.edgevision.core.inference

import com.edgevision.core.nativeproc.Letterbox
import java.nio.FloatBuffer

/** Box in normalized [0, 1] coordinates of the upright camera frame. */
data class Detection(
    val classId: Int,
    val score: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
) {
    val label: String get() = CocoLabels.name(classId)
}

/**
 * Decodes the YOLO11 head ([1, 4 + classes, anchors], channel-major) and applies class-aware NMS.
 * Class scores are scanned row by row so memory is read sequentially; scratch arrays are reused.
 */
class YoloPostprocessor(
    private val numClasses: Int = 80,
    var confThreshold: Float = 0.35f,
    var iouThreshold: Float = 0.45f,
    private val maxDetections: Int = 100,
) {
    private var bestScore = FloatArray(0)
    private var bestClass = IntArray(0)

    fun process(output: FloatBuffer, numAnchors: Int, lb: Letterbox, frameWidth: Int, frameHeight: Int): List<Detection> {
        if (bestScore.size != numAnchors) {
            bestScore = FloatArray(numAnchors)
            bestClass = IntArray(numAnchors)
        }
        bestScore.fill(0f)
        for (c in 0 until numClasses) {
            val row = (4 + c) * numAnchors
            for (i in 0 until numAnchors) {
                val s = output.get(row + i)
                if (s > bestScore[i]) {
                    bestScore[i] = s
                    bestClass[i] = c
                }
            }
        }

        val candidates = (0 until numAnchors).filter { bestScore[it] > confThreshold }
            .sortedByDescending { bestScore[it] }
        val boxes = Array(candidates.size) { k ->
            val i = candidates[k]
            val cx = output.get(i)
            val cy = output.get(numAnchors + i)
            val w = output.get(2 * numAnchors + i)
            val h = output.get(3 * numAnchors + i)
            floatArrayOf(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        }

        val suppressed = BooleanArray(candidates.size)
        val result = ArrayList<Detection>()
        for (a in candidates.indices) {
            if (suppressed[a]) continue
            val i = candidates[a]
            result += toDetection(bestClass[i], bestScore[i], boxes[a], lb, frameWidth, frameHeight)
            if (result.size >= maxDetections) break
            for (b in a + 1 until candidates.size) {
                if (!suppressed[b] && bestClass[candidates[b]] == bestClass[i] && iou(boxes[a], boxes[b]) > iouThreshold) {
                    suppressed[b] = true
                }
            }
        }
        return result
    }

    private fun toDetection(cls: Int, score: Float, box: FloatArray, lb: Letterbox, fw: Int, fh: Int): Detection {
        fun x(v: Float) = ((v - lb.padX) / lb.scale / fw).coerceIn(0f, 1f)
        fun y(v: Float) = ((v - lb.padY) / lb.scale / fh).coerceIn(0f, 1f)
        return Detection(cls, score, x(box[0]), y(box[1]), x(box[2]), y(box[3]))
    }

    companion object {
        fun iou(a: FloatArray, b: FloatArray): Float {
            val iw = minOf(a[2], b[2]) - maxOf(a[0], b[0])
            val ih = minOf(a[3], b[3]) - maxOf(a[1], b[1])
            if (iw <= 0f || ih <= 0f) return 0f
            val inter = iw * ih
            val union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / union
        }
    }
}
