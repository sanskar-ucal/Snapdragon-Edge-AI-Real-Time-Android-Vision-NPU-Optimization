package com.edgevision.core.inference

import com.edgevision.core.nativeproc.Letterbox
import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.FloatBuffer

class YoloPostprocessorTest {
    private val anchors = 4
    private val classes = 80

    private fun head(vararg boxes: FloatArray): FloatBuffer {
        // Each box: cx, cy, w, h, classId, score
        val data = FloatArray((4 + classes) * anchors)
        boxes.forEachIndexed { i, b ->
            for (k in 0 until 4) data[k * anchors + i] = b[k]
            data[(4 + b[4].toInt()) * anchors + i] = b[5]
        }
        return FloatBuffer.wrap(data)
    }

    private val identity = Letterbox(scale = 1f, padX = 0f, padY = 0f)

    @Test
    fun `overlapping same-class boxes are suppressed, keeping the highest score`() {
        val out = head(
            floatArrayOf(50f, 50f, 20f, 20f, 0f, 0.6f),
            floatArrayOf(51f, 51f, 20f, 20f, 0f, 0.9f),
            floatArrayOf(51f, 51f, 20f, 20f, 2f, 0.8f), // different class survives
            floatArrayOf(10f, 10f, 4f, 4f, 0f, 0.1f),   // below threshold
        )
        val dets = YoloPostprocessor().process(out, anchors, identity, 100, 100)
        assertEquals(listOf(0 to 0.9f, 2 to 0.8f), dets.map { it.classId to it.score })
    }

    @Test
    fun `boxes are mapped back through the letterbox into normalized frame coordinates`() {
        // 200x100 frame letterboxed into 100x100: scale 0.5, padY 25.
        val out = head(floatArrayOf(50f, 50f, 100f, 50f, 0f, 0.9f))
        val det = YoloPostprocessor().process(out, anchors, Letterbox(0.5f, 0f, 25f), 200, 100).single()
        assertEquals(0f, det.left, 1e-5f)
        assertEquals(0f, det.top, 1e-5f)
        assertEquals(1f, det.right, 1e-5f)
        assertEquals(1f, det.bottom, 1e-5f)
    }

    @Test
    fun `variant anchors match YOLO strides`() {
        assertEquals(8400, ModelVariant(640, Precision.FP32).numAnchors)
        assertEquals(2100, ModelVariant(320, Precision.INT8).numAnchors)
    }
}
