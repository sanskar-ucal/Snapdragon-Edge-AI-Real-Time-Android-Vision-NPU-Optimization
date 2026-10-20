package com.edgevision.core.nativeproc

import android.graphics.Bitmap
import java.nio.ByteBuffer

/** Maps detections from model input space back to the upright source frame. */
data class Letterbox(val scale: Float, val padX: Float, val padY: Float)

/**
 * JNI front-end for the C++ preprocessor. Writes a letterboxed NCHW tensor straight into a
 * caller-owned direct [ByteBuffer] (typically the ONNX Runtime input tensor), so frames go
 * from camera planes to model input with one pass and no JVM-side copies.
 *
 * Not thread-safe: use one instance per analysis thread.
 */
class NativePreprocessor : AutoCloseable {
    private var handle: Long = nativeCreate()
    private val letterbox = FloatArray(3)

    fun fromYuv(
        y: ByteBuffer,
        u: ByteBuffer,
        v: ByteBuffer,
        width: Int,
        height: Int,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        rotationDegrees: Int,
        inputSize: Int,
        out: ByteBuffer,
        fp16: Boolean = false,
    ): Letterbox {
        check(handle != 0L) { "closed" }
        val ok = nativeFromYuv(
            handle, y, u, v, width, height, yRowStride, uvRowStride, uvPixelStride,
            rotationDegrees, inputSize, fp16, out, letterbox,
        )
        require(ok) { "YUV preprocessing failed (non-direct or undersized buffer)" }
        return Letterbox(letterbox[0], letterbox[1], letterbox[2])
    }

    fun fromBitmap(bitmap: Bitmap, inputSize: Int, out: ByteBuffer, fp16: Boolean = false): Letterbox {
        check(handle != 0L) { "closed" }
        require(bitmap.config == Bitmap.Config.ARGB_8888) { "bitmap must be ARGB_8888" }
        require(nativeFromBitmap(handle, bitmap, inputSize, fp16, out, letterbox)) { "bitmap preprocessing failed" }
        return Letterbox(letterbox[0], letterbox[1], letterbox[2])
    }

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private companion object {
        init {
            System.loadLibrary("edgevision_native")
        }

        @JvmStatic external fun nativeCreate(): Long
        @JvmStatic external fun nativeDestroy(handle: Long)

        @JvmStatic external fun nativeFromYuv(
            handle: Long, y: ByteBuffer, u: ByteBuffer, v: ByteBuffer, width: Int, height: Int,
            yRowStride: Int, uvRowStride: Int, uvPixelStride: Int, rotation: Int, dst: Int,
            fp16: Boolean, out: ByteBuffer, letterbox: FloatArray,
        ): Boolean

        @JvmStatic external fun nativeFromBitmap(
            handle: Long, bitmap: Bitmap, dst: Int, fp16: Boolean, out: ByteBuffer, letterbox: FloatArray,
        ): Boolean
    }
}
