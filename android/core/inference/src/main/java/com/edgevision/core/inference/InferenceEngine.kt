package com.edgevision.core.inference

import java.nio.ByteBuffer
import java.nio.FloatBuffer

interface InferenceEngine : AutoCloseable {
    val variant: ModelVariant
    val backend: BackendInfo
    val loadTimeMs: Long

    /** Direct, native-order buffer bound to the model input; fill it, then call [run]. */
    val input: ByteBuffer

    /** Runs inference and returns the [1, 84, numAnchors] head output (reused between calls). */
    fun run(): FloatBuffer
}
