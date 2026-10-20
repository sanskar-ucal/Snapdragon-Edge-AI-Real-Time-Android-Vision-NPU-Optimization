package com.edgevision.core.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtProvider
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * ONNX Runtime session with pre-allocated, zero-copy input and output tensors: the native
 * preprocessor writes into [input] and ORT writes the head output into a pinned buffer.
 */
class OrtEngine internal constructor(
    override val variant: ModelVariant,
    override val backend: BackendInfo,
    override val loadTimeMs: Long,
    private val session: OrtSession,
    env: OrtEnvironment,
) : InferenceEngine {

    override val input: ByteBuffer = ByteBuffer
        .allocateDirect(3 * variant.resolution * variant.resolution * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
    private val output: ByteBuffer = ByteBuffer
        .allocateDirect(84 * variant.numAnchors * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
    private val outputView: FloatBuffer = output.asFloatBuffer()

    private val inputName = session.inputNames.first()
    private val outputName = session.outputNames.first()
    private val inputTensor = OnnxTensor.createTensor(
        env, input.asFloatBuffer(), longArrayOf(1, 3, variant.resolution.toLong(), variant.resolution.toLong()),
    )
    private val outputTensor = OnnxTensor.createTensor(
        env, outputView, longArrayOf(1, 84, variant.numAnchors.toLong()),
    )
    private val inputs = mapOf(inputName to inputTensor)
    private val pinnedOutputs = mapOf(outputName to outputTensor)

    override fun run(): FloatBuffer {
        session.run(inputs, pinnedOutputs).close()
        return outputView
    }

    override fun close() {
        inputTensor.close()
        outputTensor.close()
        session.close()
    }
}

class EngineFactory(private val context: Context, private val cpuThreads: Int = 4) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    val qnnAvailable: Boolean = OrtProvider.QNN in OrtEnvironment.getAvailableProviders()

    /**
     * Tries, in order: QNN HTP with CPU fallback disabled (verified NPU), QNN HTP with fallback
     * (partial, unverified), then the CPU EP. The returned [BackendInfo] records which one won.
     */
    fun create(variant: ModelVariant, requested: RequestedBackend): InferenceEngine {
        val model = materialize(variant)
        if (requested == RequestedBackend.NPU) {
            if (!qnnAvailable) {
                return cpu(variant, model, requested, "QNN execution provider not present in this ORT build")
            }
            try {
                return build(variant, model, qnnOptions(strict = true),
                    BackendInfo(requested, ActiveBackend.QNN_HTP, npuVerified = true,
                        detail = "All nodes assigned to QNN HTP (session.disable_cpu_ep_fallback=1)"))
            } catch (e: OrtException) {
                Log.w(TAG, "Strict QNN HTP session failed for ${variant.id}: ${e.message}")
            }
            try {
                return build(variant, model, qnnOptions(strict = false),
                    BackendInfo(requested, ActiveBackend.QNN_HTP_PARTIAL, npuVerified = false,
                        detail = "Some nodes unsupported by HTP fall back to CPU"))
            } catch (e: OrtException) {
                Log.w(TAG, "QNN HTP session failed for ${variant.id}: ${e.message}")
                return cpu(variant, model, requested, "QNN HTP unavailable: ${e.message?.take(160)}")
            }
        }
        return cpu(variant, model, requested, "ORT CPUExecutionProvider, $cpuThreads threads")
    }

    private fun cpu(variant: ModelVariant, model: File, requested: RequestedBackend, detail: String) =
        build(variant, model, baseOptions(), BackendInfo(requested, ActiveBackend.CPU, npuVerified = false, detail))

    private fun build(variant: ModelVariant, model: File, options: OrtSession.SessionOptions, info: BackendInfo): OrtEngine {
        val start = System.nanoTime()
        options.use {
            val session = env.createSession(model.absolutePath, it)
            return OrtEngine(variant, info, (System.nanoTime() - start) / 1_000_000, session, env)
        }
    }

    private fun baseOptions() = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setIntraOpNumThreads(cpuThreads)
    }

    private fun qnnOptions(strict: Boolean) = baseOptions().apply {
        if (strict) addConfigEntry("session.disable_cpu_ep_fallback", "1")
        addQnn(
            mapOf(
                "backend_path" to "libQnnHtp.so",
                "htp_performance_mode" to "burst",
                "htp_graph_finalization_optimization_mode" to "3",
                // FP32/FP16 graphs execute in FP16 on HTP; QDQ (INT8) graphs run quantized.
                "enable_htp_fp16_precision" to "1",
            ),
        )
    }

    /** ORT needs a file path for mmap; copy the asset once (keyed by size) into no-backup storage. */
    private fun materialize(variant: ModelVariant): File {
        val dest = File(context.noBackupFilesDir, "models/${variant.id}.onnx")
        val assetSize = context.assets.openFd(variant.assetPath).use { it.length }
        if (!dest.exists() || dest.length() != assetSize) {
            dest.parentFile?.mkdirs()
            context.assets.open(variant.assetPath).use { src -> dest.outputStream().use { src.copyTo(it) } }
        }
        return dest
    }

    fun availableVariants(): List<ModelVariant> {
        val present = context.assets.list("models")?.toSet().orEmpty()
        return ModelRegistry.all.filter { "${it.id}.onnx" in present }
    }

    private companion object {
        const val TAG = "EngineFactory"
    }
}
