package com.edgevision.feature.camera

import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.edgevision.core.inference.ActiveBackend
import com.edgevision.core.inference.BackendInfo
import com.edgevision.core.inference.Detection
import com.edgevision.core.inference.EngineFactory
import com.edgevision.core.inference.InferenceEngine
import com.edgevision.core.inference.ModelRegistry
import com.edgevision.core.inference.ModelVariant
import com.edgevision.core.inference.Precision
import com.edgevision.core.inference.RequestedBackend
import com.edgevision.core.inference.YoloPostprocessor
import com.edgevision.core.inference.adaptive.AdaptiveController
import com.edgevision.core.inference.adaptive.AdaptiveInput
import com.edgevision.core.nativeproc.NativePreprocessor
import com.edgevision.core.telemetry.SessionInfo
import com.edgevision.core.telemetry.TelemetryRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

data class PipelineConfig(
    val backend: RequestedBackend = RequestedBackend.NPU,
    val variant: ModelVariant = ModelVariant(640, Precision.INT8),
    val adaptive: Boolean = true,
)

data class FrameResult(
    val detections: List<Detection> = emptyList(),
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
    val variant: ModelVariant? = null,
    val backend: BackendInfo? = null,
)

/**
 * CameraX analyzer: native preprocessing -> ONNX Runtime -> NMS, with per-stage timing fed to
 * telemetry and an [AdaptiveController] that swaps model variants under load. All engine work
 * happens on the single [executor] thread, so engines are never used concurrently.
 */
class DetectionPipeline(
    private val factory: EngineFactory,
    private val telemetry: TelemetryRepository,
) : ImageAnalysis.Analyzer {

    val executor: ExecutorService = Executors.newSingleThreadExecutor { Thread(it, "detection") }

    private val _config = MutableStateFlow(PipelineConfig())
    val config: StateFlow<PipelineConfig> = _config.asStateFlow()

    private val _results = MutableStateFlow(FrameResult())
    val results: StateFlow<FrameResult> = _results.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    private var preprocessor: NativePreprocessor? = null
    private val postprocessor = YoloPostprocessor()
    private val engines = HashMap<Pair<ModelVariant, RequestedBackend>, InferenceEngine>()
    private var appliedConfig: PipelineConfig? = null
    private var controller: AdaptiveController? = null
    private var engine: InferenceEngine? = null
    private var lastAdaptiveCheckMs = 0L

    fun updateConfig(transform: (PipelineConfig) -> PipelineConfig) = _config.update(transform)

    override fun analyze(image: ImageProxy) {
        try {
            processFrame(image)
            _error.value = null
        } catch (t: Throwable) {
            Log.e(TAG, "frame failed", t)
            _error.value = t.message ?: t.javaClass.simpleName
        } finally {
            image.close()
        }
    }

    private fun processFrame(image: ImageProxy) {
        val engine = currentEngine()
        val pre = preprocessor ?: NativePreprocessor().also { preprocessor = it }
        val rotation = image.imageInfo.rotationDegrees
        val planes = image.planes

        val t0 = System.nanoTime()
        val lb = pre.fromYuv(
            planes[0].buffer, planes[1].buffer, planes[2].buffer, image.width, image.height,
            planes[0].rowStride, planes[1].rowStride, planes[1].pixelStride,
            rotation, engine.variant.resolution, engine.input,
        )
        val t1 = System.nanoTime()
        val output = engine.run()
        val t2 = System.nanoTime()
        val upright = rotation == 90 || rotation == 270
        val fw = if (upright) image.height else image.width
        val fh = if (upright) image.width else image.height
        val detections = postprocessor.process(output, engine.variant.numAnchors, lb, fw, fh)
        val t3 = System.nanoTime()

        telemetry.recordFrame((t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6)
        _results.value = FrameResult(detections, fw, fh, engine.variant, engine.backend)
        maybeAdapt()
    }

    private fun currentEngine(): InferenceEngine {
        val cfg = _config.value
        if (cfg != appliedConfig) {
            appliedConfig = cfg
            controller = null
            engine = if (cfg.adaptive) startAdaptive(cfg.backend) else load(cfg.variant, cfg.backend)
        }
        return engine!!
    }

    private fun startAdaptive(requested: RequestedBackend): InferenceEngine {
        val guess = if (requested == RequestedBackend.NPU) ActiveBackend.QNN_HTP else ActiveBackend.CPU
        var first = load(ModelRegistry.ladder(guess).first(), requested)
        val ladder = ModelRegistry.ladder(first.backend.active)
        if (ladder.first() != first.variant) first = load(ladder.first(), requested)
        controller = AdaptiveController(ladder)
        return first
    }

    private fun load(variant: ModelVariant, requested: RequestedBackend): InferenceEngine {
        val e = engines.getOrPut(variant to requested) {
            factory.create(variant, requested).also {
                telemetry.logEvent("Loaded ${variant.id} on ${it.backend.label} in ${it.loadTimeMs} ms")
            }
        }
        telemetry.updateSession(SessionInfo(variant.id, e.backend.label, e.backend.npuVerified, e.backend.detail))
        telemetry.resetLatency()
        return e
    }

    private fun maybeAdapt() {
        val c = controller ?: return
        val now = SystemClock.elapsedRealtime()
        if (now - lastAdaptiveCheckMs < ADAPT_PERIOD_MS) return
        lastAdaptiveCheckMs = now
        val total = telemetry.total.summary()
        val thermal = telemetry.snapshot.value.thermal
        val decision = c.evaluate(AdaptiveInput(now, total.p95, total.count, thermal.status, thermal.headroom)) ?: return
        telemetry.logEvent("Adaptive: ${decision.from} -> ${decision.to} (${decision.reason})")
        engine = load(decision.to, appliedConfig!!.backend)
    }

    /** Releases native resources on the analysis thread. */
    fun release() {
        executor.execute {
            engines.values.forEach(InferenceEngine::close)
            engines.clear()
            preprocessor?.close()
            preprocessor = null
            appliedConfig = null
            engine = null
        }
    }

    private companion object {
        const val TAG = "DetectionPipeline"
        const val ADAPT_PERIOD_MS = 1_000L
    }
}
