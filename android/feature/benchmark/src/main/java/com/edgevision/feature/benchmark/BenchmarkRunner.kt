package com.edgevision.feature.benchmark

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import com.edgevision.core.inference.ActiveBackend
import com.edgevision.core.inference.BackendInfo
import com.edgevision.core.inference.BuildConfig
import com.edgevision.core.inference.EngineFactory
import com.edgevision.core.inference.InferenceEngine
import com.edgevision.core.inference.ModelRegistry
import com.edgevision.core.inference.ModelVariant
import com.edgevision.core.inference.RequestedBackend
import com.edgevision.core.inference.YoloPostprocessor
import com.edgevision.core.inference.adaptive.AdaptiveController
import com.edgevision.core.inference.adaptive.AdaptiveInput
import com.edgevision.core.nativeproc.NativePreprocessor
import com.edgevision.core.telemetry.DeviceSource
import com.edgevision.core.telemetry.FpsMeter
import com.edgevision.core.telemetry.LatencyStats
import com.edgevision.core.telemetry.LatencySummary
import com.edgevision.core.telemetry.toJson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.coroutines.coroutineContext

data class BenchmarkConfig(
    val warmup: Int = 10,
    val iterations: Int = 100,
    val variants: List<ModelVariant> = ModelRegistry.all,
    val backends: List<RequestedBackend> = RequestedBackend.entries,
    val sustainedSeconds: Int = 0,
    val sustainedAdaptive: Boolean = true,
    val sustainedBackend: RequestedBackend = RequestedBackend.NPU,
    /** Paces the sustained loop like a camera; 0 runs flat out. */
    val targetFps: Int = 30,
)

data class BenchmarkResult(
    val variant: ModelVariant,
    val backend: BackendInfo,
    val loadMs: Long,
    val total: LatencySummary,
    val preprocess: LatencySummary,
    val inference: LatencySummary,
    val postprocess: LatencySummary,
) {
    val fps: Double get() = if (total.mean > 0) 1000.0 / total.mean else 0.0
}

data class BenchmarkReport(val json: JSONObject, val results: List<BenchmarkResult>, val file: File)

/**
 * Reproducible on-device benchmark: fixed bundled images, fixed warmup/iteration counts, every
 * variant x backend, plus an optional sustained-load run that logs FPS/latency/thermal per second.
 */
class BenchmarkRunner(
    private val context: Context,
    private val factory: EngineFactory,
    private val device: DeviceSource,
) {
    suspend fun run(config: BenchmarkConfig, progress: (String) -> Unit = {}): BenchmarkReport =
        withContext(Dispatchers.Default) {
            val images = loadImages()
            NativePreprocessor().use { pre ->
                val results = ArrayList<BenchmarkResult>()
                val skipped = JSONArray()
                for (variant in config.variants) for (backend in config.backends) {
                    coroutineContext.ensureActive()
                    progress("Benchmarking $variant on $backend")
                    factory.create(variant, backend).use { engine ->
                        if (backend == RequestedBackend.NPU && engine.backend.active == ActiveBackend.CPU) {
                            skipped.put(JSONObject().put("variant", variant.id).put("backend", backend.name)
                                .put("reason", engine.backend.detail))
                        } else {
                            results += measure(engine, pre, images, config)
                        }
                    }
                }
                val sustained = if (config.sustainedSeconds > 0) sustained(config, pre, images, progress) else null
                images.forEach(Bitmap::recycle)
                val json = report(config, results, skipped, sustained)
                BenchmarkReport(json, results, save(json))
            }
        }

    private fun measure(
        engine: InferenceEngine,
        pre: NativePreprocessor,
        images: List<Bitmap>,
        config: BenchmarkConfig,
    ): BenchmarkResult {
        val post = YoloPostprocessor()
        val n = config.iterations
        val stats = List(4) { LatencyStats(n) }
        repeat(config.warmup + n) { i ->
            val image = images[i % images.size]
            val t0 = System.nanoTime()
            val lb = pre.fromBitmap(image, engine.variant.resolution, engine.input)
            val t1 = System.nanoTime()
            val out = engine.run()
            val t2 = System.nanoTime()
            post.process(out, engine.variant.numAnchors, lb, image.width, image.height)
            val t3 = System.nanoTime()
            if (i >= config.warmup) {
                stats[0].add((t3 - t0) / 1e6)
                stats[1].add((t1 - t0) / 1e6)
                stats[2].add((t2 - t1) / 1e6)
                stats[3].add((t3 - t2) / 1e6)
            }
        }
        return BenchmarkResult(engine.variant, engine.backend, engine.loadTimeMs,
            stats[0].summary(), stats[1].summary(), stats[2].summary(), stats[3].summary())
    }

    private suspend fun sustained(
        config: BenchmarkConfig,
        pre: NativePreprocessor,
        images: List<Bitmap>,
        progress: (String) -> Unit,
    ): JSONObject {
        val engines = HashMap<ModelVariant, InferenceEngine>()
        fun engine(v: ModelVariant) = engines.getOrPut(v) { factory.create(v, config.sustainedBackend) }

        val probe = engine(ModelRegistry.ladder(ActiveBackend.QNN_HTP).first())
        val ladder = ModelRegistry.ladder(probe.backend.active)
        val controller = AdaptiveController(ladder)
        var current = engine(ladder.first())
        val post = YoloPostprocessor()
        val window = LatencyStats(300)
        val fps = FpsMeter()
        val samples = JSONArray()
        val events = JSONArray()
        val frameNs = if (config.targetFps > 0) 1_000_000_000L / config.targetFps else 0L
        val start = System.nanoTime()
        val end = start + config.sustainedSeconds * 1_000_000_000L
        var nextSample = start + 1_000_000_000L
        var frame = 0

        try {
            while (System.nanoTime() < end) {
                coroutineContext.ensureActive()
                val t0 = System.nanoTime()
                val image = images[frame++ % images.size]
                val lb = pre.fromBitmap(image, current.variant.resolution, current.input)
                post.process(current.run(), current.variant.numAnchors, lb, image.width, image.height)
                val t1 = System.nanoTime()
                window.add((t1 - t0) / 1e6)
                fps.tick(t1)

                if (t1 >= nextSample) {
                    nextSample += 1_000_000_000L
                    val s = window.summary()
                    val thermal = device.thermal()
                    val mem = device.memory()
                    val t = (t1 - start) / 1e9
                    samples.put(JSONObject()
                        .put("t_s", t).put("fps", fps.fps(t1))
                        .put("p50_ms", s.p50).put("p95_ms", s.p95).put("p99_ms", s.p99)
                        .put("thermal_status", thermal.status)
                        .put("headroom", thermal.headroom ?: JSONObject.NULL)
                        .put("battery_temp_c", thermal.batteryTempC ?: JSONObject.NULL)
                        .put("pss_mb", mem.pssMb)
                        .put("variant", current.variant.id))
                    progress("Sustained %.0f/%d s: %.1f FPS, p95 %.1f ms, %s".format(
                        t, config.sustainedSeconds, fps.fps(t1), s.p95, thermal.statusName))
                    if (config.sustainedAdaptive) {
                        val d = controller.evaluate(
                            AdaptiveInput((t1 - start) / 1_000_000, s.p95, s.count, thermal.status, thermal.headroom),
                        )
                        if (d != null) {
                            events.put(JSONObject().put("t_s", t).put("from", d.from.id).put("to", d.to.id)
                                .put("reason", d.reason))
                            current = engine(d.to)
                            window.reset()
                        }
                    }
                }
                if (frameNs > 0) {
                    val sleepNs = frameNs - (System.nanoTime() - t0)
                    if (sleepNs > 0) Thread.sleep(sleepNs / 1_000_000, (sleepNs % 1_000_000).toInt())
                }
            }
        } finally {
            engines.values.forEach(InferenceEngine::close)
        }
        return JSONObject()
            .put("mode", if (config.sustainedAdaptive) "adaptive" else "fixed")
            .put("backend", probe.backend.label)
            .put("target_fps", config.targetFps)
            .put("duration_s", config.sustainedSeconds)
            .put("samples", samples)
            .put("events", events)
    }

    private fun report(
        config: BenchmarkConfig,
        results: List<BenchmarkResult>,
        skipped: JSONArray,
        sustained: JSONObject?,
    ): JSONObject = JSONObject()
        .put("device", JSONObject()
            .put("manufacturer", Build.MANUFACTURER)
            .put("model", Build.MODEL)
            .put("soc", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE)
            .put("android", Build.VERSION.RELEASE)
            .put("ort_version", BuildConfig.ORT_VERSION))
        .put("config", JSONObject()
            .put("warmup", config.warmup).put("iterations", config.iterations)
            .put("images", BENCH_IMAGES).put("target_fps", config.targetFps))
        .put("results", JSONArray(results.map { r ->
            JSONObject()
                .put("variant", r.variant.id)
                .put("backend", r.backend.label)
                .put("npu_verified", r.backend.npuVerified)
                .put("backend_detail", r.backend.detail)
                .put("load_ms", r.loadMs)
                .put("p50_ms", r.total.p50).put("p95_ms", r.total.p95).put("p99_ms", r.total.p99)
                .put("mean_ms", r.total.mean)
                .put("preprocess_p50_ms", r.preprocess.p50)
                .put("inference", r.inference.toJson())
                .put("postprocess_p50_ms", r.postprocess.p50)
                .put("fps", r.fps)
        }))
        .put("skipped", skipped)
        .apply { sustained?.let { put("sustained", it) } }

    private fun save(json: JSONObject): File {
        val dir = File(context.getExternalFilesDir(null), "benchmarks").apply { mkdirs() }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val name = "${Build.MODEL.replace(Regex("[^A-Za-z0-9]+"), "_")}_$stamp.json"
        return File(dir, name).apply { writeText(json.toString(2)) }
    }

    private fun loadImages(): List<Bitmap> {
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        return context.assets.list("bench").orEmpty().sorted().map { name ->
            context.assets.open("bench/$name").use { BitmapFactory.decodeStream(it, null, opts)!! }
        }.also { require(it.isNotEmpty()) { "no benchmark images bundled" } }
    }

    private companion object {
        const val BENCH_IMAGES = "COCO128: 000000000009, 025, 030, 034"
    }
}
