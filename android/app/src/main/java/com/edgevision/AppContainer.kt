package com.edgevision

import android.content.Context
import com.edgevision.core.inference.EngineFactory
import com.edgevision.core.telemetry.DeviceMonitor
import com.edgevision.core.telemetry.TelemetryRepository
import com.edgevision.core.telemetry.toJson
import com.edgevision.feature.benchmark.BenchmarkController
import com.edgevision.feature.benchmark.BenchmarkRunner
import com.edgevision.feature.camera.DetectionPipeline
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.Dispatchers
import java.io.File

/** Manual dependency graph; one instance per process. */
class AppContainer(private val context: Context) {
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    val device = DeviceMonitor(context)
    val telemetry = TelemetryRepository(device).also { it.start(scope) }
    val engineFactory = EngineFactory(context)
    val variants = engineFactory.availableVariants()
    val pipeline = DetectionPipeline(engineFactory, telemetry)
    val benchmark = BenchmarkController(BenchmarkRunner(context, engineFactory, device), scope)

    fun exportSession(): String {
        val dir = File(context.getExternalFilesDir(null), "sessions").apply { mkdirs() }
        val file = File(dir, "session_${System.currentTimeMillis()}.json")
        file.writeText(telemetry.toJson().toString(2))
        return file.absolutePath
    }
}
