package com.edgevision.core.telemetry

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

data class SessionInfo(
    val variantId: String = "-",
    val backendLabel: String = "-",
    val npuVerified: Boolean = false,
    val backendDetail: String = "",
)

data class TelemetrySnapshot(
    val fps: Double = 0.0,
    val total: LatencySummary = LatencySummary(),
    val preprocess: LatencySummary = LatencySummary(),
    val inference: LatencySummary = LatencySummary(),
    val postprocess: LatencySummary = LatencySummary(),
    val memory: MemoryStats = MemoryStats(),
    val thermal: ThermalStats = ThermalStats(),
    val session: SessionInfo = SessionInfo(),
)

data class TelemetrySample(
    val tSeconds: Double,
    val fps: Double,
    val p50Ms: Double,
    val p95Ms: Double,
    val p99Ms: Double,
    val pssMb: Double,
    val nativeHeapMb: Double,
    val thermalStatus: Int,
    val headroom: Float?,
    val batteryTempC: Float?,
    val variantId: String,
)

data class TelemetryEvent(val tSeconds: Double, val message: String)

/**
 * Single source of truth for runtime performance. The analysis thread calls [recordFrame];
 * a sampler coroutine publishes a [TelemetrySnapshot] and appends a history point each period.
 */
class TelemetryRepository(
    private val device: DeviceSource,
    private val historySize: Int = 600,
    private val clockNs: () -> Long = System::nanoTime,
) {
    val preprocess = LatencyStats()
    val inference = LatencyStats()
    val postprocess = LatencyStats()
    val total = LatencyStats()
    private val fpsMeter = FpsMeter()
    private val startNs = clockNs()

    @Volatile var session = SessionInfo()
        private set

    private val _snapshot = MutableStateFlow(TelemetrySnapshot())
    val snapshot: StateFlow<TelemetrySnapshot> = _snapshot.asStateFlow()

    private val _history = MutableStateFlow<List<TelemetrySample>>(emptyList())
    val history: StateFlow<List<TelemetrySample>> = _history.asStateFlow()

    private val _events = MutableStateFlow<List<TelemetryEvent>>(emptyList())
    val events: StateFlow<List<TelemetryEvent>> = _events.asStateFlow()

    fun recordFrame(preMs: Double, inferMs: Double, postMs: Double) {
        preprocess.add(preMs)
        inference.add(inferMs)
        postprocess.add(postMs)
        total.add(preMs + inferMs + postMs)
        fpsMeter.tick(clockNs())
    }

    fun updateSession(info: SessionInfo) {
        session = info
    }

    fun logEvent(message: String) {
        _events.update { (it + TelemetryEvent(elapsedSeconds(), message)).takeLast(100) }
    }

    /** Clears latency windows, e.g. after a model switch so stale samples don't drive decisions. */
    fun resetLatency() {
        listOf(preprocess, inference, postprocess, total).forEach(LatencyStats::reset)
    }

    fun sample(): TelemetrySnapshot {
        val snap = TelemetrySnapshot(
            fps = fpsMeter.fps(clockNs()),
            total = total.summary(),
            preprocess = preprocess.summary(),
            inference = inference.summary(),
            postprocess = postprocess.summary(),
            memory = device.memory(),
            thermal = device.thermal(),
            session = session,
        )
        _snapshot.value = snap
        val point = TelemetrySample(
            tSeconds = elapsedSeconds(),
            fps = snap.fps,
            p50Ms = snap.total.p50,
            p95Ms = snap.total.p95,
            p99Ms = snap.total.p99,
            pssMb = snap.memory.pssMb,
            nativeHeapMb = snap.memory.nativeHeapMb,
            thermalStatus = snap.thermal.status,
            headroom = snap.thermal.headroom,
            batteryTempC = snap.thermal.batteryTempC,
            variantId = snap.session.variantId,
        )
        _history.update { (it + point).takeLast(historySize) }
        return snap
    }

    fun start(scope: CoroutineScope, periodMs: Long = 1000): Job = scope.launch(Dispatchers.Default) {
        while (isActive) {
            sample()
            delay(periodMs)
        }
    }

    private fun elapsedSeconds() = (clockNs() - startNs) / 1e9
}
