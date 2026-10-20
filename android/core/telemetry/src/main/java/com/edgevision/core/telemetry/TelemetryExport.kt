package com.edgevision.core.telemetry

import org.json.JSONArray
import org.json.JSONObject

fun LatencySummary.toJson(): JSONObject = JSONObject()
    .put("p50_ms", p50).put("p95_ms", p95).put("p99_ms", p99).put("mean_ms", mean).put("count", count)

fun TelemetrySample.toJson(): JSONObject = JSONObject()
    .put("t_s", tSeconds)
    .put("fps", fps)
    .put("p50_ms", p50Ms)
    .put("p95_ms", p95Ms)
    .put("p99_ms", p99Ms)
    .put("pss_mb", pssMb)
    .put("native_heap_mb", nativeHeapMb)
    .put("thermal_status", thermalStatus)
    .put("headroom", headroom ?: JSONObject.NULL)
    .put("battery_temp_c", batteryTempC ?: JSONObject.NULL)
    .put("variant", variantId)

/** Live-session export: latest snapshot, full time series and adaptive events. */
fun TelemetryRepository.toJson(): JSONObject {
    val snap = snapshot.value
    return JSONObject()
        .put("session", JSONObject()
            .put("variant", snap.session.variantId)
            .put("backend", snap.session.backendLabel)
            .put("npu_verified", snap.session.npuVerified)
            .put("backend_detail", snap.session.backendDetail))
        .put("latency", JSONObject()
            .put("total", snap.total.toJson())
            .put("preprocess", snap.preprocess.toJson())
            .put("inference", snap.inference.toJson())
            .put("postprocess", snap.postprocess.toJson()))
        .put("samples", JSONArray(history.value.map { it.toJson() }))
        .put("events", JSONArray(events.value.map { JSONObject().put("t_s", it.tSeconds).put("message", it.message) }))
}
