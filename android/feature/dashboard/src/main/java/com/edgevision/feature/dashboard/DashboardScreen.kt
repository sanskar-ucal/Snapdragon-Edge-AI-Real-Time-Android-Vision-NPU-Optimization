package com.edgevision.feature.dashboard

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.edgevision.core.telemetry.LatencySummary
import com.edgevision.core.telemetry.TelemetryRepository

private val Green = Color(0xFF4CAF50)
private val Amber = Color(0xFFFFC107)
private val Blue = Color(0xFF40C4FF)
private val Red = Color(0xFFFF5252)

@Composable
fun DashboardScreen(telemetry: TelemetryRepository, onExport: () -> String) {
    val snap by telemetry.snapshot.collectAsStateWithLifecycle()
    val history by telemetry.history.collectAsStateWithLifecycle()
    val events by telemetry.events.collectAsStateWithLifecycle()
    val recent = history.takeLast(120)

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Text(snap.session.backendLabel, color = if (snap.session.npuVerified) Green else Amber,
                    style = MaterialTheme.typography.titleMedium)
                Text("Model ${snap.session.variantId}")
                Text(snap.session.backendDetail, style = MaterialTheme.typography.bodySmall)
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Metric("FPS", "%.1f".format(snap.fps), Modifier.weight(1f))
            Metric("PSS", "%.0f MB".format(snap.memory.pssMb), Modifier.weight(1f))
            Metric("Thermal", snap.thermal.statusName, Modifier.weight(1f))
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Metric("Java heap", "%.0f MB".format(snap.memory.javaHeapMb), Modifier.weight(1f))
            Metric("Native heap", "%.0f MB".format(snap.memory.nativeHeapMb), Modifier.weight(1f))
            Metric("Battery", snap.thermal.batteryTempC?.let { "%.1f C".format(it) } ?: "n/a", Modifier.weight(1f))
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Text("Latency (ms, rolling ${snap.total.count} frames)", style = MaterialTheme.typography.titleSmall)
                LatencyRow("Total", snap.total)
                LatencyRow("Preprocess (C++)", snap.preprocess)
                LatencyRow("Inference", snap.inference)
                LatencyRow("Postprocess", snap.postprocess)
                Text("Thermal headroom: ${snap.thermal.headroom?.let { "%.2f".format(it) } ?: "n/a"}")
            }
        }
        LineChart("Latency", "ms", listOf(
            Series("p50", Green, recent.map { it.p50Ms }),
            Series("p95", Amber, recent.map { it.p95Ms }),
            Series("p99", Red, recent.map { it.p99Ms }),
        ))
        LineChart("Throughput", "FPS", listOf(Series("FPS", Blue, recent.map { it.fps })))
        LineChart("Memory", "MB", listOf(
            Series("PSS", Blue, recent.map { it.pssMb }),
            Series("Native", Green, recent.map { it.nativeHeapMb }),
        ))
        LineChart("Thermal", "C / status", listOf(
            Series("Battery C", Red, recent.map { (it.batteryTempC ?: 0f).toDouble() }),
            Series("Status x10", Amber, recent.map { it.thermalStatus * 10.0 }),
        ))
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Text("Events", style = MaterialTheme.typography.titleSmall)
                if (events.isEmpty()) Text("none yet", style = MaterialTheme.typography.bodySmall)
                events.takeLast(12).reversed().forEach {
                    Text("[%.0fs] %s".format(it.tSeconds, it.message), style = MaterialTheme.typography.bodySmall)
                }
            }
        }
        var exported by remember { mutableStateOf<String?>(null) }
        Button(onClick = { exported = onExport() }) { Text("Export session JSON") }
        exported?.let { Text("Saved: $it", style = MaterialTheme.typography.bodySmall) }
    }
}

@Composable
private fun Metric(title: String, value: String, modifier: Modifier) {
    Card(modifier) {
        Column(Modifier.padding(10.dp)) {
            Text(title, style = MaterialTheme.typography.labelSmall)
            Text(value, style = MaterialTheme.typography.titleMedium)
        }
    }
}

@Composable
private fun LatencyRow(name: String, s: LatencySummary) {
    Text("%-18s p50 %6.2f  p95 %6.2f  p99 %6.2f".format(name, s.p50, s.p95, s.p99),
        style = MaterialTheme.typography.bodySmall)
}
