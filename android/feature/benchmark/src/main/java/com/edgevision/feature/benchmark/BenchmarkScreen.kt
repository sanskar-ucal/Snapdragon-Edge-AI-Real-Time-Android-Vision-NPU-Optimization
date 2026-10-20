package com.edgevision.feature.benchmark

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
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class BenchmarkUiState(
    val running: Boolean = false,
    val progress: String = "",
    val results: List<BenchmarkResult> = emptyList(),
    val savedPath: String? = null,
    val error: String? = null,
)

/** Owns benchmark runs at app scope so they survive navigation. */
class BenchmarkController(private val runner: BenchmarkRunner, private val scope: CoroutineScope) {
    private val _state = MutableStateFlow(BenchmarkUiState())
    val state: StateFlow<BenchmarkUiState> = _state.asStateFlow()
    private var job: Job? = null

    fun start(config: BenchmarkConfig) {
        if (job?.isActive == true) return
        _state.value = BenchmarkUiState(running = true, progress = "Starting")
        job = scope.launch {
            _state.value = try {
                val report = runner.run(config) { msg -> _state.value = _state.value.copy(progress = msg) }
                BenchmarkUiState(results = report.results, savedPath = report.file.absolutePath, progress = "Done")
            } catch (t: Throwable) {
                BenchmarkUiState(error = t.message ?: t.javaClass.simpleName)
            }
        }
    }

    fun cancel() {
        job?.cancel()
        _state.value = _state.value.copy(running = false, progress = "Cancelled")
    }
}

@Composable
fun BenchmarkScreen(controller: BenchmarkController, variantsAvailable: Boolean) {
    val state by controller.state.collectAsStateWithLifecycle()
    var iterations by remember { mutableStateOf(100) }
    var sustained by remember { mutableStateOf(0) }
    var adaptive by remember { mutableStateOf(true) }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text("Benchmark", style = MaterialTheme.typography.titleLarge)
        Text(
            "Every bundled variant on CPU and QNN HTP, using fixed images. The NPU is only reported " +
                "as verified when ONNX Runtime accepts the graph with CPU fallback disabled.",
            style = MaterialTheme.typography.bodySmall,
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(50, 100, 300).forEach {
                FilterChip(selected = iterations == it, onClick = { iterations = it }, label = { Text("$it iters") })
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(0 to "No sustained", 60 to "1 min", 600 to "10 min").forEach { (s, label) ->
                FilterChip(selected = sustained == s, onClick = { sustained = s }, label = { Text(label) })
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(selected = adaptive, onClick = { adaptive = true }, label = { Text("Adaptive") })
            FilterChip(selected = !adaptive, onClick = { adaptive = false }, label = { Text("Fixed model") })
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                enabled = !state.running && variantsAvailable,
                onClick = {
                    controller.start(BenchmarkConfig(iterations = iterations, sustainedSeconds = sustained,
                        sustainedAdaptive = adaptive))
                },
            ) { Text("Run") }
            OutlinedButton(enabled = state.running, onClick = controller::cancel) { Text("Cancel") }
        }
        if (!variantsAvailable) Text("No models bundled. Run scripts/make_models.sh.", color = Color(0xFFFF5252))
        Text(state.progress, style = MaterialTheme.typography.bodySmall)
        state.error?.let { Text("Error: $it", color = Color(0xFFFF5252)) }
        state.savedPath?.let { Text("Saved: $it", style = MaterialTheme.typography.bodySmall) }
        state.results.forEach { r ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(10.dp)) {
                    Text("${r.variant}  ${r.backend.label}",
                        color = if (r.backend.npuVerified) Color(0xFF4CAF50) else Color.Unspecified,
                        style = MaterialTheme.typography.titleSmall)
                    Text("p50 %.2f  p95 %.2f  p99 %.2f ms  |  %.1f FPS".format(
                        r.total.p50, r.total.p95, r.total.p99, r.fps))
                    Text("pre %.2f  infer %.2f  post %.2f ms  |  load %d ms".format(
                        r.preprocess.p50, r.inference.p50, r.postprocess.p50, r.loadMs),
                        style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}
