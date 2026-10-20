package com.edgevision.feature.camera

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Paint
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size as GeoSize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.edgevision.core.inference.ModelVariant
import com.edgevision.core.inference.RequestedBackend
import com.edgevision.core.telemetry.TelemetryRepository

@Composable
fun CameraScreen(pipeline: DetectionPipeline, telemetry: TelemetryRepository, variants: List<ModelVariant>) {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }

    if (variants.isEmpty()) {
        Message("No models bundled. Run scripts/make_models.sh, then rebuild the app.")
        return
    }
    if (!granted) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }) { Text("Grant camera permission") }
        }
        return
    }

    Box(Modifier.fillMaxSize()) {
        CameraPreview(pipeline)
        DetectionOverlay(pipeline)
        Column(Modifier.align(Alignment.TopStart).fillMaxWidth()) {
            Hud(pipeline, telemetry)
        }
        Controls(pipeline, variants, Modifier.align(Alignment.BottomCenter))
    }
}

@Composable
private fun CameraPreview(pipeline: DetectionPipeline) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply { scaleType = PreviewView.ScaleType.FIT_CENTER }
    }
    DisposableEffect(lifecycleOwner) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val selector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(
                    ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER),
                )
                .build()
            val preview = Preview.Builder()
                .setResolutionSelector(ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build())
                .build()
                .also { it.surfaceProvider = previewView.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(selector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also { it.setAnalyzer(pipeline.executor, pipeline) }
            provider.unbindAll()
            provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(context))
        onDispose {
            if (providerFuture.isDone) providerFuture.get().unbindAll()
        }
    }
    AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
}

@Composable
private fun DetectionOverlay(pipeline: DetectionPipeline) {
    val result by pipeline.results.collectAsStateWithLifecycle()
    val labelPaint = remember {
        Paint().apply { color = android.graphics.Color.WHITE; textSize = 36f; isAntiAlias = true }
    }
    Canvas(Modifier.fillMaxSize()) {
        if (result.frameWidth == 0) return@Canvas
        // Matches PreviewView FIT_CENTER: the frame is scaled to fit and centered.
        val scale = minOf(size.width / result.frameWidth, size.height / result.frameHeight)
        val w = result.frameWidth * scale
        val h = result.frameHeight * scale
        val ox = (size.width - w) / 2
        val oy = (size.height - h) / 2
        for (d in result.detections) {
            val topLeft = Offset(ox + d.left * w, oy + d.top * h)
            drawRect(
                color = BOX_COLORS[d.classId % BOX_COLORS.size],
                topLeft = topLeft,
                size = GeoSize((d.right - d.left) * w, (d.bottom - d.top) * h),
                style = Stroke(width = 4f),
            )
            drawContext.canvas.nativeCanvas.drawText(
                "${d.label} ${(d.score * 100).toInt()}%", topLeft.x + 6, topLeft.y + 36, labelPaint,
            )
        }
    }
}

@Composable
private fun Hud(pipeline: DetectionPipeline, telemetry: TelemetryRepository) {
    val snap by telemetry.snapshot.collectAsStateWithLifecycle()
    val result by pipeline.results.collectAsStateWithLifecycle()
    val error by pipeline.error.collectAsStateWithLifecycle()
    val verified = result.backend?.npuVerified == true
    Column(
        Modifier.fillMaxWidth().background(Color(0x99000000)).padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        Text(
            result.backend?.label ?: "Loading model...",
            color = if (verified) Color(0xFF4CAF50) else Color(0xFFFFC107),
            style = MaterialTheme.typography.titleSmall,
        )
        Text(
            "%s | %.1f FPS | p50 %.1f / p95 %.1f / p99 %.1f ms".format(
                result.variant?.toString() ?: "-", snap.fps, snap.total.p50, snap.total.p95, snap.total.p99,
            ),
            color = Color.White,
        )
        Text(
            "pre %.1f | infer %.1f | post %.1f ms | %s | %d objects".format(
                snap.preprocess.p50, snap.inference.p50, snap.postprocess.p50, snap.thermal.statusName,
                result.detections.size,
            ),
            color = Color.White,
        )
        error?.let { Text("Error: $it", color = Color(0xFFFF5252)) }
    }
}

@Composable
private fun Controls(pipeline: DetectionPipeline, variants: List<ModelVariant>, modifier: Modifier) {
    val cfg by pipeline.config.collectAsStateWithLifecycle()
    var menu by remember { mutableStateOf(false) }
    Row(
        modifier.fillMaxWidth().background(Color(0x99000000)).padding(8.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        RequestedBackend.entries.forEach { b ->
            FilterChip(
                selected = cfg.backend == b,
                onClick = { pipeline.updateConfig { it.copy(backend = b) } },
                label = { Text(b.name) },
            )
        }
        Text("Adaptive", color = Color.White)
        Switch(checked = cfg.adaptive, onCheckedChange = { on -> pipeline.updateConfig { it.copy(adaptive = on) } })
        Box {
            OutlinedButton(onClick = { menu = true }, enabled = !cfg.adaptive) { Text(cfg.variant.toString()) }
            DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                variants.forEach { v ->
                    DropdownMenuItem(text = { Text(v.toString()) }, onClick = {
                        menu = false
                        pipeline.updateConfig { it.copy(variant = v) }
                    })
                }
            }
        }
    }
}

@Composable
private fun Message(text: String) {
    Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) { Text(text) }
}

private val BOX_COLORS = listOf(
    Color(0xFF00E676), Color(0xFF40C4FF), Color(0xFFFFD740), Color(0xFFFF6E40), Color(0xFFE040FB),
)
