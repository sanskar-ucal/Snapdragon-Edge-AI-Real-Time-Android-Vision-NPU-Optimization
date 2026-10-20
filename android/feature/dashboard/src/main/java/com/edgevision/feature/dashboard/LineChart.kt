package com.edgevision.feature.dashboard

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp

data class Series(val name: String, val color: Color, val values: List<Double>)

/** Minimal multi-series line chart sharing one y-axis, scaled to the visible max. */
@Composable
fun LineChart(title: String, unit: String, series: List<Series>, modifier: Modifier = Modifier) {
    val max = series.flatMap { it.values }.filter { it.isFinite() }.maxOrNull()?.takeIf { it > 0 } ?: 1.0
    Column(modifier.fillMaxWidth().padding(vertical = 6.dp)) {
        Text("$title (max %.1f %s)".format(max, unit), style = MaterialTheme.typography.titleSmall)
        Row {
            series.forEach { s ->
                Text("  ${s.name}: %.1f".format(s.values.lastOrNull() ?: 0.0), color = s.color,
                    style = MaterialTheme.typography.bodySmall)
            }
        }
        Canvas(Modifier.fillMaxWidth().height(120.dp)) {
            drawLine(Color.Gray, Offset(0f, size.height), Offset(size.width, size.height))
            series.forEach { s ->
                if (s.values.size < 2) return@forEach
                val step = size.width / (s.values.size - 1)
                val path = Path()
                s.values.forEachIndexed { i, v ->
                    val y = size.height * (1f - (v / max).toFloat().coerceIn(0f, 1f))
                    if (i == 0) path.moveTo(0f, y) else path.lineTo(i * step, y)
                }
                drawPath(path, s.color, style = Stroke(width = 3f))
            }
        }
    }
}
