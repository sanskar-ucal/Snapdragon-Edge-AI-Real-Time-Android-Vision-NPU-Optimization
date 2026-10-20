package com.edgevision

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.edgevision.feature.benchmark.BenchmarkScreen
import com.edgevision.feature.camera.CameraScreen
import com.edgevision.feature.dashboard.DashboardScreen

private enum class Tab(val title: String) { LIVE("Live"), DASHBOARD("Dashboard"), BENCHMARK("Benchmark") }

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val c = (application as EdgeVisionApp).container
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                var tab by rememberSaveable { mutableStateOf(Tab.LIVE) }
                Scaffold(bottomBar = {
                    NavigationBar {
                        Tab.entries.forEach { t ->
                            NavigationBarItem(selected = tab == t, onClick = { tab = t }, icon = {},
                                label = { Text(t.title) })
                        }
                    }
                }) { padding ->
                    Box(Modifier.padding(padding)) {
                        // The camera is only bound while the Live tab is composed, so benchmarks
                        // never compete with the live pipeline for the accelerator.
                        when (tab) {
                            Tab.LIVE -> CameraScreen(c.pipeline, c.telemetry, c.variants)
                            Tab.DASHBOARD -> DashboardScreen(c.telemetry, c::exportSession)
                            Tab.BENCHMARK -> BenchmarkScreen(c.benchmark, c.variants.isNotEmpty())
                        }
                    }
                }
            }
        }
    }
}
