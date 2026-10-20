package com.edgevision

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.edgevision.core.inference.EngineFactory
import com.edgevision.core.telemetry.DeviceMonitor
import com.edgevision.feature.benchmark.BenchmarkConfig
import com.edgevision.feature.benchmark.BenchmarkRunner
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Device benchmark entry point used by scripts/run_device_bench.sh. Instrumentation args:
 * iterations, warmup, sustainedSeconds, adaptive (true/false).
 */
@RunWith(AndroidJUnit4::class)
class BenchmarkInstrumentedTest {
    @Test
    fun benchmarkAllVariants() = runBlocking {
        val args = InstrumentationRegistry.getArguments()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val factory = EngineFactory(context)
        val variants = factory.availableVariants()
        assertTrue("no models bundled", variants.isNotEmpty())

        val config = BenchmarkConfig(
            warmup = args.getString("warmup")?.toInt() ?: 5,
            iterations = args.getString("iterations")?.toInt() ?: 20,
            variants = variants,
            sustainedSeconds = args.getString("sustainedSeconds")?.toInt() ?: 0,
            sustainedAdaptive = args.getString("adaptive")?.toBoolean() ?: true,
        )
        val report = BenchmarkRunner(context, factory, DeviceMonitor(context)).run(config) { println(it) }

        assertTrue(report.results.isNotEmpty())
        report.results.forEach {
            assertTrue("${it.variant}: percentiles out of order", it.total.p50 <= it.total.p95 && it.total.p95 <= it.total.p99)
        }
        println("BENCHMARK_REPORT=${report.file.absolutePath}")
    }
}
