package com.edgevision.core.telemetry

import org.junit.Assert.assertEquals
import org.junit.Test

class LatencyStatsTest {
    @Test
    fun `percentiles match numpy linear interpolation`() {
        val s = LatencyStats.summarize((1..100).map { it.toDouble() })
        assertEquals(50.5, s.p50, 1e-9)
        assertEquals(95.05, s.p95, 1e-9)
        assertEquals(99.01, s.p99, 1e-9)
        assertEquals(50.5, s.mean, 1e-9)
    }

    @Test
    fun `ring buffer keeps only the most recent window`() {
        val stats = LatencyStats(capacity = 4)
        listOf(100.0, 100.0, 1.0, 2.0, 3.0, 4.0).forEach(stats::add)
        val s = stats.summary()
        assertEquals(4, s.count)
        assertEquals(4.0, s.p99, 0.1)
    }

    @Test
    fun `fps counts frames in the trailing second`() {
        val meter = FpsMeter()
        for (i in 0..30) meter.tick(i * 33_333_333L)
        assertEquals(30.0, meter.fps(30 * 33_333_333L), 0.5)
    }
}
