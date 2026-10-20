package com.edgevision.core.inference.adaptive

import com.edgevision.core.inference.ActiveBackend
import com.edgevision.core.inference.ModelRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class AdaptiveControllerTest {
    private val ladder = ModelRegistry.ladder(ActiveBackend.QNN_HTP)

    private fun input(t: Long, p95: Double, thermal: Int = 0, samples: Int = 100) =
        AdaptiveInput(nowMs = t, p95Ms = p95, samples = samples, thermalStatus = thermal, headroom = null)

    @Test
    fun `steps down when latency exceeds budget`() {
        val c = AdaptiveController(ladder)
        assertNotNull(c.evaluate(input(0, p95 = 50.0)))
        assertEquals(1, c.index)
    }

    @Test
    fun `steps down on thermal pressure even when fast`() {
        val c = AdaptiveController(ladder)
        val d = c.evaluate(input(0, p95 = 5.0, thermal = 2))
        assertEquals(ladder[1], d?.to)
    }

    @Test
    fun `ignores decisions until enough samples and respects cooldown`() {
        val c = AdaptiveController(ladder)
        assertNull(c.evaluate(input(0, p95 = 50.0, samples = 5)))
        c.evaluate(input(0, p95 = 50.0))
        assertNull(c.evaluate(input(1_000, p95 = 50.0)))
        assertNotNull(c.evaluate(input(3_000, p95 = 50.0)))
        assertEquals(2, c.index)
        assertNull(c.evaluate(input(10_000, p95 = 50.0))) // already at the bottom rung
    }

    @Test
    fun `steps back up only after the longer cooldown with headroom (hysteresis)`() {
        val c = AdaptiveController(ladder)
        c.evaluate(input(0, p95 = 50.0))
        assertNull(c.evaluate(input(5_000, p95 = 10.0)))   // fast, but within up-cooldown
        assertNull(c.evaluate(input(20_000, p95 = 25.0)))  // within budget but not enough margin
        assertNotNull(c.evaluate(input(20_000, p95 = 10.0)))
        assertEquals(0, c.index)
    }
}
