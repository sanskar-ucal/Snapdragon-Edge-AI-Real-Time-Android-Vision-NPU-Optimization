package com.edgevision.core.telemetry

import kotlin.math.floor

data class LatencySummary(
    val p50: Double = 0.0,
    val p95: Double = 0.0,
    val p99: Double = 0.0,
    val mean: Double = 0.0,
    val count: Int = 0,
)

/** Rolling window of latency samples (ms) with numpy-compatible linear-interpolated percentiles. */
class LatencyStats(private val capacity: Int = 300) {
    private val samples = DoubleArray(capacity)
    private val scratch = DoubleArray(capacity)
    private var count = 0
    private var next = 0

    @Synchronized
    fun add(ms: Double) {
        samples[next] = ms
        next = (next + 1) % capacity
        if (count < capacity) count++
    }

    @Synchronized
    fun reset() {
        count = 0
        next = 0
    }

    @Synchronized
    fun summary(): LatencySummary {
        if (count == 0) return LatencySummary()
        System.arraycopy(samples, 0, scratch, 0, count)
        scratch.sort(0, count)
        var sum = 0.0
        for (i in 0 until count) sum += scratch[i]
        return LatencySummary(
            p50 = percentile(scratch, count, 50.0),
            p95 = percentile(scratch, count, 95.0),
            p99 = percentile(scratch, count, 99.0),
            mean = sum / count,
            count = count,
        )
    }

    companion object {
        fun percentile(sorted: DoubleArray, n: Int, p: Double): Double {
            if (n == 1) return sorted[0]
            val pos = p / 100.0 * (n - 1)
            val lo = floor(pos).toInt()
            val hi = minOf(lo + 1, n - 1)
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
        }

        fun summarize(values: List<Double>): LatencySummary {
            val stats = LatencyStats(maxOf(values.size, 1))
            values.forEach(stats::add)
            return stats.summary()
        }
    }
}

/** Frames per second over the trailing [windowNs]. */
class FpsMeter(private val windowNs: Long = 1_000_000_000L, capacity: Int = 256) {
    private val stamps = LongArray(capacity)
    private var count = 0
    private var next = 0

    @Synchronized
    fun tick(nowNs: Long) {
        stamps[next] = nowNs
        next = (next + 1) % stamps.size
        if (count < stamps.size) count++
    }

    @Synchronized
    fun fps(nowNs: Long): Double {
        var frames = 0
        var oldest = nowNs
        for (i in 0 until count) {
            val t = stamps[i]
            if (nowNs - t <= windowNs) {
                frames++
                if (t < oldest) oldest = t
            }
        }
        if (frames < 2) return frames.toDouble()
        return (frames - 1) * 1e9 / (nowNs - oldest).coerceAtLeast(1)
    }

    @Synchronized
    fun reset() {
        count = 0
        next = 0
    }
}
