package com.edgevision.core.inference.adaptive

import com.edgevision.core.inference.ModelVariant

data class AdaptiveInput(
    val nowMs: Long,
    val p95Ms: Double,
    val samples: Int,
    val thermalStatus: Int,
    val headroom: Float?,
)

data class AdaptiveDecision(val from: ModelVariant, val to: ModelVariant, val reason: String)

/**
 * Walks a quality ladder (index 0 = best quality) in response to latency and thermal pressure.
 *
 * Steps down when p95 exceeds the frame budget or the device is thermally stressed. Steps up only
 * after a longer cooldown with clear latency and thermal headroom, which gives hysteresis so the
 * controller does not oscillate between two rungs.
 */
class AdaptiveController(
    private val ladder: List<ModelVariant>,
    private val frameBudgetMs: Double = 33.3,
    private val downCooldownMs: Long = 3_000,
    private val upCooldownMs: Long = 15_000,
    private val upgradeLatencyRatio: Double = 0.6,
    private val minSamples: Int = 30,
) {
    init {
        require(ladder.isNotEmpty())
    }

    var index = 0
        private set
    val current: ModelVariant get() = ladder[index]
    private var lastSwitchMs: Long? = null

    fun evaluate(input: AdaptiveInput): AdaptiveDecision? {
        if (input.samples < minSamples) return null
        val since = lastSwitchMs?.let { input.nowMs - it } ?: Long.MAX_VALUE

        val hot = input.thermalStatus >= THERMAL_STATUS_MODERATE || (input.headroom ?: 0f) >= 0.9f
        val slow = input.p95Ms > frameBudgetMs
        if ((hot || slow) && index < ladder.lastIndex && since >= downCooldownMs) {
            val reason = if (hot) {
                "thermal status ${input.thermalStatus}, headroom ${input.headroom ?: "n/a"}"
            } else {
                "p95 %.1f ms > %.1f ms budget".format(input.p95Ms, frameBudgetMs)
            }
            return move(index + 1, input.nowMs, reason)
        }

        val cool = input.thermalStatus < THERMAL_STATUS_MODERATE && (input.headroom ?: 0f) < 0.7f
        val fast = input.p95Ms < frameBudgetMs * upgradeLatencyRatio
        if (cool && fast && index > 0 && since >= upCooldownMs) {
            return move(index - 1, input.nowMs, "p95 %.1f ms leaves headroom, device cool".format(input.p95Ms))
        }
        return null
    }

    private fun move(to: Int, nowMs: Long, reason: String): AdaptiveDecision {
        val decision = AdaptiveDecision(ladder[index], ladder[to], reason)
        index = to
        lastSwitchMs = nowMs
        return decision
    }

    private companion object {
        /** android.os.PowerManager.THERMAL_STATUS_MODERATE */
        const val THERMAL_STATUS_MODERATE = 2
    }
}
