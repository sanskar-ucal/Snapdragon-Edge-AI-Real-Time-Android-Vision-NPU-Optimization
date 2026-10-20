package com.edgevision.core.telemetry

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager

data class MemoryStats(val javaHeapMb: Double = 0.0, val nativeHeapMb: Double = 0.0, val pssMb: Double = 0.0)

data class ThermalStats(
    /** PowerManager.THERMAL_STATUS_* (0 = none ... 6 = shutdown). */
    val status: Int = 0,
    /** Forecast headroom; 1.0 means the device is at its throttling threshold. Null if unsupported. */
    val headroom: Float? = null,
    val batteryTempC: Float? = null,
) {
    val statusName: String get() = STATUS_NAMES.getOrElse(status) { "UNKNOWN" }

    companion object {
        const val MODERATE = 2
        private val STATUS_NAMES = listOf("NONE", "LIGHT", "MODERATE", "SEVERE", "CRITICAL", "EMERGENCY", "SHUTDOWN")
    }
}

interface DeviceSource {
    fun memory(): MemoryStats
    fun thermal(): ThermalStats
}

class DeviceMonitor(private val context: Context) : DeviceSource {
    private val power = context.getSystemService(PowerManager::class.java)
    private val memInfo = Debug.MemoryInfo()

    override fun memory(): MemoryStats {
        val rt = Runtime.getRuntime()
        Debug.getMemoryInfo(memInfo)
        return MemoryStats(
            javaHeapMb = (rt.totalMemory() - rt.freeMemory()) / MB,
            nativeHeapMb = Debug.getNativeHeapAllocatedSize() / MB,
            pssMb = memInfo.totalPss / 1024.0,
        )
    }

    override fun thermal(): ThermalStats {
        val headroom = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            power.getThermalHeadroom(HEADROOM_FORECAST_S).takeIf { !it.isNaN() }
        } else {
            null
        }
        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val temp = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            ?.takeIf { it != Int.MIN_VALUE }?.div(10f)
        return ThermalStats(power.currentThermalStatus, headroom, temp)
    }

    private companion object {
        const val MB = 1024.0 * 1024.0
        const val HEADROOM_FORECAST_S = 10
    }
}
