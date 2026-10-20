package com.edgevision.core.inference

/** What the user asked for. */
enum class RequestedBackend { CPU, NPU }

/** What ONNX Runtime actually delivered. */
enum class ActiveBackend { CPU, QNN_HTP, QNN_HTP_PARTIAL }

/**
 * [npuVerified] is true only when the session was created with CPU fallback disabled, i.e.
 * ONNX Runtime assigned every graph node to the QNN HTP backend. Anything else is reported
 * as unverified or CPU so the app never overstates NPU execution.
 */
data class BackendInfo(
    val requested: RequestedBackend,
    val active: ActiveBackend,
    val npuVerified: Boolean,
    val detail: String,
) {
    val label: String
        get() = when (active) {
            ActiveBackend.QNN_HTP -> if (npuVerified) "NPU (QNN HTP) - verified" else "QNN HTP - unverified"
            ActiveBackend.QNN_HTP_PARTIAL -> "QNN HTP + CPU fallback - unverified"
            ActiveBackend.CPU -> if (requested == RequestedBackend.NPU) "CPU fallback" else "CPU"
        }
}
