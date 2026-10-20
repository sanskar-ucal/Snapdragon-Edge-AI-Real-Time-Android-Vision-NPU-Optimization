package com.edgevision.core.inference

enum class Precision { FP32, FP16, INT8 }

data class ModelVariant(val resolution: Int, val precision: Precision) {
    val id: String get() = "yolo11n_${resolution}_${precision.name.lowercase()}"
    val assetPath: String get() = "models/$id.onnx"

    /** Detect-head anchors for strides 8/16/32. */
    val numAnchors: Int get() = listOf(8, 16, 32).sumOf { (resolution / it) * (resolution / it) }

    override fun toString(): String = "${resolution}px ${precision.name}"
}

object ModelRegistry {
    val resolutions = listOf(320, 640)
    val all: List<ModelVariant> = resolutions.flatMap { r -> Precision.entries.map { ModelVariant(r, it) } }

    /**
     * Ordered fastest-quality-first degradation ladder used by the adaptive controller.
     * The CPU EP has no native FP16 kernels (FP16 is slower than FP32 there), so the CPU
     * ladder starts from FP32 while the HTP ladder starts from FP16.
     */
    fun ladder(backend: ActiveBackend): List<ModelVariant> {
        val top = if (backend == ActiveBackend.CPU) Precision.FP32 else Precision.FP16
        return listOf(ModelVariant(640, top), ModelVariant(640, Precision.INT8), ModelVariant(320, Precision.INT8))
    }

    fun parse(id: String): ModelVariant? = all.firstOrNull { it.id == id }
}
