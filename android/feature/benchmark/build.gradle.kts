plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.edgevision.feature.benchmark"
    buildFeatures { compose = true }
}

dependencies {
    api(project(":core:inference"))
    api(project(":core:telemetry"))
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    implementation(libs.lifecycle.runtime.compose)
}
