plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.edgevision.core.telemetry"
}

dependencies {
    api(libs.kotlinx.coroutines.android)
    testImplementation(libs.junit)
}
