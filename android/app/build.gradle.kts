plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.edgevision"

    defaultConfig {
        applicationId = "com.edgevision"
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    buildFeatures { compose = true }
    androidResources { noCompress += "onnx" }
    packaging {
        // QNN/HTP libraries are loaded by path at runtime and must stay extractable.
        jniLibs.useLegacyPackaging = true
    }
}

dependencies {
    implementation(project(":feature:camera"))
    implementation(project(":feature:dashboard"))
    implementation(project(":feature:benchmark"))

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    implementation(libs.activity.compose)
    implementation(libs.lifecycle.runtime.compose)

    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.ext)
    androidTestImplementation(libs.junit)
}
