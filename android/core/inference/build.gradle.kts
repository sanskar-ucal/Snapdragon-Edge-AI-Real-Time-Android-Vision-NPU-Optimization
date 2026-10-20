plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

val generatedModelAssets = layout.buildDirectory.dir("generated/modelAssets")

val copyModels by tasks.registering(Copy::class) {
    description = "Copies ONNX variants built by ml/ into the APK assets."
    from(rootProject.file("../ml/artifacts")) {
        include("yolo11n_*.onnx")
        exclude("*_prep.onnx")
    }
    into(generatedModelAssets.map { it.dir("models") })
}

android {
    namespace = "com.edgevision.core.inference"
    sourceSets["main"].assets.srcDir(generatedModelAssets)
    buildFeatures { buildConfig = true }
    defaultConfig {
        buildConfigField("String", "ORT_VERSION", "\"${libs.versions.onnxruntime.get()}\"")
    }
    androidResources { noCompress += "onnx" }
}

tasks.named("preBuild") { dependsOn(copyModels) }

dependencies {
    api(project(":core:native"))
    api(libs.onnxruntime.qnn)
    implementation(libs.kotlinx.coroutines.android)
    testImplementation(libs.junit)
}
