pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "EdgeVision"

include(
    ":app",
    ":core:native",
    ":core:inference",
    ":core:telemetry",
    ":feature:camera",
    ":feature:dashboard",
    ":feature:benchmark",
)
