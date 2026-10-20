#!/usr/bin/env bash
# Runs the on-device benchmark via instrumentation, pulls the JSON report and regenerates RESULTS.md.
#   ITERATIONS=200 SUSTAINED_SECONDS=600 ADAPTIVE=true scripts/run_device_bench.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ADB="${ADB:-${ANDROID_HOME:-$HOME/Library/Android/sdk}/platform-tools/adb}"
ITERATIONS="${ITERATIONS:-200}"
WARMUP="${WARMUP:-20}"
SUSTAINED_SECONDS="${SUSTAINED_SECONDS:-600}"
ADAPTIVE="${ADAPTIVE:-true}"
PKG="com.edgevision"

"$ADB" get-state >/dev/null
echo "Device: $("$ADB" shell getprop ro.product.model) / SoC $("$ADB" shell getprop ro.soc.model)"

cd "$ROOT/android"
./gradlew -q :app:installDebug :app:installDebugAndroidTest

"$ADB" shell rm -rf "/sdcard/Android/data/$PKG/files/benchmarks"
"$ADB" shell am instrument -w \
  -e class com.edgevision.BenchmarkInstrumentedTest \
  -e iterations "$ITERATIONS" -e warmup "$WARMUP" \
  -e sustainedSeconds "$SUSTAINED_SECONDS" -e adaptive "$ADAPTIVE" \
  "$PKG.test/androidx.test.runner.AndroidJUnitRunner" | tee /tmp/edgevision_instrument.log

grep -q "OK (1 test)" /tmp/edgevision_instrument.log || { echo "instrumentation failed" >&2; exit 1; }

mkdir -p "$ROOT/benchmarks/device"
"$ADB" pull "/sdcard/Android/data/$PKG/files/benchmarks/." "$ROOT/benchmarks/device/"
"${PYTHON:-$ROOT/ml/.venv/bin/python}" "$ROOT/ml/report.py"
