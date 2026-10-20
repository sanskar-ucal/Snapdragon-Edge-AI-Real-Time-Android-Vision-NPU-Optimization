#!/usr/bin/env bash
# Installs the Android NDK/CMake/platform and the Python environment used by ml/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
CMDLINE_VERSION="11076708"
NDK_VERSION="27.2.12479018"
CMAKE_VERSION="3.22.1"
PLATFORM="android-35"
BUILD_TOOLS="35.0.0"

case "$(uname -s)" in
  Darwin) OS_TAG="mac" ;;
  Linux) OS_TAG="linux" ;;
  *) echo "Unsupported OS" >&2; exit 1 ;;
esac

mkdir -p "$SDK"
if [ ! -x "$SDK/cmdline-tools/latest/bin/sdkmanager" ]; then
  echo "==> Installing Android cmdline-tools"
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/cmdline.zip" \
    "https://dl.google.com/android/repository/commandlinetools-${OS_TAG}-${CMDLINE_VERSION}_latest.zip"
  unzip -q "$tmp/cmdline.zip" -d "$tmp"
  mkdir -p "$SDK/cmdline-tools"
  rm -rf "$SDK/cmdline-tools/latest"
  mv "$tmp/cmdline-tools" "$SDK/cmdline-tools/latest"
  rm -rf "$tmp"
fi

SDKMANAGER="$SDK/cmdline-tools/latest/bin/sdkmanager"
echo "==> Installing NDK, CMake, platform and build-tools"
yes | "$SDKMANAGER" --sdk_root="$SDK" --licenses >/dev/null || true
"$SDKMANAGER" --sdk_root="$SDK" \
  "ndk;${NDK_VERSION}" "cmake;${CMAKE_VERSION}" "platforms;${PLATFORM}" \
  "build-tools;${BUILD_TOOLS}" "platform-tools"

echo "sdk.dir=$SDK" > "$ROOT/android/local.properties"

if [ "${SKIP_PYTHON:-0}" != "1" ]; then
  echo "==> Creating Python venv at ml/.venv"
  python3 -m venv "$ROOT/ml/.venv"
  "$ROOT/ml/.venv/bin/pip" install --upgrade pip -q
  "$ROOT/ml/.venv/bin/pip" install -q -r "$ROOT/ml/requirements.txt"
fi

echo "==> Toolchain ready"
