#!/bin/bash
# Build SignBridge.app: the SwiftUI app plus a bundled Python engine, signed and installed to ~/Applications.
#
#   macos/build.sh            build and install
#   macos/build.sh --no-install
#
# Signs with your first "Apple Development" identity so macOS keeps Camera and Accessibility permissions
# across rebuilds (an ad-hoc signature changes every build). Override: SIGN_IDENTITY="..." or SIGN_IDENTITY=-
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
APP=build/SignBridge.app
RES=$APP/Contents/Resources

echo "== Swift app"
(cd macos && swift build -c release)

echo "== Bundle layout"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$RES/engine" "$RES/models"
cp macos/.build/release/SignBridge "$APP/Contents/MacOS/SignBridge"
cp macos/Info.plist "$APP/Contents/Info.plist"

echo "== Python runtime"
PYHOME=$("$(uv python find 3.12)" -c "import sys; print(sys.base_prefix)")  # the real install, not uv's alias
cp -R "$PYHOME" "$RES/python"
rm -rf "$RES/python/lib/python3.12/site-packages"/*
rm -rf "$RES/python/lib/python3.12/"{test,idlelib,tkinter,turtledemo,ensurepip} "$RES/python/lib/"{tcl*,tk*,itcl*} 2>/dev/null || true
find "$RES/python" -name "EXTERNALLY-MANAGED" -delete

echo "== Engine dependencies"
uv pip install -q --python "$RES/python/bin/python3" --target "$RES/site-packages" -r requirements-app.txt
mkdir -p "$RES/engine/signbridge"
cp signbridge/*.py "$RES/engine/signbridge/"
cp models/holistic_landmarker.task models/popsign.npz "$RES/models/"
rm -rf "$RES/site-packages/PyObjCTest"
find "$APP" -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "== Smoke test: engine imports with the bundled Python"
SIGNBRIDGE_RESOURCES="$ROOT/$RES" PYTHONPATH="$RES/engine:$RES/site-packages" "$RES/python/bin/python3" -c \
  "import signbridge.runtime, signbridge.server; print('engine imports ok')"

echo "== Sign"
IDENTITY=${SIGN_IDENTITY:-$(security find-identity -p codesigning -v | grep -m1 '"Apple Development' | awk '{print $2}')}
IDENTITY=${IDENTITY:--}
# Sign nested code first (inside out), then the app.
find "$RES" -type f \( -name "*.so" -o -name "*.dylib" -o -perm -u+x \) -print0 | while IFS= read -r -d '' f; do
  if file "$f" | grep -q "Mach-O"; then codesign --force --sign "$IDENTITY" --timestamp=none "$f" 2>/dev/null || echo "  could not sign $f"; fi
done
codesign --force --sign "$IDENTITY" --timestamp=none "$APP"
codesign --verify --deep --strict "$APP" && echo "signature ok ($IDENTITY)"
du -sh "$APP"

if [[ "${1:-}" != "--no-install" ]]; then
  echo "== Install"
  mkdir -p ~/Applications
  pkill -x SignBridge 2>/dev/null || true
  rm -rf ~/Applications/SignBridge.app
  cp -R "$APP" ~/Applications/
  echo "Installed ~/Applications/SignBridge.app. Open it with: open ~/Applications/SignBridge.app"
fi
