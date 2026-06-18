#!/usr/bin/env bash
set -euo pipefail

# Build the Swift Metal bridge library and install it where Python can find it.

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT/skmetal_bridge"

echo "==> Compiling Metal kernels..."
"$REPO_ROOT/skmetal_bridge/compile_metal.sh"

echo "==> Building SkMetalBridge (release)..."
swift build --configuration release

# Determine the dylib path (architecture-dependent on Apple Silicon)
DYLIB=".build/arm64-apple-macosx/release/libSkMetalBridge.dylib"
if [ ! -f "$DYLIB" ]; then
    DYLIB=".build/release/libSkMetalBridge.dylib"
fi

if [ ! -f "$DYLIB" ]; then
    echo "ERROR: dylib not found at expected paths after build." >&2
    exit 1
fi

# Copy dylib into the skmetal Python package (zero-copy: sits next to _bridge.py)
cp "$DYLIB" "$REPO_ROOT/skmetal/libSkMetalBridge.dylib"

# Copy the compiled Metal bundle
BUNDLE=".build/arm64-apple-macosx/release/SkMetalBridge_SkMetalBridge.bundle"
if [ -d "$BUNDLE" ]; then
    cp -R "$BUNDLE" "$REPO_ROOT/skmetal/SkMetalBridge_SkMetalBridge.bundle"
fi

# Also install to ~/.local/lib/ for pip-installed wheels
INSTALL_DIR="$HOME/.local/lib"
mkdir -p "$INSTALL_DIR"
cp "$DYLIB" "$INSTALL_DIR/libSkMetalBridge.dylib"

echo "==> Installed dylib to skmetal/libSkMetalBridge.dylib and $INSTALL_DIR/libSkMetalBridge.dylib"
echo "==> Done. You can now run: pip install -e . && cd skmetal && python3 -m pytest ../tests/"
