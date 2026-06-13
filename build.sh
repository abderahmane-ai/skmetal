#!/usr/bin/env bash
set -euo pipefail

# Build the Swift Metal bridge library and install it where Python can find it.

cd "$(dirname "$0")"/skmetal/skmetal_bridge

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

# Install to ~/.local/lib/ so the installed package can find it
INSTALL_DIR="$HOME/.local/lib"
mkdir -p "$INSTALL_DIR"
cp "$DYLIB" "$INSTALL_DIR/libSkMetalBridge.dylib"
echo "==> Installed dylib to $INSTALL_DIR/libSkMetalBridge.dylib"

echo "==> Done. You can now 'pip install -e .' or 'import skmetal'."
