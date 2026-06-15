#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/skmetal_bridge"
OUTPUT_DIR="$SCRIPT_DIR/skmetal"

echo "=== Building SkMetalBridge ==="
cd "$BUILD_DIR"

# Compile Metal kernels into .metallib
bash compile_metal.sh

# Build
swift build --configuration release

# Copy dylib
mkdir -p "$OUTPUT_DIR"
cp ".build/release/libSkMetalBridge.dylib" "$OUTPUT_DIR/libSkMetalBridge.dylib"

echo ""
echo "=== Done ==="
echo "dylib: $OUTPUT_DIR/libSkMetalBridge.dylib"
echo ""
