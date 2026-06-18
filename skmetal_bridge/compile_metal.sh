#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNELS_DIR="$SCRIPT_DIR/Sources/SkMetalBridge/Kernels"
OUTPUT="$KERNELS_DIR/SkMetalBridge.metallib"

echo "==> Compiling Metal kernels from $KERNELS_DIR"

AIR_FILES=()
for f in "$KERNELS_DIR"/*.metal; do
    AIR_FILE="${f%.metal}.air"
    xcrun metal -c "$f" -o "$AIR_FILE"
    AIR_FILES+=("$AIR_FILE")
done

xcrun metallib -o "$OUTPUT" "${AIR_FILES[@]}"

for air in "${AIR_FILES[@]}"; do
    rm "$air"
done

echo "==> Created $OUTPUT ($(stat -f%z "$OUTPUT") bytes)"
