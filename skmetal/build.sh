#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/skmetal_bridge"
OUTPUT_DIR="$SCRIPT_DIR/skmetal"

echo "=== Building SkMetalBridge ==="
cd "$BUILD_DIR"

# Regenerate MetalSource.swift from .metal files
echo "Generating MetalSource.swift..."
python3 -c "
import pathlib
metal_dir = 'Sources/SkMetalBridge/Kernels'
metal_files = sorted(pathlib.Path(metal_dir).glob('*.metal'))
combined = '\n'.join(f.read_text() for f in metal_files)
with open('Sources/SkMetalBridge/MetalSource.swift', 'w') as out:
    out.write('// Auto-generated from .metal files. Do not edit.\n')
    out.write('// Run: python3 generate_metal_source.py\n\n')
    out.write('import Foundation\n\n')
    out.write('enum MetalSource {\n')
    out.write('    static let all: String = \"\"\"\n')
    out.write(combined)
    out.write('\"\"\"\n')
    out.write('}\n')
print(f'  Generated Sources/SkMetalBridge/MetalSource.swift ({len(combined)} chars)')
"

# Build
swift build --configuration release

# Copy dylib
mkdir -p "$OUTPUT_DIR"
cp ".build/release/libSkMetalBridge.dylib" "$OUTPUT_DIR/libSkMetalBridge.dylib"

echo ""
echo "=== Done ==="
echo "dylib: $OUTPUT_DIR/libSkMetalBridge.dylib"
echo ""
