#!/usr/bin/env python3
"""Generate MetalSource.swift from individual .metal kernel files."""

import os
import glob
import textwrap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.join(SCRIPT_DIR, "Sources", "SkMetalBridge", "Kernels")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "Sources", "SkMetalBridge", "MetalSource.swift")

HEADER_LINES = 2

def main():
    metal_files = sorted(glob.glob(os.path.join(KERNELS_DIR, "*.metal")))
    if not metal_files:
        raise RuntimeError(f"No .metal files found in {KERNELS_DIR}")

    lines = ["// Auto-generated from .metal files. Do not edit.",
             "// Run: python3 generate_metal_source.py",
             "",
             "import Foundation",
             "",
             "enum MetalSource {",
             "    static let all: String = \"\"\"",
             "    #include <metal_stdlib>",
             "    using namespace metal;"]

    for mf in metal_files:
        basename = os.path.basename(mf)
        with open(mf) as f:
            content = f.read()
        parts = content.split("\n", HEADER_LINES)
        body = parts[HEADER_LINES] if len(parts) > HEADER_LINES else ""
        if body.startswith("\n"):
            body = body[1:]
        lines.append(f"    // --- {basename} ---")
        if body:
            lines.append(textwrap.indent(body, "    "))

    lines.append('    """')
    lines.append("}")

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {OUTPUT_FILE} ({len(metal_files)} source files)")


if __name__ == "__main__":
    main()
