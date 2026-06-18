# Contributing to skmetal

Thanks for contributing! skmetal brings GPU acceleration to scikit-learn on Apple Silicon.

## Development Setup

### Prerequisites
- macOS 14+ on Apple Silicon (M1–M5)
- Python 3.10+
- Swift 6.1+ and Xcode Command Line Tools (`xcrun metal` must work)

### Clone and install

```bash
git clone https://github.com/abderahmane-ai/skmetal.git
cd skmetal/skmetal_bridge

# Compile Metal shaders → .metallib
bash compile_metal.sh

# Build the Swift bridge dylib
swift build --configuration release

# Install dylib into the Python package
cp .build/arm64-apple-macosx/release/libSkMetalBridge.dylib ../skmetal/

cd ..
pip install -e ".[dev]"
```

## Architecture Overview

skmetal is a 4-layer stack:

```
Python API (@accelerate) → ctypes bridge → Swift @_cdecl → Metal GPU kernels
```

- **Layer 1 — `accelerate.py` / `_dispatch.py`**: Decorator + type-based registry dispatch
- **Layer 2 — `_bridge.py`**: ctypes bridge, auto-converts numpy arrays to `.ctypes.data`
- **Layer 3 — Swift bridge** (`skmetal_bridge/Sources/SkMetalBridge/`): `@_cdecl` functions, MPS, MTLBuffer
- **Layer 4 — Metal kernels** (`skmetal_bridge/Sources/SkMetalBridge/Kernels/`): `.metal` compute shaders

Apple Silicon's **unified memory** enables zero-copy: numpy arrays pass directly to Metal via `bytesNoCopy` — no data copies.

### Key design patterns
- **Zero-copy dispatch**: Bridge functions receive raw pointers to numpy data, wrap them in `MTLBuffer(bytesNoCopy:)`.
- **Threshold-based routing**: `_should_use_gpu()` gates dispatch — small matrices use CPU where GPU overhead dominates.
- **MLX dual backend**: KMeans and TruncatedSVD optionally use MLX Python API for GPU ops.
- **Transparent fallback**: On non-Apple-Silicon machines, every estimator falls back to CPU scikit-learn.

## Adding a New Estimator

### Step 1: Create the GPU class

Create a file in `skmetal/estimators/` extending `BaseGPUEstimator`:

```python
# skmetal/estimators/my_estimator.py
import numpy as np
from ._base import BaseGPUEstimator

class MetalMyEstimator(BaseGPUEstimator):
    """GPU-accelerated MyEstimator via ..."""

    def fit(self, X, y=None, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        # Allocate output arrays, call bridge function
        result = my_bridge_function(X, y)
        self._estimator.result_ = result
        self._fitted = True
        return self

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        return my_predict_function(X, self._estimator.result_)
```

The `_should_use_gpu()` gate checks device setting, dtype, sparsity, and thresholds automatically. Override it if you need additional checks (e.g., kernel-type constraints).

### Step 2: Register in `_registry.py`

```python
# skmetal/estimators/_registry.py
from sklearn.my_module import MyEstimator

GPU_REGISTRY[MyEstimator] = ("skmetal.estimators.my_estimator", "MetalMyEstimator")
```

### Step 3: Export in `estimators/__init__.py`

```python
from .my_estimator import MetalMyEstimator
__all__.append("MetalMyEstimator")
```

### Step 4: Add per-estimator thresholds in `_config.py`

```python
PER_ESTIMATOR_THRESHOLDS["MyEstimator"] = (1_000, 10)
```

The `_dispatch.py` system consumes `GPU_REGISTRY` automatically — no other changes needed.

## Swift Bridge Development

### Adding a Metal kernel

1. Create or edit a `.metal` file in `skmetal_bridge/Sources/SkMetalBridge/Kernels/`
2. Use `device float*` for buffers, `uint tid [[thread_position_in_grid]]` for thread indexing
3. Use `float4` vectorized loads/stores for 4× memory throughput where possible
4. Keep threadgroup memory under 32 KB (Apple GPU limit)

### Adding a bridge function

1. Add a `@_cdecl` function in the appropriate `*Bridge.swift` file
2. Use `wrapInput()` / `wrapOutput()` for zero-copy buffer creation
3. Return `Int32(0)` on success, nonzero on error

### Adding the Python wrapper

1. Add the C function signature to `_BRIDGE_REGISTRY` in `_bridge.py`
2. Write a Python wrapper function using `_bridge_call()` which auto-converts numpy arrays → `.ctypes.data`

### Rebuild flow

```bash
cd skmetal_bridge
bash compile_metal.sh              # .metal → .metallib
swift build --configuration release
cp .build/arm64-apple-macosx/release/libSkMetalBridge.dylib ../skmetal/
```

## Running Tests

Tests live in `tests/` (one level above `pyproject.toml`). Run from the `skmetal/` directory:

```bash
cd skmetal
python3 -m pytest ../tests/ -q --tb=short
```

### Test file overview

| File | What it covers |
|------|---------------|
| `test_fallback.py` | Platform-independent (works without dylib). Registry invariants, dispatch routing. |
| `test_correctness.py` | GPU vs CPU numerical equivalence for every registered estimator. |
| `test_accelerate.py` | Decorator, context manager, pipeline wrapping, thread safety. |
| `test_config.py` | Config API: set_device, set_threshold, update_threshold, reset_thresholds. |
| `test_mlx_*.py` | MLX backend integration tests. |
| `test_kernels.py` | Low-level Metal kernel tests. |
| `test_large_matrix.py` | Scale tests with large datasets. |
| `test_stress.py` | Edge-case inputs and stress tests. |

## Code Style

- **Lint**: `ruff check skmetal/ tests/` (config in `skmetal/pyproject.toml`)
- **Line length**: 120
- **Python target**: 3.10
- **Docstrings**: Required on all public API functions and classes
- **Type annotations**: Required on function signatures
- **Swift**: Follow standard Swift conventions; no force-unwraps (`!`) — use `guard let` instead

## Pull Request Checklist

1. [ ] `ruff check skmetal/ tests/` — clean
2. [ ] `cd skmetal && python3 -m pytest ../tests/ -q --tb=short` — all pass (263+ tests)
3. [ ] If Swift or Metal changed: rebuild dylib (`bash compile_metal.sh && swift build`)
4. [ ] If adding a new estimator: add per-estimator threshold in `_config.py`
5. [ ] If adding a new estimator: add correctness tests in `test_correctness.py`
6. [ ] `cd skmetal && python3 -m build --wheel` — builds cleanly
