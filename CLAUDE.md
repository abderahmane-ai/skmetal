# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Build (required before testing if Swift/Metal changed)
```sh
bash build.sh          # compiles .metal → .metallib, builds Swift package, installs dylib to ~/.local/lib/
```
The `.metallib` is pre-compiled by `compile_metal.sh` — SPM only copies it, does NOT compile `.metal` files.
After `rm -rf .build`, you **must** re-run `compile_metal.sh` or the metallib won't have new kernels.

### Lint
```sh
cd skmetal && ruff check skmetal/ tests/
```

### Test
```sh
cd skmetal && python3 -m pytest ../tests/ -q --tb=short
```
Tests run from `skmetal/` dir (tests are at `../tests/` relative to it). `conftest.py` auto-resets config before each test.

### Pre-commit checklist
1. `ruff check skmetal/ tests/` — fix all violations
2. `find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null; find . -name "*.pyc" -delete`
3. Run full test suite
4. Ensure dylib is up-to-date before testing (see build above)

### Version bump (new release)
1. Update `skmetal/skmetal/_about.py` and `skmetal/pyproject.toml`
2. Build wheel: `cd skmetal && python3 -m build --wheel`
3. Commit, tag, push — CI/CD auto-pushes to PyPI on tag. **Do not use twine.**

## Architecture

skmetal is a 4-layer stack that executes scikit-learn estimators on Apple Silicon GPUs:

```
Python API (@accelerate) → ctypes bridge → Swift @_cdecl → Metal GPU kernels
```

### Core concept: zero-copy dispatch
numpy arrays pass directly to Metal via `bytesNoCopy` (unified memory) — no data copies. The `_bridge_call()` helper in `_bridge.py` auto-converts numpy arrays to `.ctypes.data` raw pointers before calling Swift `@_cdecl` functions.

### Dispatch flow
1. `@accelerate` decorator wraps any estimator-returning function
2. `_dispatch.py` looks up the sklearn class in `GPU_REGISTRY` (type-based, NOT name-based)
3. Returns a GPU wrapper (e.g. `MetalLinearRegression`) that wraps the original sklearn estimator
4. Each GPU wrapper's `fit()`/`predict()` calls `_should_use_gpu()` which checks: device setting, sparsity, float32 dtype, and configurable size thresholds
5. If GPU is unsuitable, transparently falls back to the wrapped sklearn estimator via `_fallback_fit()`/`_fallback_predict()`
6. Pipeline wrapping uses pure type-based dispatch — no substring matching on step names

### Estimator registry (`_registry.py`)
Single source of truth mapping sklearn classes → `(module_path, GPUClassName)`. Adding an estimator requires: (1) create GPU impl, (2) add to `GPU_REGISTRY`, (3) done — `_dispatch.py` consumes it automatically.

When MLX is available, `TruncatedSVD` is replaced with `MetalTruncatedSVDMLX` (GPU SVD via `mx.linalg.svd`). Other iterative estimators (Lasso, ElasticNet, LogisticRegression, KMeans) stay on the Metal bridge because `mx.compile` is unsuitable for iterative control flow.

### Threshold system (`_config.py`)
Two-tier: a global `threshold` (min `n*d` for GPU) and per-estimator `(min_rows, min_cols)` thresholds tuned from benchmarks. Thread-safe via `threading.Lock`. Per-thread device override via `accelerate_context` class using `threading.local()`.

### Swift bridge structure
Bridge files are organized by domain:
- `Bridge.swift` — device init, warmup, reduction ops
- `LinearModelBridge.swift` — ridge, FISTA, IRLS/L-BFGS for logistic regression
- `KMeansBridge.swift` — single fused command buffer (all iterations on GPU)
- `KNNBridge.swift` — tile-based top-k selection + voting
- `LinearAlgebraBridge.swift` — GEMM via MPS
- `PreprocessingBridge.swift` — StandardScaler, MinMaxScaler
- `MinMaxBridge.swift`, `LogisticBridge.swift`, `SVCBridge.swift`, `SVTreeBridge.swift`

### Metal kernels (15 files in `Sources/SkMetalBridge/Kernels/`)
Key kernel families: `KMeansKernels.metal` (assign/accumulate/combine), `KNNKernels.metal` (tile top-k, merge, vote), `IrlsKernels.metal` (weight, scale, hessians, L-BFGS), `ReductionKernels.metal` (sum, Welford mean/var), `CenterColumns.metal`, `ElementWiseKernels.metal`, `SIMDGroupGEMM.metal`, and others.

### MLX integration (optional)
- `_mlx_registry.py` — detects MLX availability, version, and capabilities
- `_mlx_svd.py` — `MetalTruncatedSVDMLX` uses GPU SVD via random projection + `mx.linalg.svd`
- MLX ops use `mx.array` directly (no ctypes/dylib) — separate path from the Metal bridge

### Key files
| File | Role |
|------|------|
| `skmetal/skmetal/accelerate.py` | @accelerate decorator + accelerate_context |
| `skmetal/skmetal/_dispatch.py` | Type-based estimator wrapping |
| `skmetal/skmetal/_bridge.py` | ctypes → Swift bridge, dylib discovery |
| `skmetal/skmetal/_config.py` | Config, thresholds, thread-local device |
| `skmetal/skmetal/estimators/_registry.py` | GPU_REGISTRY (single source of truth) |
| `skmetal/skmetal/estimators/_base.py` | BaseGPUEstimator with fallback logic |

## Known quirks
- `pytest-timeout` may not be installed; ignore `PytestConfigWarning: Unknown config option: timeout`
- `ruff` config lives in `skmetal/pyproject.toml` — run from `skmetal/` dir
- Resource bundle (`Bundle.module`) in Swift embeds build-time paths; after clean build, paths match the local machine automatically
- No float64 on Apple GPU — skmetal uses float32 exclusively, this is correct
- Threadgroup memory is only 32 KB on Apple GPU — design smaller tile sizes than CUDA

## GPU acceleration research
`GPU_ACCELERATION_RESEARCH.md` at project root contains the definitive reference for GPU/Metal/MLX implementation decisions. Read it before starting any GPU work.

## graphify
This project has a knowledge graph at `graphify-out/`. For codebase questions, use `graphify query`, `graphify path`, or `graphify explain` before raw file search. Use `graphify-out/wiki/index.md` for broad navigation. Run `graphify update .` after code changes to keep the graph current.
