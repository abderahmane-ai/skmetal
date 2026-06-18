# skmetal

**Apple Silicon GPU acceleration for scikit-learn**

[![PyPI](https://img.shields.io/pypi/v/skmetal?color=3776AB&style=flat-square)](https://pypi.org/project/skmetal/)
[![Python](https://img.shields.io/pypi/pyversions/skmetal?logo=python&style=flat-square)](https://pypi.org/project/skmetal/)
[![Platform](https://img.shields.io/badge/platform-macOS%2014%2B%20|%20Apple%20Silicon-000000?logo=apple&style=flat-square)](https://github.com/abderahmane-ai/skmetal)
[![CI](https://github.com/abderahmane-ai/skmetal/actions/workflows/ci.yml/badge.svg)](https://github.com/abderahmane-ai/skmetal/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/abderahmane-ai/skmetal)

Decorate any scikit-learn estimator with `@skmetal.accelerate` and `fit()`/`predict()` run on the GPU — no code changes. Leverages Apple Silicon's unified memory for zero-copy data sharing between numpy and Metal.

```python
import skmetal
from sklearn.linear_model import LinearRegression

@skmetal.accelerate
def model():
    return LinearRegression()

m = model()
m.fit(X_train, y_train)
m.predict(X_test)
```

---

## Installation

```bash
pip install skmetal
```

macOS 14+ and Apple Silicon (M1–M5) required. No Xcode needed — the pip package includes a pre-built dylib and Metal library.

### From source (for development)

```bash
git clone https://github.com/abderahmane-ai/skmetal.git
cd skmetal
pip install -e ".[dev]"

# Build Swift + Metal (required after any .metal or .swift change)
cd skmetal_bridge
bash compile_metal.sh
swift build --configuration release
cp .build/arm64-apple-macosx/release/libSkMetalBridge.dylib ../skmetal/
cd ..
```

---

## Benchmarks

Measured on an M4 Air (16 GB) and M4 Max (128 GB). All data float32, `n_init=1`, `max_iter=30` for KMeans.

| Estimator | Data Size | CPU | GPU | Speedup | Notes |
|-----------|-----------|-----|-----|---------|-------|
| `KMeans` (MLX) | 200,000 × 64, k=500 | 24.4s | 0.8s | **30×** | flash-kmeans-mlx `mx.compile`-d kernel |
| `KMeans` (MLX) | 100,000 × 128, k=200 | 10.0s | 0.6s | **16×** | 3-10× typical for n_init ≥ 3 |
| `StandardScaler` | 1,000,000 × 100 | 0.29s | 0.03s | **10×** | Fused Welford (1 dispatch) |
| `LinearRegression` | 200,000 × 500 | 1.25s | 0.15s | **8.4×** | MPS GEMM + Cholesky solve |
| `TruncatedSVD` | 100,000 × 500 | 0.27s | 0.09s | **2.9×** | Randomized SVD on GPU |
| `MinMaxScaler` | 1,000,000 × 100 | 0.04s | 0.04s | **1.2×** | Threadgroup tree reduction |
| `LogisticRegression` | 100,000 × 200 | 0.03s | 0.03s | 0.9× | Dispatch-limited at this size |
| `Ridge` | 200,000 × 500 | 0.12s | 0.13s | 0.9× | CPU Accelerate framework wins at all sizes |

KMeans MLX requires `pip install skmetal[mlx]`. Without MLX, KMeans falls back to a Metal fused command-buffer (slower than CPU — 0.1×). The MLX path uses [flash-kmeans-mlx](https://github.com/hanxiao/flash-kmeans-mlx) which fuses distance + argmin + update into a single compiled GPU kernel.

### Apple M3 Pro (18 GB)

| Estimator | Data Size | Speedup |
|-----------|-----------|---------|
| `LinearRegression` | 200,000 × 500 | **10.42×** |
| `StandardScaler` | 1,000,000 × 100 | **9.93×** |
| `TruncatedSVD` | 100,000 × 500 | **4.20×** |
| `MinMaxScaler` | 1,000,000 × 100 | **1.36×** |
| `KMeans` | 500,000 × 100 | **1.32×** |
| `Ridge` | 200,000 × 500 | 0.94× |
| `LogisticRegression` | 100,000 × 200 | 0.93× |

Run benchmarks locally:
```bash
pip install skmetal[mlx]
python benchmarks/run_compare.py     # moderate data sizes
python benchmarks/benchmark_suite.py # large data (generates baseline.json)
```

---

## Features

- **Zero-copy GPU execution** — numpy arrays passed directly to Metal via `bytesNoCopy` on unified memory
- **Drop-in acceleration** — decorate any estimator-returning function, wrap an existing instance, or use a context manager
- **Smart dispatch** — automatically routes to CPU for small datasets where GPU overhead dominates; configurable per-estimator thresholds
- **GPU solvers** — Cholesky, FISTA, L-BFGS, IRLS, KNN tile-then-merge, SIMD-group GEMM, flash-kmeans-mlx compiled kernels
- **float4 vectorization** — 6 kernel families use float4 loads/stores for 4× memory throughput
- **Optional MLX acceleration** — install `skmetal[mlx]` for flash-kmeans-mlx GPU KMeans (3–30×) and TruncatedSVD (GPU SVD)
- **Transparent fallback** — imports cleanly on non-Apple-Silicon machines; all operations fall back to scikit-learn CPU
- **Verbose logging** — `skmetal.set_verbose(True)` prints why each estimator chose GPU or CPU

---

## Usage

### Decorator (recommended)

Wrap any function that returns an estimator:

```python
import skmetal
from sklearn.linear_model import LogisticRegression

@skmetal.accelerate
def model():
    return LogisticRegression(random_state=42)

clf = model()
clf.fit(X_train, y_train)
clf.predict(X_test)
```

Works with pipelines too:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

@skmetal.accelerate
def pipe():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ])

p = pipe()
p.fit(X, y)
```

### Function call

```python
model = skmetal.accelerate(LinearRegression())
model.fit(X, y)
```

### Context manager

```python
with skmetal.accelerate_context():
    model = LinearRegression()
    model.fit(X, y)
```

### Checking GPU availability

```python
import skmetal

if skmetal.METAL_AVAILABLE:
    info = skmetal.device_info()
    print(info)
    # {'name': 'Apple M4 Max', ...,
    #  'has_unified_memory': True,
    #  'recommended_working_set_size_bytes': 68719476736}
```

### Configuration

```python
import skmetal

skmetal.set_device("cpu")             # force CPU fallback globally
skmetal.set_verbose(True)             # log dispatch decisions
skmetal.set_threshold(100_000)        # global min n*d for GPU
skmetal.update_threshold("KMeans",    # per-estimator override
                         min_rows=100_000, min_cols=50)
skmetal.reset_thresholds()            # restore defaults

config = skmetal.get_config()
print(config)
```

On non-Apple-Silicon machines `skmetal` imports cleanly and all estimators transparently fall back to scikit-learn CPU.

---

## Supported Estimators

| Estimator | GPU Strategy |
|-----------|-------------|
| `LinearRegression` | Normal equations via MPS GEMM + Cholesky solve |
| `Ridge` | Fused centering + XTX + XTy (1 dispatch) |
| `LogisticRegression` | L-BFGS on GPU (full loop in Swift, fused kernels) |
| `Lasso` | FISTA with GPU residual updates |
| `ElasticNet` | FISTA with GPU residual updates |
| `KMeans` | flash-kmeans-mlx GPU kernel (MLX) or fused command buffer (Metal) |
| `DBSCAN` | GPU pairwise distance + per-point neighbor counting |
| `KNeighborsClassifier` | GPU pairwise distance + fused voting (weighted/unweighted) |
| `KNeighborsRegressor` | GPU pairwise distance + fused averaging |
| `NearestNeighbors` | GPU pairwise distance + index |
| `TruncatedSVD` | Randomized SVD (random projection + GPU GEMM) |
| `SVC` | GPU RBF kernel + precomputed kernel predict |
| `SVR` | GPU RBF kernel + precomputed kernel predict |
| `GaussianNB` | GPU mean/var per class |
| `StandardScaler` | Fused Welford mean/variance (1 dispatch) |
| `MinMaxScaler` | Column min/max with threadgroup tree reduction |
| `RobustScaler` | GPU quantile approximation |
| `HistGradientBoostingRegressor` | C++ HGBT from sklearn (CPU) |
| `HistGradientBoostingClassifier` | C++ HGBT from sklearn (CPU) |

Each estimator has per-estimator (min_rows, min_cols) thresholds. Below the threshold the estimator uses CPU. Override via `skmetal.update_threshold()` or force GPU with `skmetal.set_device("gpu")`.

---

## Architecture

```
numpy array → np.ctypes.data → UnsafeMutableRawPointer → MTLBuffer(bytesNoCopy:) → Metal GPU
                   |                                                    |
                   +--------- same physical memory (unified) -----------+
```

Apple Silicon's unified memory enables zero-copy data sharing. The Swift bridge exposes `@_cdecl` functions callable from Python via `ctypes`. Each estimator's `fit()`/`predict()` calls the appropriate bridge function, which dispatches Metal Performance Shaders or custom compute kernels.

### Metal kernels (13 files)

| Kernel file | Operations |
|-------------|------------|
| `ReductionKernels.metal` | `reduce_sum`, `norm2`, `max_abs_diff` (float4 vectorized) |
| `CenterColumns.metal` | `column_means`, `center_columns`, `column_means_and_center` (fused) |
| `KMeansKernels.metal` | assign, accumulate, combine_normalize, inertia (float4 vectorized) |
| `KNNKernels.metal` | tile top-k, merge, negate distances, fused voting (float4 vectorized) |
| `IrlsKernels.metal` | compute_linear_irls, compute_error_scale (float4), l2_reg_irls, sigmoid, log_loss, multinomial_grad_l2 |
| `ElementWiseKernels.metal` | sigmoid, subtract, axpy, add_diagonal, softmax_residual, rbf_apply |
| `PairwiseDistKernels.metal` | `pairwise_from_cross`, `row_norm_sq`, `distance_correct` |
| `DistanceKernels.metal` | `row_norm_sq`, `compute_mindists`, `distance_correct` |
| `ExtraKernels.metal` | `soft_threshold`, `column_transform`, `scale_f32`, `sv_init`, `sv_hook`, `sv_shortcut` |
| `StandardScalerKernels.metal` | `scaler_fit` (fused Welford) |
| `MinMaxKernels.metal` | `column_minmax` (threadgroup tree reduction) |
| `TreeKernels.metal` | `tree_predict_all` |
| `SIMDGroupGEMM.metal` | `simdgroup_gemm_f32`, `simdgroup_gemm_f16` |

### Swift bridge (9 files)

| File | Domain |
|------|--------|
| `Bridge.swift` | Device init, warmup, reduction ops |
| `LinearModelBridge.swift` | Ridge, FISTA, L-BFGS for logistic regression |
| `KMeansBridge.swift` | Single fused command buffer (all iterations on GPU) |
| `KNNBridge.swift` | Tile-based top-k selection + voting |
| `LinearAlgebraBridge.swift` | GEMM via MPS, SIMD-group GEMM, pairwise distance |
| `PreprocessingBridge.swift` | StandardScaler, MinMaxScaler |
| `MinMaxBridge.swift` | MinMax transform |
| `LogisticBridge.swift` | IRLS/L-BFGS GPU loop |
| `SVCBridge.swift` | SVC/SVR RBF predict |
| `SVTreeBridge.swift` | Union-find, tree predict |

---

## Project Structure

```
skmetal/
  skmetal/
    __init__.py          # public API: accelerate, config, device_info
    _about.py            # version
    _bridge.py           # ctypes → Swift @_cdecl exports (47 functions)
    _config.py           # Config dataclass, thresholds, device control
    _dispatch.py         # estimator registry + wrapping logic
    accelerate.py        # @accelerate decorator + context manager
    estimators/
      _base.py           # BaseGPUEstimator with fallback logic
      _registry.py       # GPU_REGISTRY (19 estimators)
      _mlx_registry.py   # MLX detection
      _mlx_svd.py        # TruncatedSVD MLX backend
      linear_model.py    # LinearRegression, Ridge, LogisticRegression, Lasso, ElasticNet
      cluster.py         # KMeans, DBSCAN
      decomposition.py   # TruncatedSVD
      ensemble.py        # HistGradientBoosting
      naive_bayes.py     # GaussianNB
      neighbors.py       # KNeighbors, NearestNeighbors
      preprocessing.py   # StandardScaler, MinMaxScaler, RobustScaler
      svm.py             # SVC, SVR
  skmetal_bridge/        # Swift + Metal SPM package
    Sources/SkMetalBridge/
      Bridge.swift
      LinearModelBridge.swift
      KMeansBridge.swift
      KNNBridge.swift
      LinearAlgebraBridge.swift
      PreprocessingBridge.swift
      MinMaxBridge.swift
      LogisticBridge.swift
      SVCBridge.swift
      SVTreeBridge.swift
      MetalContext.swift
      Kernels/*.metal    # 13 Metal kernel files
  benchmarks/
    run_compare.py       # quick comparison benchmark
    benchmark_suite.py   # full benchmark suite (generates baseline)
    baseline.json
  tests/                 # 263 tests across 11 files
  pyproject.toml
  LICENSE
  .github/workflows/
    ci.yml               # build + ruff + pytest + benchmarks
    release.yml          # PyPI publish on v* tag
```

---

## Development

```bash
git clone https://github.com/abderahmane-ai/skmetal.git
cd skmetal

# Build Swift + Metal (required after any .metal or .swift change)
cd skmetal_bridge
bash compile_metal.sh
swift build --configuration release
cp .build/arm64-apple-macosx/release/libSkMetalBridge.dylib ../skmetal/
cd ..

# Install in editable mode
pip install -e ".[dev]"

# Lint
ruff check skmetal/ tests/
ruff format --check skmetal/ tests/

# Run tests (from skmetal/ dir)
cd skmetal && python3 -m pytest ../tests/ -q --tb=short

# Run benchmarks
python benchmarks/run_compare.py
```

### Adding a new estimator

1. Create `skmetal/estimators/my_model.py` with `MetalMyModel(BaseGPUEstimator)`
2. Implement `fit()`/`predict()` calling the Swift bridge via `_bridge_call()`
3. Register in `estimators/_registry.py` → `GPU_REGISTRY`
4. Add a parametrized test in `tests/test_correctness.py`
5. Write a Metal kernel + Swift bridge function if needed

---

## License

MIT
