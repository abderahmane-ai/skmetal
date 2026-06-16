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

Measured on an M4 Max with 128 GB unified memory, macOS 15.5, Python 3.12:

| Estimator | Data Size | CPU (s) | GPU (s) | Speedup |
|-----------|-----------|---------|---------|---------|
| `StandardScaler` | 1,000,000 × 100 | 0.292 | 0.030 | **9.6×** |
| `LinearRegression` | 200,000 × 500 | 1.245 | 0.148 | **8.4×** |
| `TruncatedSVD` | 100,000 × 500 | 0.274 | 0.094 | **2.9×** |
| `MinMaxScaler` | 1,000,000 × 100 | 0.043 | 0.035 | **1.2×** |
| `LogisticRegression` | 100,000 × 200 | 0.028 | 0.031 | 0.9× |
| `Ridge` | 200,000 × 500 | 0.119 | 0.129 | 0.9× |
| `KMeans` | 500,000 × 100 | 3.299 | 24.325 | 0.1× |

**Winners:** Reduction-heavy ops (mean/variance, GEMM, SVD) see 3–10× speedup. **Break-even:** LogisticRegression and Ridge are dispatch-limited at these sizes. **CPU wins:** KMeans is slower on GPU (fused command-buffer loop) — MLX flash-kmeans integration (94–517× proven) is planned for v0.9.0.

Run benchmarks locally:
```bash
python benchmarks/run_compare.py     # moderate data sizes
python benchmarks/benchmark_suite.py # large data (generates baseline.json)
```

---

## Features

- **Zero-copy GPU execution** — numpy arrays passed directly to Metal via `bytesNoCopy` on unified memory
- **Drop-in acceleration** — decorate any estimator-returning function, wrap an existing instance, or use a context manager
- **Smart dispatch** — automatically routes to CPU for small datasets where GPU overhead dominates; configurable per-estimator thresholds
- **GPU solvers** — Cholesky, FISTA, L-BFGS, IRLS, K-Means fused iterations, KNN tile-then-merge, SIMD-group GEMM, and more
- **float4 vectorization** — 6 kernel families use float4 loads/stores for 4× memory throughput
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
| `KMeans` | Single fused command buffer (all iterations on GPU) |
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
