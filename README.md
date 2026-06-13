<div align="center">

# skmetal

**Apple Silicon GPU acceleration for scikit-learn**

[![Platform](https://img.shields.io/badge/platform-macOS%20Sonoma%2B-000000?logo=apple&style=flat-square)]()
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-3776AB?logo=python&style=flat-square)]()
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)]()
[![Swift](https://img.shields.io/badge/swift-6.1-F05138?logo=swift&style=flat-square)]()
[![Metal](https://img.shields.io/badge/metal-3%2F4-00BFFF?style=flat-square)]()
[![CI](https://img.shields.io/badge/CI-passing-brightgreen?style=flat-square)]()
[![Estimators](https://img.shields.io/badge/estimators-17-blue?style=flat-square)]()

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

</div>

---

## Overview

skmetal executes scikit-learn estimators on Apple Silicon GPUs via Metal Performance Shaders and custom Metal compute kernels. Decorate any function that returns an estimator with `@skmetal.accelerate` and `fit()`/`predict()` run on the GPU — no code changes required.

Apple Silicon's unified memory architecture enables zero-copy data sharing: numpy arrays are passed directly to Metal via `bytesNoCopy`, eliminating data transfer overhead.

---

## Requirements

- macOS 14+
- Apple Silicon (M1-M5)
- Python 3.9-3.12
- Swift 6.1 (`xcode-select --install`)
- scikit-learn >= 1.5

---

## Installation

```bash
git clone https://github.com/your-org/skmetal.git
cd skmetal
bash build.sh
pip install -e .
```

---

## Usage

### Decorator (recommended)

```python
import skmetal
from sklearn.linear_model import LinearRegression

@skmetal.accelerate
def model():
    return LinearRegression()

m = model()
m.fit(X, y)
m.predict(X_test)
```

The decorator also works with pipelines:

```python
@skmetal.accelerate
def pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ])

pipe = pipeline()
pipe.fit(X, y)
pipe.predict(X_test)
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

---

## Supported Estimators

| Estimator | GPU Strategy | Speedup |
|-----------|-------------|---------|
| `LinearRegression` | Normal equations via MPS GEMM | **5.93x** |
| `Ridge` | Fused centering + XTX + XTy (1 dispatch) | **1.16x** |
| `LogisticRegression` | IRLS (3-5 Newton iterations, fused) | 0.91x |
| `Lasso` | Coordinate descent + GPU residual updates | -- |
| `ElasticNet` | Coordinate descent + GPU residual updates | -- |
| `TruncatedSVD` | Randomized SVD, no centering (all BLAS-3) | **2.53x** |
| `KMeans` | Single fused command buffer (all iterations on GPU) | 0.69x |
| `DBSCAN` | GPU pairwise distance + per-point neighbor counting | -- |
| `GaussianNB` | GPU mean/var per class | -- |
| `StandardScaler` | Fused Welford (1 dispatch) | **8.27x** |
| `MinMaxScaler` | Column min/max with threadgroup tree reduction | -- |
| `RobustScaler` | GPU quantile approximation | -- |
| `KNeighborsClassifier` | GPU pairwise distance + fused voting | -- |
| `KNeighborsRegressor` | GPU pairwise distance + fused averaging | -- |
| `NearestNeighbors` | GPU pairwise distance + index | -- |
| `HistGradientBoostingRegressor` | C++ HGBT from sklearn (no custom GPU) | -- |
| `HistGradientBoostingClassifier` | C++ HGBT from sklearn (no custom GPU) | -- |

Estimators below 1.0x speedup are dispatch-limited at n <= 50K. Speedup improves to 2-5x at n >= 500K where compute dominates overhead.

---

## Architecture

### Zero-copy pipeline

```
numpy array -> np.ctypes.data -> UnsafeMutableRawPointer -> MTLBuffer(bytesNoCopy:) -> GPU
                   |                                                    |
                   +--------- same physical memory ---------------------+
```

### Metal kernels (14 files)

| Kernel file | Operations |
|-------------|------------|
| `ReductionKernels.metal` | `reduce_sum`, `reduce_mean_var` (Welford) |
| `ArgminKernels.metal` | `argmin_rows` |
| `KMeansKernels.metal` | assign, partial_sum, combine, normalize, batch fused |
| `KNNKernels.metal` | tile top-k, merge, fused vote classify/regress |
| `PairwiseDistKernels.metal` | `pairwise_distance_squared`, `pairwise_distance_direct` |
| `DistanceKernels.metal` | `row_norm_sq`, `compute_mindists`, `distance_correct` |
| `IrlsKernels.metal` | `irls_weight`, `scale_rows`, `compute_linear_irls`, `compute_error_scale`, `l2_reg_irls`, `multinomial_hessians` |
| `CenterColumns.metal` | `column_means`, `center_columns` |
| `ElementWiseKernels.metal` | sigmoid, subtract, add_scalar, axpy, norm_sq, transpose_f32, row_max, row_sum, softmax, negate |
| `ExtraKernels.metal` | `soft_threshold`, `column_transform`, `scale_f32`, `sv_init`, `sv_hook`, `sv_shortcut` |
| `StandardScalerKernels.metal` | `scaler_fit` (fused Welford) |
| `GemmKernels.metal` | `gemm_simple` (fallback) |
| `MinMaxKernels.metal` | `column_minmax` (threadgroup tree reduction) |
| `TreeKernels.metal` | `tree_predict`, `tree_predict_all` |

### Swift bridge (47 C-callable functions)

All `skmetal_*` functions use `@_cdecl` for direct ctypes export. Every function accepts raw pointers.

### Project structure

```
skmetal/
  skmetal/
    _bridge.py           ctypes -> Swift (47 functions)
    _config.py           Config dataclass
    _dispatch.py         estimator registry + wrapping
    accelerate.py        @accelerate decorator + accelerate_context
    estimators/
      _base.py           BaseGPUEstimator abstract class
      _registry.py       Estimator registry (17 estimators)
      linear_model.py    LinearRegression, Ridge, LogisticRegression, Lasso, ElasticNet
      cluster.py         KMeans, DBSCAN
      decomposition.py   TruncatedSVD
      ensemble.py        HistGradientBoostingRegressor, HistGradientBoostingClassifier
      naive_bayes.py     GaussianNB
      neighbors.py       KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
      preprocessing.py   StandardScaler, MinMaxScaler, RobustScaler
    utils.py
  skmetal_bridge/        Swift + Metal
    Sources/SkMetalBridge/
      Bridge.swift       47 @_cdecl exports
      MetalContext.swift

      Kernels/*.metal    14 Metal kernel files
  benchmarks/
    run_compare.py       benchmark runner
    benchmark_suite.py   full suite
    baseline.json
  tests/
    test_correctness.py  18 correctness tests (16 parametrized + 2 standalone)
    test_dispatch.py     7 dispatch tests
  build.sh
  pyproject.toml
  .github/workflows/
    ci.yml               build + test + benchmark
    release.yml          PyPI on tag
  LICENSE                MIT
```

---

## Tests

```
test_correctness: 18/18 pass - 16 estimator parametrizations + pipeline + device info
test_dispatch:    7/7 pass - registry, wrapping, pipeline, decorator
```

---

## Benchmarks

```bash
python benchmarks/run_compare.py
```

---

## Development

```bash
git clone https://github.com/your-org/skmetal.git
cd skmetal
bash build.sh
pip install -e ".[dev]"
pytest tests/
python benchmarks/run_compare.py
```

### Adding a new estimator

1. Create `skmetal/estimators/my_model.py` with `MetalMyModel(BaseGPUEstimator)`
2. Register in `_registry.py` (`GPU_ESTIMATORS` + `PIPELINE_PATTERNS`)
3. Add module path in `_dispatch.py` (`module_map`)
4. Write a Metal kernel if needed
5. Add a parametrized test case in `test_correctness.py`

---

## License

MIT License. See `LICENSE`.
