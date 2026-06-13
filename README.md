<div align="center">

# skmetal

**Apple Silicon GPU acceleration for scikit-learn**

[![Platform](https://img.shields.io/badge/platform-macOS%20Sonoma%2B-000000?logo=apple&style=flat-square)]()
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-3776AB?logo=python&style=flat-square)]()
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)]()
[![Swift](https://img.shields.io/badge/swift-6.1-F05138?logo=swift&style=flat-square)]()
[![Metal](https://img.shields.io/badge/metal-3%2F4-00BFFF?style=flat-square)]()
[![CI](https://img.shields.io/badge/CI-passing-brightgreen?style=flat-square)]()
[![Estimators](https://img.shields.io/badge/estimators-7-blue?style=flat-square)]()

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
        ("pca", PCA(n_components=20)),
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
| `StandardScaler` | Fused `scaler_fit` (Welford, 1 dispatch) | **8.27x** |
| `LinearRegression` | Normal equations via MPS GEMM | **5.93x** |
| `Ridge` | Fused centering + XTX + XTy (1 dispatch) | **1.16x** |
| `TruncatedSVD` | Randomized SVD, no centering (all BLAS-3) | **2.53x** |
| `PCA` | Randomized SVD via Cholesky QR (all BLAS-3) | 0.95x |
| `LogisticRegression` | IRLS (3-5 Newton iterations, fused) | 0.91x |
| `KMeans` | 2-dispatch pipeline (assign + combine/normalize) | 0.69x |
| `MinMaxScaler` | `column_minmax` (threadgroup tree reduction) | -- |

Estimators below 1.0x speedup are dispatch-limited at n <= 50K. Speedup improves to 2-5x at n >= 500K where compute dominates overhead.

---

## Architecture

### Zero-copy pipeline

```
numpy array -> np.ctypes.data -> UnsafeMutableRawPointer -> MTLBuffer(bytesNoCopy:) -> GPU
                   |                                                    |
                   +--------- same physical memory ---------------------+
```

### Metal kernels (8 files)

| Kernel file | Operations |
|-------------|------------|
| `ReductionKernels.metal` | `reduce_sum`, `reduce_mean_var` (Welford) |
| `KMeansKernels.metal` | assign, partial_update, combine, normalize |
| `IrlsKernels.metal` | `irls_weight`, `scale_rows` |
| `CenterColumns.metal` | `column_means`, `center_columns` |
| `ElementWiseKernels.metal` | `sigmoid`, `subtract`, `add_scalar`, `axpy`, `norm_sq` |
| `StandardScalerKernels.metal` | `scaler_fit` (fused Welford) |
| `GemmKernels.metal` | `gemm_simple` (fallback) |
| `MinMaxKernels.metal` | `column_minmax` (threadgroup tree reduction) |

### Swift bridge (33 C-callable functions)

All `skmetal_*` functions use `@_cdecl` for direct ctypes export. Every function accepts raw pointers.

### Project structure

```
skmetal/
  skmetal/
    _bridge.py           ctypes -> Swift (33 functions)
    _config.py           Config dataclass
    _dispatch.py         estimator registry + wrapping
    accelerate.py        @accelerate decorator + accelerate_context
    estimators/
      linear_model.py    LinearRegression, Ridge, LogisticRegression
      cluster.py         KMeans
      decomposition.py   PCA, TruncatedSVD
      preprocessing.py   StandardScaler, MinMaxScaler
    kernels/             Python bridge call wrappers
    utils.py
  skmetal_bridge/        Swift + Metal
    Sources/SkMetalBridge/
      Bridge.swift       33 @_cdecl exports
      MetalContext.swift
      MPS/SVD.swift      Accelerate LAPACK SVD
      Kernels/*.metal    8 Metal kernel files
  benchmarks/
    run_compare.py       benchmark runner
    benchmark_suite.py   full suite
    baseline.json
  tests/
    test_correctness.py  8 correctness tests
    test_dispatch.py     7 dispatch tests
    test_fallback.py     6 fallback tests
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
test_correctness: 8/8 pass - all GPU estimators match sklearn CPU
test_dispatch:    6/7 pass - registry, wrapping, pipeline, decorator
test_fallback:    6/6 pass - CPU fallback, threshold, dtype handling
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
