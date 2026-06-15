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

macOS 14+ and Apple Silicon (M1–M5) required. No Xcode needed — the pip package includes a pre-built dylib.

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

## Features

- **Zero-copy GPU execution** — numpy arrays passed directly to Metal via `bytesNoCopy` on unified memory
- **Drop-in acceleration** — decorate any estimator-returning function, wrap an existing instance, or use a context manager
- **Smart dispatch** — automatically routes to CPU for small datasets where GPU overhead dominates; configurable per-estimator thresholds
- **GPU solvers** — Cholesky, FISTA, L-BFGS, IRLS, K-Means fused iterations, KNN tile-then-merge, and more
- **Transparent fallback** — imports cleanly on non-Apple-Silicon machines; all operations fall back to scikit-learn CPU
- **Progress logging** — `skmetal.set_verbose(True)` prints why each estimator chose GPU or CPU

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

Wrap an existing estimator instance:

```python
model = skmetal.accelerate(LinearRegression())
model.fit(X, y)
```

### Context manager

All estimators created inside the block use GPU:

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
skmetal.set_threshold(100_000)        # global min rows for GPU
skmetal.update_threshold("KMeans",    # per-estimator override
                         min_rows=100_000, min_cols=50)
skmetal.reset_thresholds()            # restore defaults

config = skmetal.get_config()
print(config)
# Config(device='gpu', verbose=True, thresholds={...})
```

On non-Apple-Silicon machines `skmetal` imports cleanly and all estimators transparently fall back to scikit-learn CPU. Check `skmetal.METAL_AVAILABLE` at runtime.

---

## Supported Estimators

| Estimator | GPU Strategy | Speedup vs CPU |
|-----------|-------------|----------------|
| `LinearRegression` | Cholesky solve on GPU (MPS GEMM + custom kernel) | **15.8–24.2×** |
| `Ridge` | Fused centering + XTX + XTy + Cholesky (1 dispatch) | *0.19–0.79×* |
| `LogisticRegression` | L-BFGS on GPU (full loop in Swift, fused GPU ops) | ≤ 1× at typical sizes |
| `Lasso` | FISTA (power iteration on CPU, rest on GPU) | 0.67–0.87× |
| `ElasticNet` | FISTA (power iteration on CPU, rest on GPU) | 0.67–0.87× |
| `KMeans` | Single fused command buffer (all iters on GPU) | *0.69×* |
| `DBSCAN` | GPU pairwise distance + per-point neighbor count | depends on density |
| `KNeighborsClassifier` | MPSMatrixFindTopK (k≤16) / insertion-sort (k>16) + fused vote | depends on n/k |
| `KNeighborsRegressor` | Same as KNeighborsClassifier | depends on n/k |
| `NearestNeighbors` | GPU pairwise distance + index | depends on n/k |
| `TruncatedSVD` | Randomized SVD, no centering (all BLAS-3) | **2.53×** |
| `GaussianNB` | GPU mean/var per class | — |
| `StandardScaler` | Fused Welford (1 dispatch) | **8.27×** |
| `MinMaxScaler` | Column min/max with threadgroup tree reduction | — |
| `RobustScaler` | GPU quantile approximation | — |
| `HistGradientBoostingRegressor` | C++ HGBT from sklearn (no custom GPU kernel) | — |
| `HistGradientBoostingClassifier` | C++ HGBT from sklearn (no custom GPU kernel) | — |

*Italic speedups* indicate dispatch-limited at n ≤ 50K. Speedup improves to 2–5× at n ≥ 500K where compute dominates overhead. `Ridge` is always slower on GPU than Apple's CPU Accelerate framework at all tested sizes.

> **GPU routing**: Each estimator has a per-estimator threshold (min rows, min cols) that must be met. Below the threshold the estimator uses CPU. Default thresholds are set to bypass GPU for estimators where the GPU path is slower. Override via `skmetal.update_threshold()` or force GPU with `skmetal.set_device("gpu")`.

---

## How It Works

```
numpy array → np.ctypes.data → UnsafeMutableRawPointer → MTLBuffer(bytesNoCopy:) → Metal GPU
                   |                                                    |
                   +--------- same physical memory (unified) -----------+
```

Apple Silicon's unified memory architecture enables zero-copy data sharing. The Swift bridge exposes `@_cdecl` functions callable from Python via `ctypes`. Each estimator's `fit()`/`predict()` calls the appropriate bridge function, which dispatches Metal Performance Shaders or custom compute kernels.

The library decides per-estimator whether to use GPU or CPU based on:
1. **Availability** — Metal must be present (Apple Silicon + macOS 14+)
2. **Device preference** — `set_device("cpu")` overrides
3. **Thresholds** — data must exceed per-estimator (min_rows, min_cols) thresholds
4. **Verbose logging** — `set_verbose(True)` prints the decision

---

## Project

```
skmetal/
  skmetal/
    __init__.py        # public API: accelerate, config, device_info
    _bridge.py         # ctypes → Swift @_cdecl exports
    _config.py         # Config dataclass, thresholds, device control
    _dispatch.py       # estimator registry + wrapping logic
    accelerate.py      # @accelerate decorator + context manager
    estimators/
      _base.py         # BaseGPUEstimator abstract class
      _registry.py     # estimator registry (17 estimators)
      linear_model.py  # LinearRegression, Ridge, LogisticRegression, Lasso, ElasticNet
      cluster.py       # KMeans, DBSCAN
      decomposition.py # TruncatedSVD
      ensemble.py      # HistGradientBoosting
      naive_bayes.py   # GaussianNB
      neighbors.py     # KNeighbors, NearestNeighbors
      preprocessing.py # StandardScaler, MinMaxScaler, RobustScaler
      svm.py           # SVC predict
  skmetal_bridge/      # Swift + Metal
    Sources/SkMetalBridge/
      CoreBridge.swift     # core @_cdecl exports
      Bridge.swift         # device info, init, warmup
      KMeansBridge.swift   # KMeans assign, inertia, shift, batch fused
      KNNBridge.swift      # KNN tile-then-merge, voting
      LinearAlgebraBridge.swift # GEMM, pairwise distance, column ops
      LinearModelBridge.swift   # Cholesky solve, FISTA, Ridge
      LogisticBridge.swift      # L-BFGS / IRLS GPU loop
      PreprocessingBridge.swift # scaler_fit, column_minmax, column_transform
      SVCBridge.swift           # SVC predict
      SVTreeBridge.swift        # union-find, tree predict
      MetalContext.swift
      Kernels/*.metal      # 14 kernel files
  benchmarks/
    run_compare.py
  tests/               # 104 tests across 7 files
    test_correctness.py
    test_dispatch.py
    test_kernels.py
    test_config.py
    test_accelerate.py
    test_stress.py
    test_fallback.py
  pyproject.toml
  .github/workflows/
    ci.yml              # build + ruff + pytest + benchmarks on push
    release.yml         # PyPI publish on v* tag
```

---

## Development

```bash
git clone https://github.com/abderahmane-ai/skmetal.git
cd skmetal

# (optional) Install skmetal in editable mode
pip install -e ".[dev]"

# Build Swift + Metal (required after any .metal or .swift change)
cd skmetal_bridge
bash compile_metal.sh
swift build --configuration release
cp .build/arm64-apple-macosx/release/libSkMetalBridge.dylib ../skmetal/
cd ..

# Run tests
pytest tests/ -q

# Lint
ruff check skmetal/ tests/
```

### Adding a new estimator

1. Create `skmetal/estimators/my_model.py` with `MetalMyModel(BaseGPUEstimator)`
2. Register in `_registry.py` (`GPU_ESTIMATORS` + `PIPELINE_PATTERNS`)
3. Add module path in `_dispatch.py` (`module_map`)
4. Write a Metal kernel if needed
5. Add a parametrized test case in `test_correctness.py`

---

## License

MIT
