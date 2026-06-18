<div align="center">

# skmetal

**Apple Silicon GPU acceleration for scikit-learn**

[![PyPI](https://img.shields.io/pypi/v/skmetal?color=3776AB&style=flat-square)](https://pypi.org/project/skmetal/)
[![Python](https://img.shields.io/pypi/pyversions/skmetal?logo=python&style=flat-square)](https://pypi.org/project/skmetal/)
[![Platform](https://img.shields.io/badge/platform-macOS%2014%2B%20|%20Apple%20Silicon-000000?logo=apple&style=flat-square)](https://github.com/abderahmane-ai/skmetal)
[![CI](https://github.com/abderahmane-ai/skmetal/actions/workflows/ci.yml/badge.svg)](https://github.com/abderahmane-ai/skmetal/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/abderahmane-ai/skmetal)
[![Estimators](https://img.shields.io/badge/estimators-19-blue?style=flat-square)]()

</div>

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

## Overview

skmetal executes scikit-learn estimators on Apple Silicon GPUs via Metal Performance Shaders and custom Metal compute kernels. Decorate any function that returns an estimator with `@skmetal.accelerate` and `fit()`/`predict()` run on the GPU — no code changes.

Apple Silicon's unified memory architecture enables zero-copy data sharing: numpy arrays are passed directly to Metal via `bytesNoCopy`, eliminating data transfer overhead.

---

## Installation

```bash
pip install skmetal
```

macOS 14+ and Apple Silicon (M1–M5) required.

For development:
```bash
git clone https://github.com/abderahmane-ai/skmetal.git
cd skmetal/skmetal_bridge
bash compile_metal.sh
swift build --configuration release
cp .build/arm64-apple-macosx/release/libSkMetalBridge.dylib ../skmetal/
cd ../..
pip install -e "skmetal[dev]"
```

---

## Benchmarks (M4 Air, 16 GB)

| Estimator | Data Size | Speedup |
|-----------|-----------|---------|
| `KMeans` (MLX) | 200,000 × 64, k=500 | **30×** |
| `KMeans` (MLX) | 100,000 × 128, k=200 | **16×** |
| `StandardScaler` | 1,000,000 × 100 | **10×** |
| `LinearRegression` | 200,000 × 500 | **8.4×** |
| `TruncatedSVD` | 100,000 × 500 | **2.9×** |
| `MinMaxScaler` | 1,000,000 × 100 | **1.2×** |
| `LogisticRegression` | 100,000 × 200 | 0.9× |
| `Ridge` | 200,000 × 500 | 0.9× |

KMeans MLX requires `pip install skmetal[mlx]`.

## Benchmarks (M3 Pro, 18 GB)

| Estimator | Data Size | Speedup |
|-----------|-----------|---------|
| `LinearRegression` | 200,000 × 500 | **10.42×** |
| `StandardScaler` | 1,000,000 × 100 | **9.93×** |
| `TruncatedSVD` | 100,000 × 500 | **4.20×** |
| `MinMaxScaler` | 1,000,000 × 100 | **1.36×** |
| `KMeans` | 500,000 × 100 | **1.32×** |
| `Ridge` | 200,000 × 500 | 0.94× |
| `LogisticRegression` | 100,000 × 200 | 0.93× |

See [skmetal/README.md](skmetal/README.md) for the full PyPI readme with architecture, kernel listing, and development guide.

---

## Quick Start

```python
import skmetal
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Decorator
@skmetal.accelerate
def model():
    return LogisticRegression()

clf = model()
clf.fit(X_train, y_train)

# Pipeline
@skmetal.accelerate
def pipe():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ])

p = pipe()
p.fit(X, y)

# Check GPU
if skmetal.METAL_AVAILABLE:
    print(skmetal.device_info())

# Force CPU
skmetal.set_device("cpu")
```

---

## License

MIT — see [LICENSE](LICENSE).
