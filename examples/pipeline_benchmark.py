"""Multi-step pipeline benchmark: CPU vs GPU end-to-end.

Demonstrates skmetal.accelerate applied to a full sklearn Pipeline
(StandardScaler → PCA → linear model), measuring wall-clock time and
verifying that RMSE matches the CPU baseline.

Pipelines tested:
  - StandardScaler + PCA + Ridge
  - StandardScaler + PCA + LinearRegression
"""
import time
import warnings
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
import skmetal

warnings.filterwarnings("ignore")


def benchmark(name, model_cls, n, d, k=20, **kwargs):
    """Benchmark a single pipeline configuration."""

    # ── Generate data ──────────────────────────────────────────────────────
    X, y = make_regression(n_samples=n, n_features=d, noise=0.1, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── CPU pipeline ───────────────────────────────────────────────────────
    cpu = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(k, d), random_state=42)),
        ("model", model_cls(**kwargs)),
    ])
    t0 = time.perf_counter()
    cpu.fit(X_train, y_train)
    cpu_fit = time.perf_counter() - t0
    y_cpu = cpu.predict(X_test)
    cpu_rmse = float(np.sqrt(mean_squared_error(y_test, y_cpu)))

    # ── GPU pipeline (via skmetal.accelerate) ──────────────────────────────
    gpu = skmetal.accelerate(Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(k, d), random_state=42)),
        ("model", model_cls(**kwargs)),
    ]))
    t0 = time.perf_counter()
    gpu.fit(X_train, y_train)
    gpu_fit = time.perf_counter() - t0
    y_gpu = gpu.predict(X_test)
    gpu_rmse = float(np.sqrt(mean_squared_error(y_test, y_gpu)))

    speedup = cpu_fit / gpu_fit if gpu_fit > 0 else float("inf")
    rmse_match = "yes" if abs(cpu_rmse - gpu_rmse) < 0.01 else f"diff {abs(cpu_rmse - gpu_rmse):.4f}"

    return {
        "name": f"Scaler+PCA+{name}",
        "n": n,
        "d": d,
        "cpu_time": cpu_fit,
        "gpu_time": gpu_fit,
        "speedup": speedup,
        "cpu_rmse": cpu_rmse,
        "gpu_rmse": gpu_rmse,
        "rmse_match": rmse_match,
    }


# ── Run benchmarks ─────────────────────────────────────────────────────────
cases = [
    ("Ridge",             Ridge,             20_000,  100, {"alpha": 1.0}),
    ("Ridge",             Ridge,            100_000,  500, {"alpha": 1.0}),
    ("LinearRegression",  LinearRegression,  20_000,  100, {}),
    ("LinearRegression",  LinearRegression, 100_000,  500, {}),
]

print(f"{'Pipeline (Scaler+PCA+model)':<35} {'n':>8} {'d':>6} {'CPU(s)':>8} "
      f"{'GPU(s)':>8} {'Speedup':>8} {'RMSE':>8} {'Match':>10}")
print("=" * 93)

for name, cls, n, d, kw in cases:
    r = benchmark(name, cls, n, d, **kw)
    s = f"{r['speedup']:.1f}x" if r['speedup'] >= 0.1 else f"{r['speedup']:.3f}x"
    print(f"{r['name']:<35} {r['n']:>8} {r['d']:>6} {r['cpu_time']:>8.4f} "
          f"{r['gpu_time']:>8.4f} {s:>8} {r['cpu_rmse']:>8.4f} {r['rmse_match']:>10}")
