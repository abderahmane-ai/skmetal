"""Multi-step pipeline: CPU vs GPU end-to-end benchmark.

Compares a full sklearn Pipeline (scaler + PCA + model) on CPU vs GPU
using the @skmetal.accelerate decorator.
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


def benchmark_pipeline(est_name, model_cls, n, d, k=20, **kwargs):
    X, y = make_regression(n_samples=n, n_features=d, noise=0.1, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    @skmetal.accelerate
    def gpu_pipe():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(k, d), random_state=42)),
            ("model", model_cls(**kwargs)),
        ])

    cpu_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(k, d), random_state=42)),
        ("model", model_cls(**kwargs)),
    ])

    cpu_pipe.fit(X_train, y_train)
    _ = gpu_pipe()
    _.fit(X_train, y_train)

    cpu_t = []
    for _ in range(5):
        p = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(k, d), random_state=42)),
            ("model", model_cls(**kwargs)),
        ])
        t0 = time.perf_counter()
        p.fit(X_train, y_train)
        cpu_t.append(time.perf_counter() - t0)

    gpu_t = []
    for _ in range(5):
        p = skmetal.accelerate(Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(k, d), random_state=42)),
            ("model", model_cls(**kwargs)),
        ]))
        t0 = time.perf_counter()
        p.fit(X_train, y_train)
        gpu_t.append(time.perf_counter() - t0)

    cpu_m = float(np.median(cpu_t))
    gpu_m = float(np.median(gpu_t))

    y_cpu = cpu_pipe.predict(X_test)
    y_gpu = _.predict(X_test)
    rmse_cpu = np.sqrt(mean_squared_error(y_test, y_cpu))
    rmse_gpu = np.sqrt(mean_squared_error(y_test, y_gpu))

    return {
        "estimator": f"{est_name}",
        "n": n,
        "d": d,
        "cpu_time": cpu_m,
        "gpu_time": gpu_m,
        "speedup": cpu_m / gpu_m if gpu_m > 0 else 0,
        "rmse_cpu": rmse_cpu,
        "rmse_gpu": rmse_gpu,
    }


cases = [
    ("Ridge", Ridge, 20000, 100, {"alpha": 1.0}),
    ("Ridge", Ridge, 100000, 500, {"alpha": 1.0}),
    ("LinearRegression", LinearRegression, 20000, 100, {}),
    ("LinearRegression", LinearRegression, 100000, 500, {}),
]

print(f"{'Pipeline (scaler+PCA+model)':<35} {'n':>8} {'d':>6} {'CPU(s)':>8} {'GPU(s)':>8} {'Speedup':>8} {'RMSE match':>10}")
print("-" * 85)

for name, cls, n, d, kw in cases:
    r = benchmark_pipeline(name, cls, n, d, **kw)
    rmse_diff = abs(r["rmse_cpu"] - r["rmse_gpu"])
    rmse_ok = "yes" if rmse_diff < 0.01 else f"diff {rmse_diff:.4f}"
    s = f"{r['speedup']:.2f}x" if r['speedup'] >= 0.1 else f"{r['speedup']:.3f}x"
    label = f"StandardScaler+PCA+{name}"
    print(f"{label:<35} {r['n']:>8} {r['d']:>6} {r['cpu_time']:>8.3f} {r['gpu_time']:>8.3f} {s:>8} {rmse_ok:>10}")
