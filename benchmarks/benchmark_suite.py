"""Benchmark suite for skmetal GPU acceleration."""

import time
import json
import numpy as np
from pathlib import Path
from sklearn.datasets import make_regression, make_classification, make_blobs
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import skmetal


def benchmark(name, est_factory, data_factory, n_runs=3):
    X, y = data_factory()
    X = X.astype(np.float32)
    has_y = y is not None
    args = (X, y) if has_y else (X,)

    cpu = est_factory()
    cpu.fit(*args)

    gpu = skmetal.accelerate(est_factory())
    gpu.fit(*args)

    cpu_t = []
    for _ in range(n_runs):
        c = est_factory()
        t0 = time.perf_counter()
        c.fit(*args)
        cpu_t.append(time.perf_counter() - t0)

    gpu_t = []
    for _ in range(n_runs):
        g = skmetal.accelerate(est_factory())
        t0 = time.perf_counter()
        g.fit(*args)
        gpu_t.append(time.perf_counter() - t0)

    cpu_m = float(np.median(cpu_t))
    gpu_m = float(np.median(gpu_t))
    return {
        "estimator": name,
        "cpu_time": cpu_m,
        "gpu_time": gpu_m,
        "speedup": cpu_m / gpu_m if gpu_m > 0 else 0.0,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
    }


BENCHMARKS = [
    ("LinearRegression", LinearRegression,
     lambda: make_regression(n_samples=200000, n_features=500, random_state=42)),
    ("Ridge", Ridge,
     lambda: make_regression(n_samples=200000, n_features=500, random_state=42)),
    ("LogisticRegression", LogisticRegression,
     lambda: make_classification(n_samples=100000, n_features=200, random_state=42)),
    ("TruncatedSVD", lambda: TruncatedSVD(n_components=50, random_state=42),
     lambda: make_regression(n_samples=100000, n_features=500, random_state=42)),
    ("PCA", lambda: PCA(n_components=50, random_state=42),
     lambda: make_regression(n_samples=100000, n_features=1000, random_state=42)),
    ("KMeans", lambda: KMeans(n_clusters=50, max_iter=300, random_state=42),
     lambda: make_blobs(n_samples=500000, n_features=100, centers=50, random_state=42)),
    ("StandardScaler", StandardScaler,
     lambda: make_regression(n_samples=1000000, n_features=100, random_state=42)),
    ("MinMaxScaler", MinMaxScaler,
     lambda: make_regression(n_samples=1000000, n_features=100, random_state=42)),
]


if __name__ == "__main__":
    results = []
    print(f"{'Estimator':<20} {'Samples':>8} {'Feats':>6} {'CPU(s)':>8} {'GPU(s)':>8} {'Speedup':>8}")
    print("-" * 62)
    for name, est, df in BENCHMARKS:
        r = benchmark(name, est, df)
        results.append(r)
        s = f"{r['speedup']:.2f}x" if r['speedup'] >= 0.1 else f"{r['speedup']:.3f}x"
        print(f"{r['estimator']:<20} {r['n_samples']:>8} {r['n_features']:>6} "
              f"{r['cpu_time']:>8.3f} {r['gpu_time']:>8.3f} {s:>8}")

    baseline = Path(__file__).parent / "baseline.json"
    with open(baseline, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline saved to {baseline}")
