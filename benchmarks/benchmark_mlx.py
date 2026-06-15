"""MLX-accelerated estimator benchmarks.

Compares MLX GPU vs CPU sklearn for compute-intensive estimators.
Requires MLX installed (``pip install skmetal[mlx]``).
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skmetal"))

import numpy as np
from sklearn.datasets import make_regression, make_classification, make_blobs
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from skmetal import accelerate
from skmetal.estimators._mlx_registry import has_mlx


def _has_mlx():
    return has_mlx()


def benchmark(name, est_factory, data_factory, n_runs=3):
    X, y = data_factory()
    X = X.astype(np.float32)
    has_y = y is not None
    args = (X, y) if has_y else (X,)

    cpu = est_factory()
    cpu.fit(*args)

    gpu = accelerate(est_factory())
    gpu.fit(*args)

    cpu_t = []
    for _ in range(n_runs):
        c = est_factory()
        t0 = time.perf_counter()
        c.fit(*args)
        cpu_t.append(time.perf_counter() - t0)

    gpu_t = []
    for _ in range(n_runs):
        g = accelerate(est_factory())
        t0 = time.perf_counter()
        g.fit(*args)
        gpu_t.append(time.perf_counter() - t0)

    cpu_m = float(np.median(cpu_t))
    gpu_m = float(np.median(gpu_t))
    return {
        "estimator": name,
        "backend": "mlx",
        "cpu_time": cpu_m,
        "gpu_time": gpu_m,
        "speedup": cpu_m / gpu_m if gpu_m > 0 else 0.0,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
    }


MLX_BENCHMARKS = [
    ("LogisticRegression_MLX", LogisticRegression,
     lambda: make_classification(n_samples=100000, n_features=200, random_state=42)),
    ("Lasso_MLX", Lasso,
     lambda: make_regression(n_samples=50000, n_features=200, random_state=42)),
    ("ElasticNet_MLX", lambda: ElasticNet(l1_ratio=0.5),
     lambda: make_regression(n_samples=50000, n_features=200, random_state=42)),
    ("KMeans_MLX", lambda: KMeans(n_clusters=50, max_iter=300, random_state=42),
     lambda: make_blobs(n_samples=500000, n_features=100, centers=50, random_state=42)),
    ("TruncatedSVD_MLX", lambda: TruncatedSVD(n_components=50, random_state=42),
     lambda: make_regression(n_samples=100000, n_features=500, random_state=42)),
]


if __name__ == "__main__":
    if not _has_mlx():
        print("MLX not installed. Install with: pip install skmetal[mlx]")
        sys.exit(0)

    results = []
    print(f"{'Estimator':<24} {'Samples':>8} {'Feats':>6} {'CPU(s)':>8} {'MLX(s)':>8} {'Speedup':>8}")
    print("-" * 68)
    for name, est, df in MLX_BENCHMARKS:
        r = benchmark(name, est, df)
        results.append(r)
        s = f"{r['speedup']:.2f}x" if r['speedup'] >= 0.1 else f"{r['speedup']:.3f}x"
        print(f"{r['estimator']:<24} {r['n_samples']:>8} {r['n_features']:>6} "
              f"{r['cpu_time']:>8.3f} {r['gpu_time']:>8.3f} {s:>8}")

    baseline = Path(__file__).parent / "baseline_mlx.json"
    with open(baseline, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMLX baseline saved to {baseline}")
