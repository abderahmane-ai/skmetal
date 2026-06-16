"""Benchmark and compare against baseline."""
import time
import json
import sys
import numpy as np
from pathlib import Path
from sklearn.datasets import make_regression, make_classification, make_blobs
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import skmetal

sys.path.insert(0, str(Path(__file__).parent.parent))


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
     lambda: make_regression(n_samples=50000, n_features=200, random_state=42)),
    ("Ridge", Ridge,
     lambda: make_regression(n_samples=50000, n_features=200, random_state=42)),
    ("LogisticRegression", LogisticRegression,
     lambda: make_classification(n_samples=20000, n_features=100, random_state=42)),
    ("TruncatedSVD", lambda: TruncatedSVD(n_components=20, random_state=42),
     lambda: make_regression(n_samples=20000, n_features=500, random_state=42)),
    ("KMeans", lambda: KMeans(n_clusters=20, max_iter=30, random_state=42),
     lambda: make_blobs(n_samples=50000, n_features=50, centers=20, random_state=42)),
    ("StandardScaler", StandardScaler,
     lambda: make_regression(n_samples=100000, n_features=100, random_state=42)),
    ("MinMaxScaler", MinMaxScaler,
     lambda: make_regression(n_samples=100000, n_features=100, random_state=42)),
]


REGRESSION_THRESHOLD = 0.5  # fail if speedup drops below 50% of baseline


if __name__ == "__main__":
    results = []
    baseline_path = Path(__file__).parent / "baseline.json"
    baseline = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            for b in json.load(f):
                baseline[b["estimator"]] = b.get("speedup", 0)

    print(f"{'Estimator':<20} {'Samples':>8} {'Feats':>6} {'CPU(s)':>8} {'GPU(s)':>8} {'Speedup':>8}  {'vs Base':>8}")
    print("-" * 70)

    regressions = []
    for name, est, df in BENCHMARKS:
        r = benchmark(name, est, df)
        results.append(r)
        s = f"{r['speedup']:.2f}x" if r['speedup'] >= 0.1 else f"{r['speedup']:.3f}x"

        vs = ""
        old = baseline.get(r["estimator"], 0)
        if old > 0:
            c = (r["speedup"] / old - 1) * 100
            vs = f"{c:+3.0f}%"
            if r["speedup"] < old * REGRESSION_THRESHOLD:
                regressions.append(
                    f"{r['estimator']}: {r['speedup']:.2f}x vs baseline {old:.2f}x "
                    f"(< {REGRESSION_THRESHOLD*100:.0f}% threshold)"
                )

        print(f"{r['estimator']:<20} {r['n_samples']:>8} {r['n_features']:>6} "
              f"{r['cpu_time']:>8.3f} {r['gpu_time']:>8.3f} {s:>8}  {vs:>8}")
    print("-" * 70)

    if regressions:
        print(f"\nREGRESSION DETECTED ({len(regressions)} estimator(s)):")
        for r in regressions:
            print(f"  {r}")
        sys.exit(1)
    print("No regressions detected.")
