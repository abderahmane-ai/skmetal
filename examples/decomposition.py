"""GPU-accelerated dimensionality reduction with skmetal.

Compares every supported decomposition estimator side-by-side on CPU vs GPU,
measuring fit/transform time and reconstruction error.

Estimators covered:
  - PCA           — principal component analysis (SVD-based)
  - TruncatedSVD  — randomized SVD for sparse-ish data
"""
import time
import warnings
import numpy as np
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import skmetal

warnings.filterwarnings("ignore")

# ── Generate data ──────────────────────────────────────────────────────────
n, n_features, n_components = 10_000, 100, 20
X, _ = make_regression(n_samples=n, n_features=n_features, random_state=42)
X = X.astype(np.float32)

# Decomposition quality: after transform + project back, the reconstruction
# error should be nearly identical between CPU and GPU.

DECOMPOSERS = [
    ("PCA",          PCA,          {"n_components": n_components, "random_state": 42}),
    ("TruncatedSVD", TruncatedSVD, {"n_components": n_components, "random_state": 42}),
]

print(f"{'Estimator':<20} {'CPU fit':>8} {'GPU fit':>8} {'CPU tr':>8} {'GPU tr':>8} "
      f"{'Speedup':>8} {'CPU err':>8} {'GPU err':>8} {'Match':>6}")
print("=" * 84)

for name, cls, kwargs in DECOMPOSERS:
    # ── CPU baseline ───────────────────────────────────────────────────────
    cpu = cls(**kwargs)
    t0 = time.perf_counter()
    cpu.fit(X)
    cpu_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_cpu_t = cpu.transform(X)
    cpu_tr = time.perf_counter() - t0
    X_cpu_recon = X_cpu_t @ cpu.components_
    cpu_err = float(np.linalg.norm(X - X_cpu_recon))

    # ── GPU accelerated ────────────────────────────────────────────────────
    gpu = skmetal.accelerate(cls(**kwargs))
    t0 = time.perf_counter()
    gpu.fit(X)
    gpu_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_gpu_t = gpu.transform(X)
    gpu_tr = time.perf_counter() - t0
    X_gpu_recon = X_gpu_t @ gpu.components_
    gpu_err = float(np.linalg.norm(X - X_gpu_recon))

    speedup = cpu_fit / gpu_fit if gpu_fit > 0 else float("inf")
    # TruncatedSVD uses randomness → expect slightly different components
    match = "yes" if gpu_err <= cpu_err * 1.1 else "no"

    print(f"{name:<20} {cpu_fit:>8.4f} {gpu_fit:>8.4f} {cpu_tr:>8.4f} {gpu_tr:>8.4f} "
          f"{speedup:>7.1f}x {cpu_err:>8.2f} {gpu_err:>8.2f} {match:>6}")
