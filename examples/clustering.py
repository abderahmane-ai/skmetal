"""GPU-accelerated clustering with skmetal.

Compares every supported clustering estimator side-by-side on CPU vs GPU,
measuring fit time, inertia / silhouette score, and cluster agreement.

Estimators covered:
  - KMeans    — centroid-based
  - DBSCAN    — density-based
"""
import time
import warnings
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import skmetal

warnings.filterwarnings("ignore")

# ── Generate data ──────────────────────────────────────────────────────────
n, n_features, n_clusters = 10_000, 50, 8
X, y_true = make_blobs(
    n_samples=n, centers=n_clusters, n_features=n_features,
    cluster_std=2.0, random_state=42,
)
X = X.astype(np.float32)

# Clustering metrics don't use ground-truth labels at test time, but we use
# adjusted_rand_score to measure label agreement between CPU and GPU results.
# We also report silhouette_score as an intrinsic quality metric.

CLUSTERERS = [
    ("KMeans",  KMeans,  {"n_clusters": n_clusters, "n_init": 5, "random_state": 42}),
    ("DBSCAN",  DBSCAN,  {"eps": 1.5, "min_samples": 5}),
]

print(f"{'Estimator':<20} {'CPU fit':>8} {'GPU fit':>8} {'Speedup':>8} "
      f"{'Silhouette':>10} {'ARI':>8} {'Match':>6}")
print("=" * 70)

for name, cls, kwargs in CLUSTERERS:
    # ── CPU baseline ───────────────────────────────────────────────────────
    cpu = cls(**kwargs)
    t0 = time.perf_counter()
    cpu.fit(X)
    cpu_fit = time.perf_counter() - t0
    cpu_labels = cpu.labels_

    # Silhouette (skip DBSCAN if too many noise points)
    cpu_sil = float(silhouette_score(X, cpu_labels)) if len(set(cpu_labels)) > 1 else float("nan")

    # ── GPU accelerated ────────────────────────────────────────────────────
    gpu = skmetal.accelerate(cls(**kwargs))
    t0 = time.perf_counter()
    gpu.fit(X)
    gpu_fit = time.perf_counter() - t0
    gpu_labels = gpu.labels_

    gpu_sil = float(silhouette_score(X, gpu_labels)) if len(set(gpu_labels)) > 1 else float("nan")

    # Compare cluster assignments (adjusted Rand index, 1 = perfect match)
    ari = float(adjusted_rand_score(cpu_labels, gpu_labels))

    speedup = cpu_fit / gpu_fit if gpu_fit > 0 else float("inf")
    match = "yes" if ari > 0.99 else "no"

    print(f"{name:<20} {cpu_fit:>8.4f} {gpu_fit:>8.4f} {speedup:>7.1f}x "
          f"{cpu_sil:>10.4f} {ari:>8.4f} {match:>6}")
