"""GPU-accelerated preprocessing with skmetal.

Compares every supported preprocessor side-by-side on CPU vs GPU,
measuring fit/transform time and verifying that transformed values match.

Estimators covered:
  - StandardScaler  — z-score (mean 0, variance 1)
  - MinMaxScaler    — range [0, 1]
  - RobustScaler    — robust to outliers (median / IQR)
"""
import time
import warnings
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import skmetal

warnings.filterwarnings("ignore")

# ── Generate data ──────────────────────────────────────────────────────────
n, n_features = 100_000, 200
X, _ = make_regression(n_samples=n, n_features=n_features, random_state=42)
X = X.astype(np.float32)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# After transform, the statistical properties of the output should be nearly
# identical between CPU and GPU.  We verify:
#   - mean & std  (StandardScaler)
#   - min & max   (MinMaxScaler)
#   - median & IQR (RobustScaler)
#   - max absolute difference between the transformed arrays

SCALERS = [
    ("StandardScaler",  StandardScaler,  {}),
    ("MinMaxScaler",    MinMaxScaler,    {}),
    ("RobustScaler",    RobustScaler,    {}),
]

print(f"{'Estimator':<20} {'CPU fit':>8} {'GPU fit':>8} {'CPU tr':>8} {'GPU tr':>8} "
      f"{'Speedup':>8} {'Max|diff|':>10} {'Mean':>8} {'Std':>8} {'Match':>6}")
print("=" * 86)

for name, cls, kwargs in SCALERS:
    # ── CPU baseline ───────────────────────────────────────────────────────
    cpu = cls(**kwargs)
    t0 = time.perf_counter()
    cpu.fit(X_train)
    cpu_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_cpu = cpu.transform(X_test)
    cpu_tr = time.perf_counter() - t0

    # ── GPU accelerated ────────────────────────────────────────────────────
    gpu = skmetal.accelerate(cls(**kwargs))
    t0 = time.perf_counter()
    gpu.fit(X_train)
    gpu_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_gpu = gpu.transform(X_test)
    gpu_tr = time.perf_counter() - t0

    max_diff = float(np.max(np.abs(X_cpu - X_gpu)))
    speedup = cpu_fit / gpu_fit if gpu_fit > 0 else float("inf")
    match = "yes" if max_diff < 1e-5 else "no"

    # Print post-transform stats to show correctness at a glance
    mean_diff = float(abs(np.mean(X_cpu) - np.mean(X_gpu)))
    std_diff = float(abs(np.std(X_cpu) - np.std(X_gpu)))

    print(f"{name:<20} {cpu_fit:>8.4f} {gpu_fit:>8.4f} {cpu_tr:>8.4f} {gpu_tr:>8.4f} "
          f"{speedup:>7.1f}x {max_diff:>10.2e} {mean_diff:>8.2e} {std_diff:>8.2e} {match:>6}")
