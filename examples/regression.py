"""GPU-accelerated regression with skmetal.

Compares every supported regression estimator side-by-side on CPU vs GPU,
measuring fit/predict time, RMSE, and R².

Estimators covered:
  - LinearRegression, Ridge, Lasso, ElasticNet   — linear models
  - KNeighborsRegressor                          — distance-based
  - HistGradientBoostingRegressor                — gradient-boosted trees
"""
import time
import warnings
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import skmetal

warnings.filterwarnings("ignore")

# ── Generate data ──────────────────────────────────────────────────────────
n, n_features = 10_000, 50
X, y = make_regression(n_samples=n, n_features=n_features, noise=0.1, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

REGRESSORS = [
    ("LinearRegression",         LinearRegression,                {}),
    ("Ridge",                    Ridge,                           {"alpha": 1.0}),
    ("Lasso",                    Lasso,                           {"alpha": 0.01, "max_iter": 1000}),
    ("ElasticNet",               ElasticNet,                      {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 1000}),
    ("KNeighborsRegressor",      KNeighborsRegressor,             {"n_neighbors": 5}),
    ("HistGradientBoostingRegressor",
                                 HistGradientBoostingRegressor,    {"max_iter": 100, "max_depth": 5, "random_state": 42}),
]

print(f"{'Estimator':<35} {'CPU fit':>8} {'GPU fit':>8} {'CPU pred':>8} {'GPU pred':>8} "
      f"{'Speedup':>8} {'RMSE':>8} {'R²':>6} {'Match':>6}")
print("=" * 100)

for name, cls, kwargs in REGRESSORS:
    # ── CPU baseline ───────────────────────────────────────────────────────
    cpu = cls(**kwargs)
    t0 = time.perf_counter()
    cpu.fit(X_train, y_train)
    cpu_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_cpu = cpu.predict(X_test)
    cpu_pred = time.perf_counter() - t0

    cpu_rmse = float(np.sqrt(mean_squared_error(y_test, y_cpu)))
    cpu_r2 = float(r2_score(y_test, y_cpu))

    # ── GPU accelerated ────────────────────────────────────────────────────
    gpu = skmetal.accelerate(cls(**kwargs))
    t0 = time.perf_counter()
    gpu.fit(X_train, y_train)
    gpu_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_gpu = gpu.predict(X_test)
    gpu_pred = time.perf_counter() - t0

    gpu_rmse = float(np.sqrt(mean_squared_error(y_test, y_gpu)))
    gpu_r2 = float(r2_score(y_test, y_gpu))

    speedup = cpu_fit / gpu_fit if gpu_fit > 0 else float("inf")
    match = "yes" if max(abs(cpu_rmse - gpu_rmse), abs(cpu_r2 - gpu_r2)) < 0.01 else "no"

    print(f"{name:<35} {cpu_fit:>8.4f} {gpu_fit:>8.4f} {cpu_pred:>8.4f} {gpu_pred:>8.4f} "
          f"{speedup:>7.1f}x {cpu_rmse:>8.4f} {cpu_r2:>6.3f} {match:>6}")
