"""GPU-accelerated classification with skmetal.

Compares every supported classifier side-by-side on CPU vs GPU,
measuring fit/predict time, accuracy, F1, and ROC-AUC.

Estimators covered:
  - LogisticRegression                        — linear classifier
  - KNeighborsClassifier                      — distance-based
  - GaussianNB                                — probabilistic
  - HistGradientBoostingClassifier            — gradient-boosted trees
"""
import time
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier
import skmetal

warnings.filterwarnings("ignore")

# ── Generate data ──────────────────────────────────────────────────────────
n, n_features = 10_000, 50
X, y = make_classification(
    n_samples=n, n_features=n_features, n_informative=20,
    n_redundant=10, random_state=42,
)
X = X.astype(np.float32)
y = y.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

CLASSIFIERS = [
    ("LogisticRegression",         LogisticRegression,                {"C": 1.0, "max_iter": 500, "random_state": 42}),
    ("KNeighborsClassifier",       KNeighborsClassifier,              {"n_neighbors": 5}),
    ("GaussianNB",                 GaussianNB,                        {}),
    ("HistGradientBoostingClassifier",
                                   HistGradientBoostingClassifier,    {"max_iter": 100, "max_depth": 5, "random_state": 42}),
]

print(f"{'Estimator':<35} {'CPU fit':>8} {'GPU fit':>8} {'CPU pred':>8} {'GPU pred':>8} "
      f"{'Speedup':>8} {'Acc':>6} {'F1':>6} {'AUC':>6} {'Match':>6}")
print("=" * 100)

for name, cls, kwargs in CLASSIFIERS:
    # ── CPU baseline ───────────────────────────────────────────────────────
    cpu = cls(**kwargs)
    t0 = time.perf_counter()
    cpu.fit(X_train, y_train)
    cpu_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_cpu = cpu.predict(X_test)
    cpu_pred = time.perf_counter() - t0

    cpu_acc = float(accuracy_score(y_test, y_cpu))
    cpu_f1 = float(f1_score(y_test, y_cpu))

    # ROC-AUC needs predict_proba (all classifiers here support it)
    cpu_auc = float(roc_auc_score(y_test, cpu.predict_proba(X_test)[:, 1]))

    # ── GPU accelerated ────────────────────────────────────────────────────
    gpu = skmetal.accelerate(cls(**kwargs))
    t0 = time.perf_counter()
    gpu.fit(X_train, y_train)
    gpu_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_gpu = gpu.predict(X_test)
    gpu_pred = time.perf_counter() - t0

    gpu_acc = float(accuracy_score(y_test, y_gpu))
    gpu_f1 = float(f1_score(y_test, y_gpu))
    gpu_auc = float(roc_auc_score(y_test, gpu.predict_proba(X_test)[:, 1]))

    speedup = cpu_fit / gpu_fit if gpu_fit > 0 else float("inf")
    match = "yes" if max(abs(cpu_acc - gpu_acc), abs(cpu_f1 - gpu_f1), abs(cpu_auc - gpu_auc)) < 0.01 else "no"

    print(f"{name:<35} {cpu_fit:>8.4f} {gpu_fit:>8.4f} {cpu_pred:>8.4f} {gpu_pred:>8.4f} "
          f"{speedup:>7.1f}x {cpu_acc:>6.3f} {cpu_f1:>6.3f} {cpu_auc:>6.3f} {match:>6}")
