"""Breast cancer: CPU vs GPU classification pipeline."""
import time
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import skmetal

warnings.filterwarnings("ignore")

X, y = load_breast_cancer(return_X_y=True)
X = X.astype(np.float32)
y = y.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

@skmetal.accelerate
def gpu_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=10, random_state=42)),
        ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])

pipe_cpu = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10, random_state=42)),
    ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
])

print("--- CPU Pipeline ---")
t0 = time.perf_counter()
pipe_cpu.fit(X_train, y_train)
cpu_fit = time.perf_counter() - t0
t0 = time.perf_counter()
y_pred_cpu = pipe_cpu.predict(X_test)
cpu_pred = time.perf_counter() - t0
cpu_acc = accuracy_score(y_test, y_pred_cpu)
cpu_f1 = f1_score(y_test, y_pred_cpu)
print(f"  Fit: {cpu_fit:.4f}s  Predict: {cpu_pred:.4f}s  Acc: {cpu_acc:.4f}  F1: {cpu_f1:.4f}")

print("\n--- GPU Pipeline (@skmetal.accelerate) ---")
pipe_gpu = gpu_pipeline()
t0 = time.perf_counter()
pipe_gpu.fit(X_train, y_train)
gpu_fit = time.perf_counter() - t0
t0 = time.perf_counter()
y_pred_gpu = pipe_gpu.predict(X_test)
gpu_pred = time.perf_counter() - t0
gpu_acc = accuracy_score(y_test, y_pred_gpu)
gpu_f1 = f1_score(y_test, y_pred_gpu)
print(f"  Fit: {gpu_fit:.4f}s  Predict: {gpu_pred:.4f}s  Acc: {gpu_acc:.4f}  F1: {gpu_f1:.4f}")

print(f"\nSpeedup: fit {cpu_fit / gpu_fit:.2f}x, predict {cpu_pred / gpu_pred:.2f}x")
print(f"Metrics match: acc diff {abs(cpu_acc - gpu_acc):.6f}, F1 diff {abs(cpu_f1 - gpu_f1):.6f}")
