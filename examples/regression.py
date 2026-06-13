"""California housing: CPU vs GPU regression pipeline."""
import time
import warnings
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import skmetal

warnings.filterwarnings("ignore")

X, y = fetch_california_housing(return_X_y=True)
X = X.astype(np.float32)
y = y.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

@skmetal.accelerate
def gpu_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=6, random_state=42)),
        ("ridge", Ridge(alpha=1.0)),
    ])

pipe_cpu = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=6, random_state=42)),
    ("ridge", Ridge(alpha=1.0)),
])

print("--- CPU Pipeline ---")
t0 = time.perf_counter()
pipe_cpu.fit(X_train, y_train)
cpu_fit = time.perf_counter() - t0
t0 = time.perf_counter()
y_pred_cpu = pipe_cpu.predict(X_test)
cpu_pred = time.perf_counter() - t0
cpu_mse = mean_squared_error(y_test, y_pred_cpu)
cpu_r2 = r2_score(y_test, y_pred_cpu)
print(f"  Fit: {cpu_fit:.4f}s  Predict: {cpu_pred:.4f}s  RMSE: {np.sqrt(cpu_mse):.4f}  R2: {cpu_r2:.4f}")

print("\n--- GPU Pipeline (@skmetal.accelerate) ---")
pipe_gpu = gpu_pipeline()
t0 = time.perf_counter()
pipe_gpu.fit(X_train, y_train)
gpu_fit = time.perf_counter() - t0
t0 = time.perf_counter()
y_pred_gpu = pipe_gpu.predict(X_test)
gpu_pred = time.perf_counter() - t0
gpu_mse = mean_squared_error(y_test, y_pred_gpu)
gpu_r2 = r2_score(y_test, y_pred_gpu)
print(f"  Fit: {gpu_fit:.4f}s  Predict: {gpu_pred:.4f}s  RMSE: {np.sqrt(gpu_mse):.4f}  R2: {gpu_r2:.4f}")

print(f"\nSpeedup: fit {cpu_fit / gpu_fit:.2f}x, predict {cpu_pred / gpu_pred:.2f}x")
print(f"Results match: R2 diff {abs(cpu_r2 - gpu_r2):.6f}")
