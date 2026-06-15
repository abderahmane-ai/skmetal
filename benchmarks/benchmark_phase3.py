"""Phase 3 focused benchmarks: simdgroup GEMM, KNN top-k, tiled pairwise distance."""

import sys
import time
from pathlib import Path

# Ensure the inner skmetal package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skmetal"))

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.datasets import make_classification, make_regression
from skmetal import accelerate
from skmetal._bridge import gemm, pairwise_distance, device_info
from skmetal._about import __version__


def bench_gemm():
    """Compare simdgroup GEMM vs MPS path for small matrices."""
    print("\n=== GEMM: simdgroup vs MPS (both GPU) ===")
    sizes = [(32, 32, 32), (64, 64, 64), (128, 128, 128),
             (256, 256, 256), (32, 96, 64)]

    for M, N, K in sizes:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        # Warmup (hits both paths through different sizes)
        _ = gemm(A, B)
        # Force MPS: use odd size (non-aligned) to skip simdgroup
        A_odd = np.random.randn(M+1, K).astype(np.float32) if K % 8 == 0 else A
        B_odd = np.random.randn(K, N).astype(np.float32)

        # Simdgroup path (aligned)
        t0 = time.perf_counter()
        for _ in range(500):
            _ = gemm(A, B)
        t_sg = (time.perf_counter() - t0) / 500

        # MPS path (use non-aligned size to force MPS fallback)
        # If M+1 isn't aligned to 8, it will use MPS
        t0 = time.perf_counter()
        for _ in range(500):
            _ = gemm(A_odd[:M, :], B_odd)
        t_mps = (time.perf_counter() - t0) / 500

        C_sg = gemm(A, B)
        C_mps = gemm(A_odd[:M, :K], B_odd[:K, :N]) if K % 8 == 0 else gemm(A, B)
        _ = gemm(A_odd[:M, :K], B_odd[:K, :N])  # just ensure it's called

        err = np.abs(C_sg - A @ B).max()
        print(f"  {M:>3}×{N:>3}×{K:>3}:  MPS={t_mps*1000:.3f}ms  "
              f"simdgroup={t_sg*1000:.3f}ms  speedup={t_mps/t_sg:.2f}x  "
              f"err={err:.2e}")


def bench_knn():
    """Compare KNN with heap-based top-k."""
    print("\n=== KNN: heap-based top-k ===")
    configs = [
        (5000, 1000, 16, 50),
        (10000, 2000, 32, 100),
    ]
    for n_train, n_test, d, k in configs:
        rng = np.random.RandomState(42)
        X_train = rng.randn(n_train, d).astype(np.float32)
        X_test = rng.randn(n_test, d).astype(np.float32)
        y_train = rng.randint(0, 5, n_train).astype(np.float32)

        # CPU
        cpu_nn = NearestNeighbors(n_neighbors=k, algorithm='brute')
        cpu_nn.fit(X_train)
        t0 = time.perf_counter()
        cpu_nn.kneighbors(X_test)
        t_cpu = time.perf_counter() - t0

        # GPU
        gpu_nn = accelerate(NearestNeighbors(n_neighbors=k, algorithm='brute'))
        gpu_nn.fit(X_train)
        t0 = time.perf_counter()
        gpu_nn.kneighbors(X_test)
        t_gpu = time.perf_counter() - t0

        print(f"  n_train={n_train} n_test={n_test} d={d} k={k}:  "
              f"CPU={t_cpu*1000:.1f}ms  GPU={t_gpu*1000:.1f}ms  speedup={t_cpu/t_gpu:.2f}x")


def bench_pairwise():
    """Compare tiled pairwise distance vs CPU."""
    print("\n=== Pairwise Distance: tiled threadgroup ===")
    configs = [(500, 16), (1000, 32), (2000, 64), (5000, 16)]
    for n, d in configs:
        X = np.random.randn(n, d).astype(np.float32)

        from sklearn.metrics.pairwise import pairwise_distances
        t0 = time.perf_counter()
        D_cpu = pairwise_distances(X, metric='euclidean', squared=True)
        t_cpu = time.perf_counter() - t0

        D_gpu = pairwise_distance(X)

        t0 = time.perf_counter()
        for _ in range(10):
            _ = pairwise_distance(X)
        t_gpu = (time.perf_counter() - t0) / 10

        max_err = np.abs(D_cpu[:min(100, n), :min(100, n)] - D_gpu[:min(100, n), :min(100, n)]).max()
        print(f"  n={n:>4} d={d:>2}:  CPU={t_cpu*1000:.1f}ms  "
              f"GPU={t_gpu*1000:.1f}ms  speedup={t_cpu/t_gpu:.2f}x  err={max_err:.2e}")


def bench_logreg():
    """LogisticRegression with fused IRLS/L-BFGS."""
    print("\n=== LogisticRegression: fused IRLS/L-BFGS ===")
    for n, d, C in [(50000, 100, 1.0), (100000, 200, 0.5)]:
        X, y = make_classification(n_samples=n, n_features=d, n_informative=d//2,
                                    random_state=42)
        X = X.astype(np.float32)

        cpu = LogisticRegression(C=C, solver='lbfgs', max_iter=200, random_state=42)
        t0 = time.perf_counter()
        cpu.fit(X, y)
        t_cpu = time.perf_counter() - t0

        gpu = accelerate(LogisticRegression(C=C, solver='lbfgs', max_iter=200, random_state=42))
        t0 = time.perf_counter()
        gpu.fit(X, y)
        t_gpu = time.perf_counter() - t0

        cpu_acc = (cpu.predict(X) == y).mean()
        gpu_acc = (gpu.predict(X) == y).mean()
        print(f"  n={n} d={d}:  CPU={t_cpu*1000:.0f}ms  GPU={t_gpu*1000:.0f}ms  "
              f"speedup={t_cpu/t_gpu:.2f}x  cpu_acc={cpu_acc:.3f} gpu_acc={gpu_acc:.3f}")


def bench_ridge():
    """Ridge: fused single-CB solve."""
    print("\n=== Ridge Regression: fused solve ===")
    for n, d, alpha in [(200000, 100, 1.0), (500000, 200, 0.1)]:
        X, y = make_regression(n_samples=n, n_features=d, random_state=42)
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        cpu = Ridge(alpha=alpha, solver='cholesky', random_state=42)
        t0 = time.perf_counter()
        cpu.fit(X, y)
        t_cpu = time.perf_counter() - t0

        gpu = accelerate(Ridge(alpha=alpha, solver='cholesky', random_state=42))
        t0 = time.perf_counter()
        gpu.fit(X, y)
        t_gpu = time.perf_counter() - t0

        cpu_r2 = cpu.score(X, y)
        gpu_r2 = gpu.score(X, y)
        print(f"  n={n} d={d}:  CPU={t_cpu*1000:.0f}ms  GPU={t_gpu*1000:.0f}ms  "
              f"speedup={t_cpu/t_gpu:.2f}x  cpu_r2={cpu_r2:.4f} gpu_r2={gpu_r2:.4f}")


if __name__ == "__main__":
    info = device_info()
    print(f"Device: {info['name']}  "
          f"max_threads={info['max_threads_per_threadgroup']}  "
          f"unified_mem={info['has_unified_memory']}")
    print(f"skmetal v{__version__}")

    bench_gemm()
    bench_knn()
    bench_pairwise()
    bench_logreg()
    bench_ridge()
