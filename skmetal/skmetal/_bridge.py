"""Python ctypes bridge to the SkMetalBridge dynamic library."""

import ctypes
import numpy as np
from pathlib import Path


def _find_library() -> str:
    search_paths = [
        Path(__file__).parent.parent / "skmetal_bridge" / ".build" / "arm64-apple-macosx" / "release" / "libSkMetalBridge.dylib",
        Path(__file__).parent.parent / "skmetal_bridge" / ".build" / "release" / "libSkMetalBridge.dylib",
        Path(__file__).parent / "libSkMetalBridge.dylib",
        Path(__file__).parent / "libSkMetalBridgeC.dylib",
        Path.home() / ".local" / "lib" / "libSkMetalBridge.dylib",
        Path.home() / ".local" / "lib" / "libSkMetalBridgeC.dylib",
    ]
    for p in search_paths:
        if p.exists():
            return str(p.resolve())
    raise RuntimeError(
        "SkMetalBridge dylib not found. Build the Swift package first:\n"
        "  cd skmetal_bridge && swift build --configuration release\n"
        "Then copy the dylib to ~/.local/lib/ or run: skmetal_bridge/build.sh"
    )

_lib = ctypes.CDLL(_find_library())

# C API signatures

_lib.skmetal_init.argtypes = []
_lib.skmetal_init.restype = ctypes.c_int

_lib.skmetal_device_info.argtypes = [
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.skmetal_device_info.restype = ctypes.c_int

_lib.skmetal_gemm.argtypes = [
    ctypes.c_void_p,  # A
    ctypes.c_void_p,  # B
    ctypes.c_void_p,  # C
    ctypes.c_size_t,  # M
    ctypes.c_size_t,  # N
    ctypes.c_size_t,  # K
    ctypes.c_float,   # alpha
    ctypes.c_float,   # beta
    ctypes.c_int,     # transpose_A
    ctypes.c_int,     # transpose_B
]
_lib.skmetal_gemm.restype = ctypes.c_int

_lib.skmetal_reduce_sum.argtypes = [
    ctypes.c_void_p,  # input
    ctypes.c_void_p,  # output
    ctypes.c_size_t,  # n
]
_lib.skmetal_reduce_sum.restype = ctypes.c_int

_lib.skmetal_reduce_mean_var.argtypes = [
    ctypes.c_void_p,  # input
    ctypes.c_void_p,  # mean_out
    ctypes.c_void_p,  # var_out
    ctypes.c_size_t,  # n
    ctypes.c_float,   # eps
]
_lib.skmetal_reduce_mean_var.restype = ctypes.c_int

_lib.skmetal_pairwise_distance.argtypes = [
    ctypes.c_void_p,  # X
    ctypes.c_void_p,  # D
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # d
]
_lib.skmetal_pairwise_distance.restype = ctypes.c_int

_lib.skmetal_kmeans_assign.argtypes = [
    ctypes.c_void_p,  # X
    ctypes.c_void_p,  # centroids
    ctypes.c_void_p,  # assignments
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # d
    ctypes.c_size_t,  # k
]
_lib.skmetal_kmeans_assign.restype = ctypes.c_int

_lib.skmetal_sigmoid.argtypes = [
    ctypes.c_void_p,  # input
    ctypes.c_void_p,  # output
    ctypes.c_size_t,  # n
]
_lib.skmetal_sigmoid.restype = ctypes.c_int

_lib.skmetal_subtract.argtypes = [
    ctypes.c_void_p,  # a
    ctypes.c_void_p,  # b
    ctypes.c_void_p,  # output
    ctypes.c_size_t,  # n
]
_lib.skmetal_subtract.restype = ctypes.c_int

_lib.skmetal_axpy.argtypes = [
    ctypes.c_void_p,  # a (in/out)
    ctypes.c_void_p,  # b
    ctypes.c_float,   # alpha
    ctypes.c_size_t,  # n
]
_lib.skmetal_axpy.restype = ctypes.c_int

_lib.skmetal_norm_sq.argtypes = [
    ctypes.c_void_p,  # input
    ctypes.c_void_p,  # output
    ctypes.c_size_t,  # n
]
_lib.skmetal_norm_sq.restype = ctypes.c_int

_lib.skmetal_center_columns.argtypes = [
    ctypes.c_void_p,  # X (in/out)
    ctypes.c_void_p,  # mean
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # d
]
_lib.skmetal_center_columns.restype = ctypes.c_int

_lib.skmetal_kmeans_combine_normalize.argtypes = [
    ctypes.c_void_p,  # partial_centroids
    ctypes.c_void_p,  # partial_counts
    ctypes.c_void_p,  # centroids (output)
    ctypes.c_size_t,  # k
    ctypes.c_size_t,  # d
    ctypes.c_size_t,  # num_groups
]
_lib.skmetal_kmeans_combine_normalize.restype = ctypes.c_int

_lib.skmetal_ridge_fit.argtypes = [
    ctypes.c_void_p,  # X (n×p, modified in-place to centered)
    ctypes.c_void_p,  # y (n)
    ctypes.c_void_p,  # XTX (out, p×p)
    ctypes.c_void_p,  # XTy (out, p)
    ctypes.c_void_p,  # X_mean_out (out, p)
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # p
]
_lib.skmetal_ridge_fit.restype = ctypes.c_int

_lib.skmetal_logreg_irls_iter.argtypes = [
    ctypes.c_void_p,  # X (n×p)
    ctypes.c_void_p,  # y (n)
    ctypes.c_void_p,  # w (p)
    ctypes.c_float,   # b (scalar bias)
    ctypes.c_void_p,  # linear (n, temp → output)
    ctypes.c_void_p,  # weight (n, temp)
    ctypes.c_void_p,  # X_scaled (n×p, temp)
    ctypes.c_void_p,  # Hessian (p×p, output)
    ctypes.c_void_p,  # gradient (p, output)
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # p
]
_lib.skmetal_logreg_irls_iter.restype = ctypes.c_int

_lib.skmetal_kmeans_batch_fused.argtypes = [
    ctypes.c_void_p,  # X (n×d)
    ctypes.c_void_p,  # centroids (k×d, in/out)
    ctypes.c_void_p,  # assignments (n, out)
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # d
    ctypes.c_size_t,  # k
    ctypes.c_size_t,  # num_groups
    ctypes.c_size_t,  # max_iter
]
_lib.skmetal_kmeans_batch_fused.restype = ctypes.c_int

_lib.skmetal_compute_mindists.argtypes = [
    ctypes.c_void_p,  # X
    ctypes.c_void_p,  # centroids
    ctypes.c_void_p,  # assignments
    ctypes.c_void_p,  # dists (out)
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # d
    ctypes.c_size_t,  # k
]
_lib.skmetal_compute_mindists.restype = ctypes.c_int

_lib.skmetal_row_norm_sq.argtypes = [
    ctypes.c_void_p,  # X
    ctypes.c_void_p,  # norms
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # d
]
_lib.skmetal_row_norm_sq.restype = ctypes.c_int

_lib.skmetal_distance_correct.argtypes = [
    ctypes.c_void_p,  # D
    ctypes.c_void_p,  # X_norm
    ctypes.c_void_p,  # C_norm
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # k
]
_lib.skmetal_distance_correct.restype = ctypes.c_int

_lib.skmetal_argmin_rows.argtypes = [
    ctypes.c_void_p,  # matrix
    ctypes.c_void_p,  # indices
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # k
]
_lib.skmetal_argmin_rows.restype = ctypes.c_int

_lib.skmetal_scaler_fit.argtypes = [
    ctypes.c_void_p,  # X
    ctypes.c_void_p,  # mean_out
    ctypes.c_void_p,  # var_out
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # d
]
_lib.skmetal_scaler_fit.restype = ctypes.c_int

_lib.skmetal_column_minmax.argtypes = [
    ctypes.c_void_p,  # X
    ctypes.c_void_p,  # min_out
    ctypes.c_void_p,  # max_out
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # d
]
_lib.skmetal_column_minmax.restype = ctypes.c_int

_lib.skmetal_irls_weight.argtypes = [
    ctypes.c_void_p,  # p
    ctypes.c_void_p,  # weights
    ctypes.c_size_t,  # n
]
_lib.skmetal_irls_weight.restype = ctypes.c_int

_lib.skmetal_scale_rows.argtypes = [
    ctypes.c_void_p,  # X
    ctypes.c_void_p,  # weights
    ctypes.c_void_p,  # output
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # d
]
_lib.skmetal_scale_rows.restype = ctypes.c_int

_lib.skmetal_knn_vote_classify.argtypes = [
    ctypes.c_void_p,  # indices
    ctypes.c_void_p,  # train_labels
    ctypes.c_void_p,  # predictions
    ctypes.c_size_t,  # N
    ctypes.c_size_t,  # k
    ctypes.c_size_t,  # n_train
]
_lib.skmetal_knn_vote_classify.restype = ctypes.c_int

_lib.skmetal_knn_vote_regress.argtypes = [
    ctypes.c_void_p,  # indices
    ctypes.c_void_p,  # train_targets
    ctypes.c_void_p,  # predictions
    ctypes.c_size_t,  # N
    ctypes.c_size_t,  # k
    ctypes.c_size_t,  # n_train
]
_lib.skmetal_knn_vote_regress.restype = ctypes.c_int

_lib.skmetal_knn_tiled_kneighbors.argtypes = [
    ctypes.c_void_p,  # X_query
    ctypes.c_void_p,  # X_train
    ctypes.c_void_p,  # out_indices
    ctypes.c_void_p,  # out_values
    ctypes.c_size_t,  # n_q
    ctypes.c_size_t,  # n_t
    ctypes.c_size_t,  # d
    ctypes.c_size_t,  # k
    ctypes.c_size_t,  # tile_size
]
_lib.skmetal_knn_tiled_kneighbors.restype = ctypes.c_int

_lib.skmetal_soft_threshold.argtypes = [
    ctypes.c_void_p,  # w (in/out)
    ctypes.c_void_p,  # w_temp
    ctypes.c_float,   # threshold
    ctypes.c_size_t,  # n
]
_lib.skmetal_soft_threshold.restype = ctypes.c_int

_lib.skmetal_fista_fit.argtypes = [
    ctypes.c_void_p,  # X
    ctypes.c_void_p,  # y
    ctypes.c_void_p,  # coef_out
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # p
    ctypes.c_float,   # alpha
    ctypes.c_float,   # l1_ratio
    ctypes.c_float,   # tol
    ctypes.c_int32,   # max_iter
    ctypes.POINTER(ctypes.c_int32),  # n_iter_out
]
_lib.skmetal_fista_fit.restype = ctypes.c_int

_lib.skmetal_column_transform.argtypes = [
    ctypes.c_void_p,  # input
    ctypes.c_void_p,  # output
    ctypes.c_void_p,  # center
    ctypes.c_void_p,  # scale
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # d
]
_lib.skmetal_column_transform.restype = ctypes.c_int

_lib.skmetal_transpose_f32.argtypes = [
    ctypes.c_void_p,  # input
    ctypes.c_void_p,  # output
    ctypes.c_size_t,  # rows
    ctypes.c_size_t,  # cols
]
_lib.skmetal_transpose_f32.restype = ctypes.c_int

_lib.skmetal_sv_init.argtypes = [
    ctypes.c_void_p,  # parent
    ctypes.c_size_t,  # n
]
_lib.skmetal_sv_init.restype = ctypes.c_int

_lib.skmetal_sv_hook.argtypes = [
    ctypes.c_void_p,  # edges
    ctypes.c_void_p,  # parent
    ctypes.c_size_t,  # edge_count
    ctypes.c_size_t,  # n
]
_lib.skmetal_sv_hook.restype = ctypes.c_int

_lib.skmetal_sv_shortcut.argtypes = [
    ctypes.c_void_p,  # parent
    ctypes.c_size_t,  # n
]
_lib.skmetal_sv_shortcut.restype = ctypes.c_int

# Initialize on import
_lib.skmetal_init()
_lib.skmetal_warmup()


# Python wrapper functions

def gemm(A: np.ndarray, B: np.ndarray, alpha=1.0, beta=0.0, trans_A=False, trans_B=False) -> np.ndarray:
    """C = alpha * op(A) @ op(B) + beta * C (zero-copy, GPU writes directly into output)."""
    if A.dtype != np.float32 or B.dtype != np.float32:
        raise TypeError("A and B must be float32")
    if not (A.flags["C_CONTIGUOUS"] and B.flags["C_CONTIGUOUS"]):
        raise ValueError("A and B must be C-contiguous")

    M, K = (A.shape[1], A.shape[0]) if trans_A else (A.shape[0], A.shape[1])
    K2, N = (B.shape[1], B.shape[0]) if trans_B else (B.shape[0], B.shape[1])
    if K != K2:
        raise ValueError(f"Incompatible dimensions: A {A.shape}, B {B.shape}")

    C = np.empty((M, N), dtype=np.float32, order="C")

    err = _lib.skmetal_gemm(
        A.ctypes.data, B.ctypes.data, C.ctypes.data,
        ctypes.c_size_t(M), ctypes.c_size_t(N), ctypes.c_size_t(K),
        ctypes.c_float(alpha), ctypes.c_float(beta),
        ctypes.c_int(int(trans_A)), ctypes.c_int(int(trans_B)),
    )
    if err != 0:
        raise RuntimeError(f"GEMM failed with code {err}")
    return C


def reduce_sum(X: np.ndarray) -> float:
    """Sum of all elements in X (zero-copy input)."""
    if X.dtype != np.float32 or not X.flags["C_CONTIGUOUS"]:
        raise TypeError("X must be float32 and C-contiguous")

    out = np.zeros((), dtype=np.float32)
    err = _lib.skmetal_reduce_sum(
        X.ctypes.data, out.ctypes.data, ctypes.c_size_t(X.size)
    )
    if err != 0:
        raise RuntimeError(f"reduce_sum failed with code {err}")
    return float(out)


def reduce_mean_var(X: np.ndarray) -> tuple[float, float]:
    """Compute mean and variance of X (Welford's algorithm, zero-copy)."""
    if X.dtype != np.float32 or not X.flags["C_CONTIGUOUS"]:
        raise TypeError("X must be float32 and C-contiguous")

    mean_out = np.zeros((), dtype=np.float32)
    var_out = np.zeros((), dtype=np.float32)
    err = _lib.skmetal_reduce_mean_var(
        X.ctypes.data, mean_out.ctypes.data, var_out.ctypes.data,
        ctypes.c_size_t(X.size), ctypes.c_float(1e-10),
    )
    if err != 0:
        raise RuntimeError(f"reduce_mean_var failed with code {err}")
    return float(mean_out), float(var_out)


def pairwise_distance(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean pairwise distance matrix (zero-copy)."""
    if X.dtype != np.float32 or not X.flags["C_CONTIGUOUS"]:
        raise TypeError("X must be float32 and C-contiguous")

    n, d = X.shape
    D = np.empty((n, n), dtype=np.float32, order="C")
    err = _lib.skmetal_pairwise_distance(
        X.ctypes.data, D.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(d),
    )
    if err != 0:
        raise RuntimeError(f"pairwise_distance failed with code {err}")
    return D


def kmeans_assign(X: np.ndarray, centroids: np.ndarray, assignments: np.ndarray,
                  n: int, d: int, k: int) -> None:
    """KMeans assignment step on GPU (zero-copy)."""
    err = _lib.skmetal_kmeans_assign(
        X.ctypes.data, centroids.ctypes.data, assignments.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(d), ctypes.c_size_t(k),
    )
    if err != 0:
        raise RuntimeError(f"kmeans_assign failed with code {err}")


def sigmoid(input: np.ndarray, output: np.ndarray) -> None:
    """GPU sigmoid: 1 / (1 + exp(-x)), in-place into output."""
    n = input.size
    err = _lib.skmetal_sigmoid(
        input.ctypes.data, output.ctypes.data, ctypes.c_size_t(n),
    )
    if err != 0:
        raise RuntimeError(f"sigmoid failed with code {err}")


def subtract(a: np.ndarray, b: np.ndarray, output: np.ndarray) -> None:
    """GPU element-wise: output = a - b."""
    n = a.size
    err = _lib.skmetal_subtract(
        a.ctypes.data, b.ctypes.data, output.ctypes.data, ctypes.c_size_t(n),
    )
    if err != 0:
        raise RuntimeError(f"subtract failed with code {err}")


def axpy(a: np.ndarray, b: np.ndarray, alpha: float) -> None:
    """GPU in-place: a += alpha * b."""
    n = a.size
    err = _lib.skmetal_axpy(
        a.ctypes.data, b.ctypes.data, ctypes.c_float(alpha), ctypes.c_size_t(n),
    )
    if err != 0:
        raise RuntimeError(f"axpy failed with code {err}")


def norm_sq(input: np.ndarray, output: np.ndarray) -> None:
    """GPU element-wise: output[i] = input[i]^2."""
    n = input.size
    err = _lib.skmetal_norm_sq(
        input.ctypes.data, output.ctypes.data, ctypes.c_size_t(n),
    )
    if err != 0:
        raise RuntimeError(f"norm_sq failed with code {err}")


def row_norm_sq(X: np.ndarray, norms: np.ndarray) -> None:
    """GPU row-wise squared norm: norms[i] = sum_j X[i][j]^2."""
    n, d = X.shape
    err = _lib.skmetal_row_norm_sq(
        X.ctypes.data, norms.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(d),
    )
    if err != 0:
        raise RuntimeError(f"row_norm_sq failed with code {err}")


def distance_correct(D: np.ndarray, X_norm: np.ndarray, C_norm: np.ndarray) -> None:
    """GPU expansion trick: D = X_norm² + C_norm² - 2*D (in-place on D)."""
    n, k = D.shape
    err = _lib.skmetal_distance_correct(
        D.ctypes.data, X_norm.ctypes.data, C_norm.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(k),
    )
    if err != 0:
        raise RuntimeError(f"distance_correct failed with code {err}")


def argmin_rows(matrix: np.ndarray, indices: np.ndarray) -> None:
    """GPU argmin per row: find column index of minimum value in each row."""
    n, k = matrix.shape
    err = _lib.skmetal_argmin_rows(
        matrix.ctypes.data, indices.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(k),
    )
    if err != 0:
        raise RuntimeError(f"argmin_rows failed with code {err}")


def scaler_fit(X: np.ndarray, mean_out: np.ndarray, var_out: np.ndarray) -> None:
    """GPU fused StandardScaler: mean and variance for ALL columns in one dispatch."""
    n, d = X.shape
    err = _lib.skmetal_scaler_fit(
        X.ctypes.data, mean_out.ctypes.data, var_out.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(d),
    )
    if err != 0:
        raise RuntimeError(f"scaler_fit failed with code {err}")


def column_minmax(X: np.ndarray, min_out: np.ndarray, max_out: np.ndarray) -> None:
    """GPU fused per-column min and max in one dispatch."""
    n, d = X.shape
    err = _lib.skmetal_column_minmax(
        X.ctypes.data, min_out.ctypes.data, max_out.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(d),
    )
    if err != 0:
        raise RuntimeError(f"column_minmax failed with code {err}")


def irls_weight(p: np.ndarray, weights: np.ndarray) -> None:
    """GPU IRLS weight: sqrt(p * (1-p)), clamped for stability."""
    n = p.size
    err = _lib.skmetal_irls_weight(
        p.ctypes.data, weights.ctypes.data, ctypes.c_size_t(n),
    )
    if err != 0:
        raise RuntimeError(f"irls_weight failed with code {err}")


def scale_rows(X: np.ndarray, weights: np.ndarray, output: np.ndarray) -> None:
    """GPU row scaling: output[i][j] = X[i][j] * weights[i]."""
    n, d = X.shape
    err = _lib.skmetal_scale_rows(
        X.ctypes.data, weights.ctypes.data, output.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(d),
    )
    if err != 0:
        raise RuntimeError(f"scale_rows failed with code {err}")


def center_columns(X: np.ndarray, mean: np.ndarray) -> None:
    """GPU: X[i][j] -= mean[j] in-place."""
    n, d = X.shape
    err = _lib.skmetal_center_columns(
        X.ctypes.data, mean.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(d),
    )
    if err != 0:
        raise RuntimeError(f"center_columns failed with code {err}")


def kmeans_combine_normalize(partial_centroids: np.ndarray, partial_counts: np.ndarray,
                              centroids: np.ndarray, k: int, d: int, num_groups: int) -> None:
    """GPU fused combine + normalize for KMeans."""
    err = _lib.skmetal_kmeans_combine_normalize(
        partial_centroids.ctypes.data, partial_counts.ctypes.data,
        centroids.ctypes.data,
        ctypes.c_size_t(k), ctypes.c_size_t(d), ctypes.c_size_t(num_groups),
    )
    if err != 0:
        raise RuntimeError(f"kmeans_combine_normalize failed with code {err}")


def logreg_irls_iter(X, y, w, b, linear, weight, X_scaled, Hessian, gradient):
    """GPU: one fused IRLS iteration (X@w + b → sigmoid → Hessian + grad) in one command buffer."""
    n, p = X.shape
    err = _lib.skmetal_logreg_irls_iter(
        X.ctypes.data, y.ctypes.data, w.ctypes.data,
        ctypes.c_float(b),
        linear.ctypes.data, weight.ctypes.data, X_scaled.ctypes.data,
        Hessian.ctypes.data, gradient.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(p),
    )
    if err != 0:
        raise RuntimeError(f"logreg_irls_iter failed with code {err}")


def kmeans_batch_fused(X, centroids, assignments,
                        n, d, k, num_groups, max_iter):
    """GPU: all KMeans iterations in one command buffer (assign + batched partial_sum + combine_normalize)."""
    err = _lib.skmetal_kmeans_batch_fused(
        X.ctypes.data, centroids.ctypes.data, assignments.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(d),
        ctypes.c_size_t(k), ctypes.c_size_t(num_groups),
        ctypes.c_size_t(max_iter),
    )
    if err != 0:
        raise RuntimeError(f"kmeans_batch_fused failed with code {err}")


def compute_mindists(X: np.ndarray, centroids: np.ndarray, assignments: np.ndarray,
                     dists: np.ndarray, n: int, d: int, k: int) -> None:
    """GPU: per-point squared distance to assigned centroid."""
    err = _lib.skmetal_compute_mindists(
        X.ctypes.data, centroids.ctypes.data, assignments.ctypes.data,
        dists.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(d), ctypes.c_size_t(k),
    )
    if err != 0:
        raise RuntimeError(f"compute_mindists failed with code {err}")


def ridge_fit(X: np.ndarray, y: np.ndarray,
               XTX: np.ndarray, XTy: np.ndarray,
               X_mean: np.ndarray) -> None:
    """GPU: center X in-place + compute X^T X + X^T y in one command buffer."""
    n, p = X.shape
    err = _lib.skmetal_ridge_fit(
        X.ctypes.data, y.ctypes.data,
        XTX.ctypes.data, XTy.ctypes.data,
        X_mean.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(p),
    )
    if err != 0:
        raise RuntimeError(f"ridge_fit failed with code {err}")


def fista_fit(X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float = 1.0,
              tol: float = 1e-4, max_iter: int = 1000) -> tuple[np.ndarray, int]:
    """GPU-resident FISTA for Lasso/ElasticNet.

    Entire FISTA loop runs on GPU. Only convergence check copies 2×p
    floats back to CPU every 10 iterations.

    Returns (coefficients, n_iter).
    """
    n, p = X.shape
    coef = np.empty(p, dtype=np.float32, order="C")
    n_iter_out = ctypes.c_int32(0)
    err = _lib.skmetal_fista_fit(
        X.ctypes.data, y.ctypes.data, coef.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(p),
        ctypes.c_float(alpha), ctypes.c_float(l1_ratio),
        ctypes.c_float(tol), ctypes.c_int32(max_iter),
        ctypes.byref(n_iter_out),
    )
    if err != 0:
        raise RuntimeError(f"fista_fit failed with code {err}")
    return coef, n_iter_out.value


def device_info() -> dict:
    """Get Metal device information."""
    name_ptr = ctypes.c_char_p()
    max_threads = ctypes.c_size_t()
    err = _lib.skmetal_device_info(
        ctypes.byref(name_ptr), ctypes.byref(max_threads)
    )
    if err != 0:
        raise RuntimeError("device_info failed")
    return {
        "name": name_ptr.value.decode("utf-8") if name_ptr.value else "unknown",
        "max_threads_per_threadgroup": max_threads.value,
    }


def knn_vote_classify(indices: np.ndarray, train_labels: np.ndarray,
                       predictions: np.ndarray, N: int, k: int, n_train: int) -> None:
    """GPU: majority-vote classification from k-nearest neighbor indices."""
    err = _lib.skmetal_knn_vote_classify(
        indices.ctypes.data, train_labels.ctypes.data, predictions.ctypes.data,
        ctypes.c_size_t(N), ctypes.c_size_t(k), ctypes.c_size_t(n_train),
    )
    if err != 0:
        raise RuntimeError(f"knn_vote_classify failed with code {err}")


def knn_vote_regress(indices: np.ndarray, train_targets: np.ndarray,
                      predictions: np.ndarray, N: int, k: int, n_train: int) -> None:
    """GPU: mean regression from k-nearest neighbor indices."""
    err = _lib.skmetal_knn_vote_regress(
        indices.ctypes.data, train_targets.ctypes.data, predictions.ctypes.data,
        ctypes.c_size_t(N), ctypes.c_size_t(k), ctypes.c_size_t(n_train),
    )
    if err != 0:
        raise RuntimeError(f"knn_vote_regress failed with code {err}")


def knn_tiled_kneighbors(X_query: np.ndarray, X_train: np.ndarray, k: int, tile_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """GPU tiled k-nearest neighbors search.

    Processes training data in tiles to avoid materializing the full N×M
    distance matrix. Each tile: GEMM → k-select → merge into global top-k.

    Returns (squared_distances, indices).
    """
    n_q, d = X_query.shape
    n_t, _ = X_train.shape
    out_indices = np.empty((n_q, k), dtype=np.int32, order="C")
    out_values = np.empty((n_q, k), dtype=np.float32, order="C")
    err = _lib.skmetal_knn_tiled_kneighbors(
        X_query.ctypes.data, X_train.ctypes.data,
        out_indices.ctypes.data, out_values.ctypes.data,
        ctypes.c_size_t(n_q), ctypes.c_size_t(n_t),
        ctypes.c_size_t(d), ctypes.c_size_t(k),
        ctypes.c_size_t(tile_size),
    )
    if err != 0:
        raise RuntimeError(f"knn_tiled_kneighbors failed with code {err}")
    return out_values, out_indices


def soft_threshold(w: np.ndarray, w_temp: np.ndarray, threshold: float) -> None:
    """GPU: soft-thresholding for Lasso (FISTA): w = sign(x) * max(|x| - t, 0)."""
    n = w.size
    err = _lib.skmetal_soft_threshold(
        w.ctypes.data, w_temp.ctypes.data, ctypes.c_float(threshold), ctypes.c_size_t(n),
    )
    if err != 0:
        raise RuntimeError(f"soft_threshold failed with code {err}")


def column_transform(input: np.ndarray, output: np.ndarray,
                      center: np.ndarray, scale: np.ndarray) -> None:
    """GPU: output[i][j] = (input[i][j] - center[j]) * scale[j]."""
    n, d = input.shape
    err = _lib.skmetal_column_transform(
        input.ctypes.data, output.ctypes.data,
        center.ctypes.data, scale.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(d),
    )
    if err != 0:
        raise RuntimeError(f"column_transform failed with code {err}")


def transpose_f32(input: np.ndarray, output: np.ndarray) -> None:
    """GPU: transpose f32 matrix (row-major ↔ column-major)."""
    if input.dtype != np.float32 or output.dtype != np.float32:
        raise TypeError("input and output must be float32")
    rows, cols = input.shape
    err = _lib.skmetal_transpose_f32(
        input.ctypes.data, output.ctypes.data,
        ctypes.c_size_t(rows), ctypes.c_size_t(cols),
    )
    if err != 0:
        raise RuntimeError(f"transpose_f32 failed with code {err}")


def sv_init(parent: np.ndarray) -> None:
    """GPU: initialize SV parent array: parent[i] = i."""
    n = parent.size
    err = _lib.skmetal_sv_init(parent.ctypes.data, ctypes.c_size_t(n))
    if err != 0:
        raise RuntimeError(f"sv_init failed with code {err}")


def sv_hook(edges: np.ndarray, parent: np.ndarray) -> None:
    """GPU: one SV hook iteration on the edge list."""
    edge_count = edges.size // 2
    n = parent.size
    err = _lib.skmetal_sv_hook(
        edges.ctypes.data, parent.ctypes.data,
        ctypes.c_size_t(edge_count), ctypes.c_size_t(n),
    )
    if err != 0:
        raise RuntimeError(f"sv_hook failed with code {err}")


def sv_shortcut(parent: np.ndarray) -> None:
    """GPU: one SV shortcut iteration: parent[i] = parent[parent[i]]."""
    n = parent.size
    err = _lib.skmetal_sv_shortcut(parent.ctypes.data, ctypes.c_size_t(n))
    if err != 0:
        raise RuntimeError(f"sv_shortcut failed with code {err}")


_lib.skmetal_warmup.argtypes = []
_lib.skmetal_warmup.restype = ctypes.c_int


def warmup():
    _lib.skmetal_warmup()


# ============================================================
# Tree-based models (GPU)
# ============================================================

_lib.skmetal_tree_predict.argtypes = [
    ctypes.c_void_p,  # X
    ctypes.c_void_p,  # tree_values
    ctypes.c_void_p,  # tree_feature
    ctypes.c_void_p,  # tree_threshold
    ctypes.c_void_p,  # tree_left
    ctypes.c_void_p,  # tree_right
    ctypes.c_void_p,  # tree_is_leaf
    ctypes.c_void_p,  # predictions (out)
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # n_features
    ctypes.c_size_t,  # n_nodes
]
_lib.skmetal_tree_predict.restype = ctypes.c_int


def tree_predict(X: np.ndarray, tree_values: np.ndarray, tree_feature: np.ndarray,
                 tree_threshold: np.ndarray, tree_left: np.ndarray, tree_right: np.ndarray,
                 tree_is_leaf: np.ndarray, predictions: np.ndarray) -> None:
    """GPU: accumulate single tree predictions into output array."""
    n, n_features = X.shape
    n_nodes = len(tree_values)
    err = _lib.skmetal_tree_predict(
        X.ctypes.data, tree_values.ctypes.data, tree_feature.ctypes.data,
        tree_threshold.ctypes.data, tree_left.ctypes.data, tree_right.ctypes.data,
        tree_is_leaf.ctypes.data, predictions.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(n_features), ctypes.c_size_t(n_nodes),
    )
    if err != 0:
        raise RuntimeError(f"tree_predict failed with code {err}")


_lib.skmetal_tree_predict_all.argtypes = [
    ctypes.c_void_p,  # X
    ctypes.c_void_p,  # all_tree_values (flat, all trees)
    ctypes.c_void_p,  # all_tree_feature
    ctypes.c_void_p,  # all_tree_threshold
    ctypes.c_void_p,  # all_tree_left
    ctypes.c_void_p,  # all_tree_right
    ctypes.c_void_p,  # all_tree_is_leaf
    ctypes.c_void_p,  # tree_offsets (per-tree start index)
    ctypes.c_void_p,  # tree_n_nodes
    ctypes.c_void_p,  # baseline (scalar f32)
    ctypes.c_void_p,  # predictions (out)
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # n_features
    ctypes.c_size_t,  # n_trees
]
_lib.skmetal_tree_predict_all.restype = ctypes.c_int


def tree_predict_all(X: np.ndarray, all_tree_values: np.ndarray, all_tree_feature: np.ndarray,
                     all_tree_threshold: np.ndarray, all_tree_left: np.ndarray,
                     all_tree_right: np.ndarray, all_tree_is_leaf: np.ndarray,
                     tree_offsets: np.ndarray, tree_n_nodes: np.ndarray,
                     baseline: np.ndarray, predictions: np.ndarray) -> None:
    """GPU: predict by traversing ALL trees in a single kernel dispatch."""
    n, n_features = X.shape
    n_trees = len(tree_offsets)
    err = _lib.skmetal_tree_predict_all(
        X.ctypes.data, all_tree_values.ctypes.data, all_tree_feature.ctypes.data,
        all_tree_threshold.ctypes.data, all_tree_left.ctypes.data,
        all_tree_right.ctypes.data, all_tree_is_leaf.ctypes.data,
        tree_offsets.ctypes.data, tree_n_nodes.ctypes.data,
        baseline.ctypes.data, predictions.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(n_features), ctypes.c_size_t(n_trees),
    )
    if err != 0:
        raise RuntimeError(f"tree_predict_all failed with code {err}")


# ============================================================
# Softmax / Multinomial helpers (GPU)
# ============================================================

_lib.skmetal_row_max.argtypes = [
    ctypes.c_void_p,  # matrix
    ctypes.c_void_p,  # max_vals (out)
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # n_cols
]
_lib.skmetal_row_max.restype = ctypes.c_int


def row_max(matrix: np.ndarray, max_vals: np.ndarray) -> None:
    """GPU: find max per row of a matrix."""
    n, n_cols = matrix.shape
    err = _lib.skmetal_row_max(
        matrix.ctypes.data, max_vals.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(n_cols),
    )
    if err != 0:
        raise RuntimeError(f"row_max failed with code {err}")


_lib.skmetal_row_sum.argtypes = [
    ctypes.c_void_p,  # matrix
    ctypes.c_void_p,  # sums (out)
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # n_cols
]
_lib.skmetal_row_sum.restype = ctypes.c_int


def row_sum(matrix: np.ndarray, sums: np.ndarray) -> None:
    """GPU: compute sum per row of a matrix."""
    n, n_cols = matrix.shape
    err = _lib.skmetal_row_sum(
        matrix.ctypes.data, sums.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(n_cols),
    )
    if err != 0:
        raise RuntimeError(f"row_sum failed with code {err}")


_lib.skmetal_softmax_exp.argtypes = [
    ctypes.c_void_p,  # matrix
    ctypes.c_void_p,  # max_vals
    ctypes.c_void_p,  # output (out)
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # n_cols
]
_lib.skmetal_softmax_exp.restype = ctypes.c_int


def softmax_exp(matrix: np.ndarray, max_vals: np.ndarray, output: np.ndarray) -> None:
    """GPU: compute exp(matrix[row][col] - max_vals[row]) for each element."""
    n, n_cols = matrix.shape
    err = _lib.skmetal_softmax_exp(
        matrix.ctypes.data, max_vals.ctypes.data, output.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(n_cols),
    )
    if err != 0:
        raise RuntimeError(f"softmax_exp failed with code {err}")


_lib.skmetal_softmax_normalize_residual.argtypes = [
    ctypes.c_void_p,  # prob (in/out)
    ctypes.c_void_p,  # row_sums
    ctypes.c_void_p,  # y (class labels)
    ctypes.c_void_p,  # residual (out)
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # n_cols
]
_lib.skmetal_softmax_normalize_residual.restype = ctypes.c_int


def softmax_normalize_residual(prob: np.ndarray, row_sums: np.ndarray,
                                y: np.ndarray, residual: np.ndarray) -> None:
    """GPU: normalize softmax probabilities and compute residual = prob - one_hot."""
    n, n_cols = prob.shape
    err = _lib.skmetal_softmax_normalize_residual(
        prob.ctypes.data, row_sums.ctypes.data, y.ctypes.data, residual.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(n_cols),
    )
    if err != 0:
        raise RuntimeError(f"softmax_normalize_residual failed with code {err}")


_lib.skmetal_negate.argtypes = [
    ctypes.c_void_p,  # a
    ctypes.c_void_p,  # output (out)
    ctypes.c_size_t,  # n
]
_lib.skmetal_negate.restype = ctypes.c_int


# ============================================================
# Multinomial IRLS iteration (GPU)
# ============================================================

_lib.skmetal_multinomial_irls_iter.argtypes = [
    ctypes.c_void_p,  # X (n×p)
    ctypes.c_void_p,  # W (p×C, in)
    ctypes.c_void_p,  # y (n, class labels as float)
    ctypes.c_void_p,  # scores (n×C, temp)
    ctypes.c_void_p,  # prob (n×C, temp)
    ctypes.c_void_p,  # max_vals (n, temp)
    ctypes.c_void_p,  # sum_exp (n, temp)
    ctypes.c_void_p,  # residual (n×C, temp)
    ctypes.c_void_p,  # gradient (p×C, out)
    ctypes.c_void_p,  # hessians (C×p×p, out)
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # p
    ctypes.c_size_t,  # C
]
_lib.skmetal_multinomial_irls_iter.restype = ctypes.c_int


def multinomial_irls_iter(X: np.ndarray, W: np.ndarray, y: np.ndarray,
                           scores: np.ndarray, prob: np.ndarray,
                           max_vals: np.ndarray, sum_exp: np.ndarray,
                           residual: np.ndarray, gradient: np.ndarray,
                           hessians: np.ndarray) -> None:
    """GPU: one fused multinomial IRLS iteration in a single command buffer.

    Computes: scores = X@W → softmax → residual → gradient + all C Hessians.
    """
    n, p = X.shape
    C = W.shape[1]
    err = _lib.skmetal_multinomial_irls_iter(
        X.ctypes.data, W.ctypes.data, y.ctypes.data,
        scores.ctypes.data, prob.ctypes.data,
        max_vals.ctypes.data, sum_exp.ctypes.data,
        residual.ctypes.data, gradient.ctypes.data,
        hessians.ctypes.data,
        ctypes.c_size_t(n), ctypes.c_size_t(p), ctypes.c_size_t(C),
    )
    if err != 0:
        raise RuntimeError(f"multinomial_irls_iter failed with code {err}")


def negate(a: np.ndarray, output: np.ndarray) -> None:
    """GPU: element-wise negation: output[i] = -a[i]."""
    n = a.size
    err = _lib.skmetal_negate(
        a.ctypes.data, output.ctypes.data, ctypes.c_size_t(n),
    )
    if err != 0:
        raise RuntimeError(f"negate failed with code {err}")
