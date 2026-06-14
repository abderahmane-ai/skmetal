"""Python ctypes bridge to the SkMetalBridge dynamic library."""
import ctypes
import platform
import sys
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Platform detection — skmetal only works on Apple Silicon macOS 14+.
# We expose METAL_AVAILABLE so callers can gate behaviour without crashing.
# ---------------------------------------------------------------------------

def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


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


def _unavailable(*args, **kwargs):
    raise RuntimeError(
        "skmetal GPU acceleration requires Apple Silicon (M1+) running macOS 14+. "
        "This device/OS is not supported. All operations fall back to scikit-learn CPU."
    )


try:
    if not _is_apple_silicon():
        raise RuntimeError("Not Apple Silicon — skipping dylib load.")
    _lib = ctypes.CDLL(_find_library())
    METAL_AVAILABLE = True
except Exception as _metal_err:  # noqa: BLE001
    import warnings
    warnings.warn(
        f"skmetal: Metal GPU not available ({_metal_err}). "
        "All estimators will run on CPU via scikit-learn.",
        stacklevel=2,
    )
    _lib = None
    METAL_AVAILABLE = False


def _bridge_call(c_func, *args):
    """Call a C bridge function with auto-conversion of Python/numpy args.

    numpy arrays → .ctypes.data  |  int → c_size_t  |  float → c_float
    bool → c_int  |  everything else passed through.
    Raises RuntimeError on non-zero return code.
    """
    c_args = []
    for a in args:
        if isinstance(a, np.ndarray):
            c_args.append(a.ctypes.data)
        elif isinstance(a, bool):
            c_args.append(ctypes.c_int(int(a)))
        elif isinstance(a, int):
            c_args.append(ctypes.c_size_t(a))
        elif isinstance(a, float):
            c_args.append(ctypes.c_float(a))
        else:
            c_args.append(a)
    err = c_func(*c_args)
    if err != 0:
        raise RuntimeError(f"{c_func.__name__} failed with code {err}")


# (c_name, argtypes...)
_BRIDGE_REGISTRY = [
    ("skmetal_reduce_sum", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t),
    ("skmetal_reduce_mean_var", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_float),
    ("skmetal_pairwise_distance", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_sigmoid", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t),
    ("skmetal_subtract", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t),
    ("skmetal_axpy", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_size_t),
    ("skmetal_norm_sq", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t),
    ("skmetal_row_norm_sq", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_distance_correct", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_argmin_rows", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_scaler_fit", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_column_minmax", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_irls_weight", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t),
    ("skmetal_scale_rows", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_center_columns", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_kmeans_assign", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_kmeans_combine_normalize", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_logreg_irls_iter", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_logreg_irls_fused", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_logreg_irls_fused_solve", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_kmeans_batch_fused", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_compute_mindists", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_gemm", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int),
    ("skmetal_ridge_fit", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_fista_fit", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)),
    ("skmetal_column_transform", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_transpose_f32", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_sv_init", ctypes.c_void_p, ctypes.c_size_t),
    ("skmetal_sv_hook", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_sv_shortcut", ctypes.c_void_p, ctypes.c_size_t),
    ("skmetal_knn_vote_classify", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_knn_vote_regress", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_knn_vote_classify_weighted", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_knn_vote_regress_weighted", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_knn_tiled_kneighbors", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int32),
    ("skmetal_soft_threshold", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_size_t),
    ("skmetal_row_max", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_row_sum", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_softmax_exp", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_softmax_normalize_residual", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_negate", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t),
    ("skmetal_multinomial_irls_iter", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_multinomial_irls_fused_solve", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_rbf_kernel_square", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_rbf_kernel_cross", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_tree_predict", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_tree_predict_all", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
]

if METAL_AVAILABLE:
    for _name, *_argtypes in _BRIDGE_REGISTRY:
        _func = getattr(_lib, _name)
        _func.argtypes = list(_argtypes)
        _func.restype = ctypes.c_int

    # device_info, init, warmup have unique argtypes not in registry
    _lib.skmetal_device_info.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]
    _lib.skmetal_device_info.restype = ctypes.c_int
    _lib.skmetal_init.argtypes = []
    _lib.skmetal_init.restype = ctypes.c_int
    _lib.skmetal_warmup.argtypes = []
    _lib.skmetal_warmup.restype = ctypes.c_int

    _lib.skmetal_init()
    _lib.skmetal_warmup()


# ============================================================
# Python wrapper functions
# ============================================================

def gemm(A: np.ndarray, B: np.ndarray, alpha=1.0, beta=0.0,
         trans_A=False, trans_B=False) -> np.ndarray:
    """C = alpha * op(A) @ op(B) + beta * C (zero-copy)."""
    if A.dtype != np.float32 or B.dtype != np.float32:
        raise TypeError("A and B must be float32")
    if not (A.flags["C_CONTIGUOUS"] and B.flags["C_CONTIGUOUS"]):
        raise ValueError("A and B must be C-contiguous")
    M, K = (A.shape[1], A.shape[0]) if trans_A else (A.shape[0], A.shape[1])
    K2, N = (B.shape[1], B.shape[0]) if trans_B else (B.shape[0], B.shape[1])
    if K != K2:
        raise ValueError(f"Incompatible dimensions: A {A.shape}, B {B.shape}")
    C = np.empty((M, N), dtype=np.float32, order="C")
    _bridge_call(_lib.skmetal_gemm, A, B, C, M, N, K,
                 ctypes.c_float(alpha), ctypes.c_float(beta),
                 ctypes.c_int(int(trans_A)), ctypes.c_int(int(trans_B)))
    return C


def reduce_sum(X: np.ndarray) -> float:
    """Sum of all elements in X."""
    if X.dtype != np.float32 or not X.flags["C_CONTIGUOUS"]:
        raise TypeError("X must be float32 and C-contiguous")
    out = np.zeros((), dtype=np.float32)
    _bridge_call(_lib.skmetal_reduce_sum, X, out, X.size)
    return float(out)


def reduce_mean_var(X: np.ndarray) -> tuple[float, float]:
    """Mean and variance via Welford's algorithm."""
    if X.dtype != np.float32 or not X.flags["C_CONTIGUOUS"]:
        raise TypeError("X must be float32 and C-contiguous")
    mean_out = np.zeros((), dtype=np.float32)
    var_out = np.zeros((), dtype=np.float32)
    _bridge_call(_lib.skmetal_reduce_mean_var, X, mean_out, var_out, X.size, 1e-10)
    return float(mean_out), float(var_out)


def pairwise_distance(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean pairwise distance matrix."""
    if X.dtype != np.float32 or not X.flags["C_CONTIGUOUS"]:
        raise TypeError("X must be float32 and C-contiguous")
    n, d = X.shape
    D = np.empty((n, n), dtype=np.float32, order="C")
    _bridge_call(_lib.skmetal_pairwise_distance, X, D, n, d)
    return D


def kmeans_assign(X: np.ndarray, centroids: np.ndarray, assignments: np.ndarray,
                  n: int, d: int, k: int) -> None:
    """KMeans assignment step."""
    _bridge_call(_lib.skmetal_kmeans_assign, X, centroids, assignments, n, d, k)


def sigmoid(input: np.ndarray, output: np.ndarray) -> None:
    """GPU sigmoid: 1 / (1 + exp(-x))."""
    _bridge_call(_lib.skmetal_sigmoid, input, output, input.size)


def subtract(a: np.ndarray, b: np.ndarray, output: np.ndarray) -> None:
    """Element-wise: output = a - b."""
    _bridge_call(_lib.skmetal_subtract, a, b, output, a.size)


def axpy(a: np.ndarray, b: np.ndarray, alpha: float) -> None:
    """In-place: a += alpha * b."""
    _bridge_call(_lib.skmetal_axpy, a, b, alpha, a.size)


def norm_sq(input: np.ndarray, output: np.ndarray) -> None:
    """Element-wise: output[i] = input[i]^2."""
    _bridge_call(_lib.skmetal_norm_sq, input, output, input.size)


def row_norm_sq(X: np.ndarray, norms: np.ndarray) -> None:
    """Row-wise squared norm: norms[i] = sum_j X[i][j]^2."""
    _bridge_call(_lib.skmetal_row_norm_sq, X, norms, X.shape[0], X.shape[1])


def distance_correct(D: np.ndarray, X_norm: np.ndarray, C_norm: np.ndarray) -> None:
    """Expansion trick: D = X_norm² + C_norm² - 2*D (in-place)."""
    _bridge_call(_lib.skmetal_distance_correct, D, X_norm, C_norm, D.shape[0], D.shape[1])


def argmin_rows(matrix: np.ndarray, indices: np.ndarray) -> None:
    """Argmin per row (column index of minimum)."""
    _bridge_call(_lib.skmetal_argmin_rows, matrix, indices, matrix.shape[0], matrix.shape[1])


def scaler_fit(X: np.ndarray, mean_out: np.ndarray, var_out: np.ndarray) -> None:
    """Fused StandardScaler: mean + variance for all columns."""
    _bridge_call(_lib.skmetal_scaler_fit, X, mean_out, var_out, X.shape[0], X.shape[1])


def column_minmax(X: np.ndarray, min_out: np.ndarray, max_out: np.ndarray) -> None:
    """Fused per-column min + max."""
    _bridge_call(_lib.skmetal_column_minmax, X, min_out, max_out, X.shape[0], X.shape[1])


def irls_weight(p: np.ndarray, weights: np.ndarray) -> None:
    """IRLS weight: sqrt(p * (1-p)), clamped."""
    _bridge_call(_lib.skmetal_irls_weight, p, weights, p.size)


def scale_rows(X: np.ndarray, weights: np.ndarray, output: np.ndarray) -> None:
    """Row scaling: output[i][j] = X[i][j] * weights[i]."""
    _bridge_call(_lib.skmetal_scale_rows, X, weights, output, X.shape[0], X.shape[1])


def center_columns(X: np.ndarray, mean: np.ndarray) -> None:
    """In-place: X[i][j] -= mean[j]."""
    _bridge_call(_lib.skmetal_center_columns, X, mean, X.shape[0], X.shape[1])


def kmeans_combine_normalize(partial_centroids: np.ndarray, partial_counts: np.ndarray,
                              centroids: np.ndarray, k: int, d: int, num_groups: int) -> None:
    """Fused combine + normalize for KMeans."""
    _bridge_call(_lib.skmetal_kmeans_combine_normalize,
                 partial_centroids, partial_counts, centroids, k, d, num_groups)


def logreg_irls_iter(X, y, w, b, linear, weight, X_scaled, Hessian, gradient):
    """One IRLS iteration (X@w+b → sigmoid → Hessian + grad), 8 dispatches."""
    _bridge_call(_lib.skmetal_logreg_irls_iter,
                 X, y, w, b, linear, weight, X_scaled, Hessian, gradient,
                 X.shape[0], X.shape[1])


def logreg_irls_fused(X, y, w, b, linear, weight, X_scaled, Hessian, gradient):
    """Fused IRLS iteration (5 dispatches, was 8) — no solve."""
    _bridge_call(_lib.skmetal_logreg_irls_fused,
                 X, y, w, b, linear, weight, X_scaled, Hessian, gradient,
                 X.shape[0], X.shape[1])


def logreg_irls_fused_solve(X, y, w, b, linear, weight, X_scaled, Hessian, gradient, delta, alpha):
    """Fused IRLS + L2 + Cholesky solve."""
    _bridge_call(_lib.skmetal_logreg_irls_fused_solve,
                 X, y, w, b, linear, weight, X_scaled, Hessian, gradient, delta, alpha,
                 X.shape[0], X.shape[1])


def kmeans_batch_fused(X, centroids, assignments,
                        n, d, k, num_groups, max_iter):
    """All KMeans iterations in one command buffer."""
    _bridge_call(_lib.skmetal_kmeans_batch_fused,
                 X, centroids, assignments, n, d, k, num_groups, max_iter)


def compute_mindists(X: np.ndarray, centroids: np.ndarray, assignments: np.ndarray,
                      dists: np.ndarray, n: int, d: int, k: int) -> None:
    """Per-point squared distance to assigned centroid."""
    _bridge_call(_lib.skmetal_compute_mindists,
                 X, centroids, assignments, dists, n, d, k)


def ridge_fit(X: np.ndarray, y: np.ndarray,
               XTX: np.ndarray, XTy: np.ndarray,
               X_mean: np.ndarray) -> None:
    """Center X in-place + compute X^T X + X^T y in one command buffer.

    .. warning::
       ``X`` is **modified in-place**: its columns are mean-centered.
    """
    _bridge_call(_lib.skmetal_ridge_fit, X, y, XTX, XTy, X_mean, X.shape[0], X.shape[1])


def fista_fit(X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float = 1.0,
              tol: float = 1e-4, max_iter: int = 1000) -> tuple[np.ndarray, int]:
    """GPU-resident FISTA for Lasso/ElasticNet."""
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
    if not METAL_AVAILABLE:
        raise RuntimeError(
            "skmetal: Metal is not available on this device. "
            "device_info() requires Apple Silicon + macOS 14+."
        )
    name_ptr = ctypes.c_char_p()
    max_threads = ctypes.c_size_t()
    err = _lib.skmetal_device_info(ctypes.byref(name_ptr), ctypes.byref(max_threads))
    if err != 0:
        raise RuntimeError("device_info failed")
    return {
        "name": name_ptr.value.decode("utf-8") if name_ptr.value else "unknown",
        "max_threads_per_threadgroup": max_threads.value,
    }



def knn_vote_classify(indices: np.ndarray, train_labels: np.ndarray,
                       predictions: np.ndarray, N: int, k: int, n_train: int) -> None:
    """Majority-vote classification from k-NN indices."""
    _bridge_call(_lib.skmetal_knn_vote_classify,
                 indices, train_labels, predictions, N, k, n_train)


def knn_vote_regress(indices: np.ndarray, train_targets: np.ndarray,
                      predictions: np.ndarray, N: int, k: int, n_train: int) -> None:
    """Mean regression from k-NN indices."""
    _bridge_call(_lib.skmetal_knn_vote_regress,
                 indices, train_targets, predictions, N, k, n_train)


def knn_vote_classify_weighted(indices: np.ndarray, distances: np.ndarray,
                                train_labels: np.ndarray,
                                predictions: np.ndarray,
                                N: int, k: int, n_train: int) -> None:
    """Weighted majority-vote classification.  weight = 1 / (distance + eps)."""
    _bridge_call(_lib.skmetal_knn_vote_classify_weighted,
                 indices, distances, train_labels, predictions, N, k, n_train)


def knn_vote_regress_weighted(indices: np.ndarray, distances: np.ndarray,
                               train_targets: np.ndarray,
                               predictions: np.ndarray,
                               N: int, k: int, n_train: int) -> None:
    """Weighted mean regression.  weight = 1 / (distance + eps)."""
    _bridge_call(_lib.skmetal_knn_vote_regress_weighted,
                 indices, distances, train_targets, predictions, N, k, n_train)


def knn_tiled_kneighbors(X_query: np.ndarray, X_train: np.ndarray, k: int,
                          tile_size: int = 4096,
                          metric: str = "euclidean") -> tuple[np.ndarray, np.ndarray]:
    """GPU tiled k-nearest neighbors search.

    Supports euclidean (squared L2), manhattan (L1), cosine (1 - cos).
    """
    metric_map = {"euclidean": 0, "manhattan": 1, "cosine": 2}
    mcode = metric_map.get(metric, 0)
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
        ctypes.c_int32(mcode),
    )
    if err != 0:
        raise RuntimeError(f"knn_tiled_kneighbors failed with code {err}")
    return out_values, out_indices


def soft_threshold(w: np.ndarray, w_temp: np.ndarray, threshold: float) -> None:
    """Soft-thresholding: w[i] = sign(x) * max(|x| - t, 0)."""
    _bridge_call(_lib.skmetal_soft_threshold, w, w_temp, threshold, w.size)


def column_transform(input: np.ndarray, output: np.ndarray,
                      center: np.ndarray, scale: np.ndarray) -> None:
    """output[i][j] = (input[i][j] - center[j]) * scale[j]."""
    _bridge_call(_lib.skmetal_column_transform,
                 input, output, center, scale, input.shape[0], input.shape[1])


def transpose_f32(input: np.ndarray, output: np.ndarray) -> None:
    """Transpose f32 matrix (row-major ↔ column-major)."""
    if input.dtype != np.float32 or output.dtype != np.float32:
        raise TypeError("input and output must be float32")
    _bridge_call(_lib.skmetal_transpose_f32, input, output, input.shape[0], input.shape[1])


def sv_init(parent: np.ndarray) -> None:
    """SV parent init: parent[i] = i."""
    _bridge_call(_lib.skmetal_sv_init, parent, parent.size)


def sv_hook(edges: np.ndarray, parent: np.ndarray) -> None:
    """One SV hook iteration on the edge list."""
    _bridge_call(_lib.skmetal_sv_hook, edges, parent, edges.size // 2, parent.size)


def sv_shortcut(parent: np.ndarray) -> None:
    """SV shortcut: parent[i] = parent[parent[i]]."""
    _bridge_call(_lib.skmetal_sv_shortcut, parent, parent.size)


def warmup():
    """Pre-compile all Metal kernels (called on import)."""
    _lib.skmetal_warmup()


def tree_predict(X: np.ndarray, tree_values: np.ndarray, tree_feature: np.ndarray,
                 tree_threshold: np.ndarray, tree_left: np.ndarray, tree_right: np.ndarray,
                 tree_is_leaf: np.ndarray, predictions: np.ndarray) -> None:
    """Accumulate single tree predictions into output array."""
    _bridge_call(_lib.skmetal_tree_predict,
                 X, tree_values, tree_feature, tree_threshold, tree_left, tree_right,
                 tree_is_leaf, predictions,
                 X.shape[0], X.shape[1], len(tree_values))


def tree_predict_all(X: np.ndarray, all_tree_values: np.ndarray, all_tree_feature: np.ndarray,
                      all_tree_threshold: np.ndarray, all_tree_left: np.ndarray,
                      all_tree_right: np.ndarray, all_tree_is_leaf: np.ndarray,
                      tree_offsets: np.ndarray, tree_n_nodes: np.ndarray,
                      baseline: np.ndarray, predictions: np.ndarray) -> None:
    """Predict by traversing ALL trees in a single kernel dispatch."""
    _bridge_call(_lib.skmetal_tree_predict_all,
                 X, all_tree_values, all_tree_feature, all_tree_threshold,
                 all_tree_left, all_tree_right, all_tree_is_leaf,
                 tree_offsets, tree_n_nodes, baseline, predictions,
                 X.shape[0], X.shape[1], len(tree_offsets))


def row_max(matrix: np.ndarray, max_vals: np.ndarray) -> None:
    """Max per row of a matrix."""
    _bridge_call(_lib.skmetal_row_max, matrix, max_vals, matrix.shape[0], matrix.shape[1])


def row_sum(matrix: np.ndarray, sums: np.ndarray) -> None:
    """Sum per row of a matrix."""
    _bridge_call(_lib.skmetal_row_sum, matrix, sums, matrix.shape[0], matrix.shape[1])


def softmax_exp(matrix: np.ndarray, max_vals: np.ndarray, output: np.ndarray) -> None:
    """exp(matrix[row][col] - max_vals[row]) for each element."""
    _bridge_call(_lib.skmetal_softmax_exp,
                 matrix, max_vals, output, matrix.shape[0], matrix.shape[1])


def softmax_normalize_residual(prob: np.ndarray, row_sums: np.ndarray,
                                y: np.ndarray, residual: np.ndarray) -> None:
    """Normalize softmax probs and compute residual = prob - one_hot."""
    _bridge_call(_lib.skmetal_softmax_normalize_residual,
                 prob, row_sums, y, residual, prob.shape[0], prob.shape[1])


def negate(a: np.ndarray, output: np.ndarray) -> None:
    """Element-wise negation: output[i] = -a[i]."""
    _bridge_call(_lib.skmetal_negate, a, output, a.size)


def multinomial_irls_iter(X: np.ndarray, W: np.ndarray, y: np.ndarray,
                           scores: np.ndarray, prob: np.ndarray,
                           max_vals: np.ndarray, sum_exp: np.ndarray,
                           residual: np.ndarray, gradient: np.ndarray,
                           hessians: np.ndarray) -> None:
    """One fused multinomial IRLS iteration in a single command buffer."""
    _bridge_call(_lib.skmetal_multinomial_irls_iter,
                 X, W, y, scores, prob, max_vals, sum_exp, residual, gradient, hessians,
                 X.shape[0], X.shape[1], W.shape[1])


def multinomial_irls_fused_solve(X: np.ndarray, W: np.ndarray, y: np.ndarray,
                                  scores: np.ndarray, prob: np.ndarray,
                                  max_vals: np.ndarray, sum_exp: np.ndarray,
                                  residual: np.ndarray, gradient: np.ndarray,
                                  hessians: np.ndarray, delta_W: np.ndarray,
                                  alpha: float) -> None:
    """Fused multinomial IRLS iteration + L2 + batched Cholesky solve."""
    _bridge_call(_lib.skmetal_multinomial_irls_fused_solve,
                 X, W, y, scores, prob, max_vals, sum_exp, residual, gradient,
                 hessians, delta_W, alpha,
                 X.shape[0], X.shape[1], int(W.shape[1]))


def rbf_kernel_square(X: np.ndarray, X_norm: np.ndarray,
                       K_out: np.ndarray, gamma: float) -> None:
    """Compute RBF kernel matrix: K[i][j] = exp(-gamma * ||X[i] - X[j]||^2)."""
    _bridge_call(_lib.skmetal_rbf_kernel_square,
                 X, X_norm, K_out, gamma,
                 X.shape[0], X.shape[1])


def rbf_kernel_cross(X1: np.ndarray, X1_norm: np.ndarray,
                      X2: np.ndarray, X2_norm: np.ndarray,
                      K_out: np.ndarray, gamma: float) -> None:
    """Cross RBF kernel: K[i][j] = exp(-gamma * ||X1[i] - X2[j]||^2)."""
    _bridge_call(_lib.skmetal_rbf_kernel_cross,
                 X1, X1_norm, X2, X2_norm, K_out, gamma,
                 X1.shape[0], X2.shape[0], X1.shape[1])
