"""Python ctypes bridge to the SkMetalBridge dynamic library."""
import ctypes
import platform
import sys
import warnings
import numpy as np
from pathlib import Path

from ._config import get_config

from .estimators._mlx_registry import has_mlx

_HAS_MLX = has_mlx()

if _HAS_MLX:
    import mlx.core as mx

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
        "Then copy the dylib to ~/.local/lib/ or run: cd skmetal_bridge && bash compile_metal.sh && swift build --configuration release"
    )


try:
    if not _is_apple_silicon():
        raise RuntimeError("Not Apple Silicon — skipping dylib load.")
    _lib = ctypes.CDLL(_find_library())
    METAL_AVAILABLE = True
except Exception as _metal_err:  # noqa: BLE001
    warnings.warn(
        f"skmetal: Metal GPU not available ({_metal_err}). "
        "All estimators will run on CPU via scikit-learn.",
        stacklevel=2,
    )
    _lib = None
    METAL_AVAILABLE = False


def _bridge_call(c_func, *args, context=""):
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
        elif isinstance(a, (float, np.floating)):
            c_args.append(ctypes.c_float(a))
        else:
            c_args.append(a)
    err = c_func(*c_args)
    if err != 0:
        msg = f"{c_func.__name__} failed with code {err}"
        if context:
            msg += f" ({context})"
        raise RuntimeError(msg)


# (c_name, argtypes...)
_BRIDGE_REGISTRY = [
    ("skmetal_pairwise_distance", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_scaler_fit", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_column_minmax", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_kmeans_assign", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_kmeans_batch_fused", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, ctypes.POINTER(ctypes.c_int32)),
    ("skmetal_compute_mindists", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_gemm", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int),
    ("skmetal_fista_fit", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)),
    ("skmetal_column_transform", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_sv_init", ctypes.c_void_p, ctypes.c_size_t),
    ("skmetal_sv_hook", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_sv_shortcut", ctypes.c_void_p, ctypes.c_size_t),
    ("skmetal_knn_vote_classify", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_knn_vote_regress", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_knn_vote_classify_weighted", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_knn_vote_regress_weighted", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_knn_tiled_kneighbors", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int32),
    ("skmetal_rbf_kernel_square", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_rbf_kernel_cross", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_tree_predict_all", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_svc_predict_binary", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float),
    ("skmetal_logreg_irls_fit", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32)),
    ("skmetal_logreg_lbfgs_fit", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32)),
    ("skmetal_ridge_fit_solve", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_size_t, ctypes.c_size_t),
    ("skmetal_multinomial_lbfgs_fit", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_int32, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32)),
    ("skmetal_minmax_transform", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, ctypes.c_float),
    ("skmetal_convert_f32_to_f16", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t),
    ("skmetal_convert_f16_to_f32", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t),
    ("skmetal_gemm_f16", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t),
]

if METAL_AVAILABLE:
    for _name, *_argtypes in _BRIDGE_REGISTRY:
        _func = getattr(_lib, _name)
        _func.argtypes = list(_argtypes)
        _func.restype = ctypes.c_int

    # device_info, init, warmup have unique argtypes not in registry
    _lib.skmetal_device_info.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t),
                                         ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint64)]
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

# ---------------------------------------------------------------------------
# Float16 conversion and GEMM (half-precision throughput on Apple GPU)
# ---------------------------------------------------------------------------

def _check_contiguous(x: np.ndarray) -> None:
    if not x.flags["C_CONTIGUOUS"]:
        raise ValueError("array must be C-contiguous")


def convert_f32_to_f16(x: np.ndarray) -> np.ndarray:
    """Convert float32 array to float16 on GPU (zero-copy)."""
    if x.dtype != np.float32:
        raise TypeError("x must be float32")
    _check_contiguous(x)
    out = np.empty(x.size, dtype=np.float16, order="C")
    _bridge_call(_lib.skmetal_convert_f32_to_f16, x, out, x.size)
    return out.reshape(x.shape)


def convert_f16_to_f32(x: np.ndarray) -> np.ndarray:
    """Convert float16 array to float32 on GPU (zero-copy)."""
    if x.dtype != np.float16:
        raise TypeError("x must be float16")
    _check_contiguous(x)
    out = np.empty(x.size, dtype=np.float32, order="C")
    _bridge_call(_lib.skmetal_convert_f16_to_f32, x, out, x.size)
    return out.reshape(x.shape)


def gemm_f16(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """C = A @ B using float16 simdgroup GEMM (half in/out, zero-copy).

    Only supports M,N,K <= 256, all aligned to 8, no transpose.
    Unsupported dimensions return RuntimeError from the Swift bridge.
    """
    if A.dtype != np.float16 or B.dtype != np.float16:
        raise TypeError("A and B must be float16")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Incompatible dimensions: A {A.shape}, B {B.shape}")
    C = np.empty((M, N), dtype=np.float16, order="C")
    _bridge_call(_lib.skmetal_gemm_f16, A, B, C, M, N, K)
    return C


# Metal simdgroup GEMM constraints (Apple GPU hardware limits)
_F16_SIMD_MAX_DIM = 256     # max dimension per simdgroup matmul
_F16_SIMD_ALIGN = 8         # required byte/simd alignment
_F16_SIMD_MIN_DIM = 128     # min dim to amortize f32→f16 conversion


def _should_use_f16(A: np.ndarray, B: np.ndarray) -> bool:
    """Check if f16 GEMM would be beneficial.

    Conditions: dims aligned to 8, within simdgroup limits (<=256),
    and large enough that conversion overhead is worth it (>= 128 per dim).
    """
    if not (A.dtype == np.float32 and B.dtype == np.float32):
        return False
    M, K1 = A.shape
    K2, N = B.shape
    if K1 != K2:
        return False
    K = K1
    return (M <= _F16_SIMD_MAX_DIM and N <= _F16_SIMD_MAX_DIM and K <= _F16_SIMD_MAX_DIM
            and M % _F16_SIMD_ALIGN == 0 and N % _F16_SIMD_ALIGN == 0 and K % _F16_SIMD_ALIGN == 0
            and M >= _F16_SIMD_MIN_DIM and N >= _F16_SIMD_MIN_DIM and K >= _F16_SIMD_MIN_DIM)


# ---------------------------------------------------------------------------
# MLX-native GEMM — avoids ctypes/dylib overhead when MLX is available
# ---------------------------------------------------------------------------

def _use_mlx_gemm(A: np.ndarray, B: np.ndarray, trans_A: bool = False,
                  trans_B: bool = False) -> bool:
    """Check if MLX GEMM should be used instead of Metal bridge.

    MLX matmul is preferred when available because it avoids ctypes data
    pointer passing and the f32→f16→f32 conversion round-trip.
    """
    if not _HAS_MLX:
        return False
    M, K = (A.shape[1], A.shape[0]) if trans_A else (A.shape[0], A.shape[1])
    K2, N = (B.shape[1], B.shape[0]) if trans_B else (B.shape[0], B.shape[1])
    # MLX matmul amortization: skip for trivially small ops
    return M * N * K >= 4096


def _gemm_mlx(A: np.ndarray, B: np.ndarray, alpha=1.0, beta=0.0,
              trans_A=False, trans_B=False) -> np.ndarray:
    """MLX-native GEMM — zero dylib overhead, zero ctypes conversion."""
    A_mx = mx.array(A if not trans_A else A.T, dtype=mx.float32)
    B_mx = mx.array(B if not trans_B else B.T, dtype=mx.float32)
    result = alpha * (A_mx @ B_mx)
    if beta != 0.0:
        result = result + beta * mx.zeros_like(result)
    return np.array(result)


def _should_use_f16_mlx(A: np.ndarray, B: np.ndarray) -> bool:
    """Check if MLX native float16 GEMM is beneficial."""
    if not _HAS_MLX:
        return False
    if not (A.dtype == np.float32 and B.dtype == np.float32):
        return False
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        return False
    return M >= 128 and N >= 128 and K >= 128


def _gemm_f16_mlx(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """MLX-native float16 GEMM — skips Metal bridge f16 converter entirely."""
    A_mx = mx.array(A, dtype=mx.float16)
    B_mx = mx.array(B, dtype=mx.float16)
    C_mx = A_mx @ B_mx
    return np.array(C_mx, dtype=np.float32)


def gemm(A: np.ndarray, B: np.ndarray, alpha=1.0, beta=0.0,
         trans_A=False, trans_B=False) -> np.ndarray:
    """C = alpha * op(A) @ op(B) + beta * C (zero-copy).

    Routing priority:
    1. MLX native GEMM (when MLX available + problem large enough)
    2. Metal bridge float16 simdgroup GEMM (small aligned matrices)
    3. Metal bridge float32 MPS GEMM (fallback)
    """
    if A.dtype != np.float32 or B.dtype != np.float32:
        raise TypeError("A and B must be float32")
    if not (A.flags["C_CONTIGUOUS"] and B.flags["C_CONTIGUOUS"]):
        raise ValueError("A and B must be C-contiguous")
    M, K = (A.shape[1], A.shape[0]) if trans_A else (A.shape[0], A.shape[1])
    K2, N = (B.shape[1], B.shape[0]) if trans_B else (B.shape[0], B.shape[1])
    if K != K2:
        raise ValueError(f"Incompatible dimensions: A {A.shape}, B {B.shape}")

    # Priority 1: MLX-native float16 GEMM (2× throughput, native mx.float16)
    if not trans_A and not trans_B and alpha == 1.0 and beta == 0.0:
        if _should_use_f16_mlx(A, B):
            try:
                return _gemm_f16_mlx(A, B)
            except Exception:
                if get_config().verbose:
                    warnings.warn("MLX f16 GEMM failed, falling back", stacklevel=2)

    # Priority 2: MLX-native float32 GEMM (skips ctypes/dylib overhead)
    if not trans_A and not trans_B and alpha == 1.0 and beta == 0.0:
        if _use_mlx_gemm(A, B, trans_A, trans_B):
            try:
                return _gemm_mlx(A, B, alpha, beta, trans_A, trans_B)
            except Exception:
                if get_config().verbose:
                    warnings.warn("MLX f32 GEMM failed, falling back", stacklevel=2)

    # Priority 3: Metal bridge float16 simdgroup GEMM
    if not trans_A and not trans_B and alpha == 1.0 and beta == 0.0:
        if _should_use_f16(A, B):
            try:
                A_f16 = convert_f32_to_f16(A)
                B_f16 = convert_f32_to_f16(B)
                C_f16 = gemm_f16(A_f16, B_f16)
                return convert_f16_to_f32(C_f16)
            except (RuntimeError, TypeError):
                if get_config().verbose:
                    warnings.warn("Metal f16 simdgroup GEMM failed, falling back", stacklevel=2)

    # Priority 4: Metal bridge float32 MPS GEMM
    if not METAL_AVAILABLE:
        raise RuntimeError("skmetal: Metal not available — gemm() requires Apple Silicon.")
    C = np.empty((M, N), dtype=np.float32, order="C")
    _bridge_call(_lib.skmetal_gemm, A, B, C, M, N, K,
                 ctypes.c_float(alpha), ctypes.c_float(beta),
                 ctypes.c_int(int(trans_A)), ctypes.c_int(int(trans_B)))
    return C


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


def scaler_fit(X: np.ndarray, mean_out: np.ndarray, var_out: np.ndarray) -> None:
    """Fused StandardScaler: mean + variance for all columns."""
    _bridge_call(_lib.skmetal_scaler_fit, X, mean_out, var_out, X.shape[0], X.shape[1])


def column_minmax(X: np.ndarray, min_out: np.ndarray, max_out: np.ndarray) -> None:
    """Fused per-column min + max."""
    _bridge_call(_lib.skmetal_column_minmax, X, min_out, max_out, X.shape[0], X.shape[1])



def kmeans_batch_fused(X, centroids, assignments,
                        n, d, k, num_groups, max_iter, tol):
    """KMeans iterations with GPU-side convergence detection."""
    n_iter_out = ctypes.c_int32(0)
    _bridge_call(_lib.skmetal_kmeans_batch_fused,
                 X, centroids, assignments, n, d, k, num_groups, max_iter,
                 ctypes.c_float(tol), ctypes.byref(n_iter_out))
    return n_iter_out.value


def compute_mindists(X: np.ndarray, centroids: np.ndarray, assignments: np.ndarray,
                      dists: np.ndarray, n: int, d: int, k: int) -> None:
    """Per-point squared distance to assigned centroid."""
    _bridge_call(_lib.skmetal_compute_mindists,
                 X, centroids, assignments, dists, n, d, k)


def kmeans_inertia(X: np.ndarray, centroids: np.ndarray,
                    assignments: np.ndarray, n: int, d: int, k: int) -> float:
    """Total squared distance Σ‖X[i] - centroids[assignments[i]]‖² (GPU)."""
    _lib.skmetal_kmeans_inertia.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    _lib.skmetal_kmeans_inertia.restype = ctypes.c_float
    return _lib.skmetal_kmeans_inertia(
        X.ctypes.data, centroids.ctypes.data, assignments.ctypes.data, n, d, k)


def ridge_fit_solve(X: np.ndarray, y: np.ndarray,
                     X_mean: np.ndarray, coef: np.ndarray,
                     alpha: float) -> None:
    """Fused Ridge: center X + XTX + XTy + L2 + Cholesky solve (one GPU dispatch)."""
    _bridge_call(_lib.skmetal_ridge_fit_solve, X, y, X_mean, coef, alpha,
                 X.shape[0], X.shape[1])


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
        raise RuntimeError(f"fista_fit failed with code {err} (n={n}, p={p}, alpha={alpha}, l1_ratio={l1_ratio})")
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
    unified = ctypes.c_uint8()
    working_set = ctypes.c_uint64()
    err = _lib.skmetal_device_info(ctypes.byref(name_ptr), ctypes.byref(max_threads),
                                    ctypes.byref(unified), ctypes.byref(working_set))
    if err != 0:
        raise RuntimeError("skmetal: device_info() failed (Metal driver may be in an invalid state)")
    return {
        "name": name_ptr.value.decode("utf-8") if name_ptr.value else "unknown",
        "max_threads_per_threadgroup": max_threads.value,
        "has_unified_memory": bool(unified.value),
        "recommended_working_set_size_bytes": working_set.value,
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
        raise RuntimeError(f"knn_tiled_kneighbors failed with code {err} (n_q={n_q}, n_t={n_t}, d={d}, k={k}, metric={metric})")
    return out_values, out_indices


def minmax_transform(X: np.ndarray, X_out: np.ndarray,
                      min_vals: np.ndarray, max_vals: np.ndarray,
                      feature_min: float = 0.0, feature_max: float = 1.0) -> None:
    """GPU min-max normalization: X_scaled = (X - min) / (max - min) * (fmax - fmin) + fmin."""
    if X.dtype != np.float32 or not X.flags["C_CONTIGUOUS"]:
        raise TypeError("X must be float32 and C-contiguous")
    if X_out.dtype != np.float32 or not X_out.flags["C_CONTIGUOUS"]:
        raise TypeError("X_out must be float32 and C-contiguous")
    _bridge_call(_lib.skmetal_minmax_transform,
                 X, X_out, min_vals, max_vals,
                 X.shape[0], X.shape[1],
                 ctypes.c_float(feature_min), ctypes.c_float(feature_max))


def column_transform(input: np.ndarray, output: np.ndarray,
                      center: np.ndarray, scale: np.ndarray) -> None:
    """output[i][j] = (input[i][j] - center[j]) * scale[j]."""
    _bridge_call(_lib.skmetal_column_transform,
                 input, output, center, scale, input.shape[0], input.shape[1])


def sv_init(parent: np.ndarray) -> None:
    """SV parent init: parent[i] = i."""
    _bridge_call(_lib.skmetal_sv_init, parent, parent.size)


def sv_hook(edges: np.ndarray, parent: np.ndarray) -> None:
    """One SV hook iteration on the edge list."""
    _bridge_call(_lib.skmetal_sv_hook, edges, parent, edges.size // 2, parent.size)


def sv_shortcut(parent: np.ndarray) -> None:
    """SV shortcut: parent[i] = parent[parent[i]]."""
    _bridge_call(_lib.skmetal_sv_shortcut, parent, parent.size)



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





def rbf_kernel_square(X: np.ndarray, K_out: np.ndarray, gamma: float) -> None:
    """Compute RBF kernel matrix: K[i][j] = exp(-gamma * ||X[i] - X[j]||^2).
    Row norms are computed internally on GPU.
    """
    _bridge_call(_lib.skmetal_rbf_kernel_square,
                 X, K_out, gamma,
                 X.shape[0], X.shape[1])


def rbf_kernel_cross(X1: np.ndarray, X2: np.ndarray,
                      K_out: np.ndarray, gamma: float) -> None:
    """Cross RBF kernel: K[i][j] = exp(-gamma * ||X1[i] - X2[j]||^2).
    Row norms for both matrices computed internally on GPU.
    """
    _bridge_call(_lib.skmetal_rbf_kernel_cross,
                 X1, X2, K_out, gamma,
                 X1.shape[0], X2.shape[0], X1.shape[1])


def svc_predict_binary(X_test: np.ndarray, X_sv: np.ndarray,
                        dual_coef: np.ndarray, intercept: np.ndarray,
                        decisions: np.ndarray, gamma: float) -> None:
    """Matrix-free SVC binary predict: decision[i] = Σ_k dc[k] * exp(-γ * ||x_i - sv_k||²) + b.
    Avoids materializing the full n_test × n_SV Gram matrix.
    """
    _bridge_call(_lib.skmetal_svc_predict_binary,
                 X_test, X_sv, dual_coef, intercept, decisions,
                 X_test.shape[0], X_sv.shape[0], X_test.shape[1], gamma)



def logreg_irls_fit(X: np.ndarray, y: np.ndarray, C: float, tol: float,
                     max_iter: int, fit_intercept: bool) -> tuple[np.ndarray, int]:
    """Full binary IRLS fit loop in Swift. Returns (coef, n_iter).

    Note: ``fit_intercept`` is passed through to the bridge (currently unused
    there). The caller is responsible for appending the ones column to X.
    """
    n, p = X.shape
    coef = np.empty(p, dtype=np.float32, order="C")
    n_iter_out = ctypes.c_int32(0)
    _bridge_call(_lib.skmetal_logreg_irls_fit,
                 X, y, coef, ctypes.c_float(C), ctypes.c_float(tol),
                 ctypes.c_int32(max_iter), ctypes.c_int32(int(fit_intercept)),
                 ctypes.c_size_t(n), ctypes.c_size_t(p),
                 ctypes.byref(n_iter_out))
    return coef, n_iter_out.value


def logreg_lbfgs_fit(X: np.ndarray, y: np.ndarray, C: float, tol: float,
                     max_iter: int, fit_intercept: bool) -> tuple[np.ndarray, int]:
    """Full binary L-BFGS fit loop in Swift. Returns (coef, n_iter)."""
    n, p = X.shape
    coef = np.empty(p, dtype=np.float32, order="C")
    n_iter_out = ctypes.c_int32(0)
    _bridge_call(_lib.skmetal_logreg_lbfgs_fit,
                 X, y, coef, ctypes.c_float(C), ctypes.c_float(tol),
                 ctypes.c_int32(max_iter), ctypes.c_int32(int(fit_intercept)),
                 ctypes.c_size_t(n), ctypes.c_size_t(p),
                 ctypes.byref(n_iter_out))
    return coef, n_iter_out.value


def multinomial_lbfgs_fit(X: np.ndarray, y_enc: np.ndarray, C: float, tol: float,
                           max_iter: int, n_classes: int) -> tuple[np.ndarray, int]:
    """Full multinomial L-BFGS fit loop in Swift. Returns (W, n_iter)."""
    n, p = X.shape
    W = np.empty((p, n_classes), dtype=np.float32, order="C")
    n_iter_out = ctypes.c_int32(0)
    _bridge_call(_lib.skmetal_multinomial_lbfgs_fit,
                 X, y_enc, W, ctypes.c_float(C), ctypes.c_float(tol),
                 ctypes.c_int32(max_iter),
                 ctypes.c_size_t(n), ctypes.c_size_t(p), ctypes.c_size_t(n_classes),
                 ctypes.byref(n_iter_out))
    return W, n_iter_out.value
