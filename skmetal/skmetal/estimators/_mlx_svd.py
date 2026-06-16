"""MLX-accelerated TruncatedSVD — GPU SVD via mlx.linalg.svd."""
import numpy as np
import warnings
from scipy import linalg
from sklearn.utils.extmath import svd_flip
from ._base import BaseGPUEstimator
from ._mlx_registry import has_mlx

try:
    import mlx.core as mx
except ImportError:
    mx = None

_HAS_MLX = has_mlx()


if _HAS_MLX:

    def _mlx_matmul(A: np.ndarray, B: np.ndarray, trans_A: bool = False,
                    trans_B: bool = False) -> np.ndarray:
        """MLX-native GEMM — uses mx.array ops directly (no ctypes/dylib)."""
        A_mx = mx.array(A if not trans_A else A.T, dtype=mx.float32)
        B_mx = mx.array(B if not trans_B else B.T, dtype=mx.float32)
        return np.array(A_mx @ B_mx)

    class MetalTruncatedSVDMLX(BaseGPUEstimator):
        """MLX-accelerated TruncatedSVD via randomized SVD with GPU SVD."""

        def __init__(self, _estimator=None):
            super().__init__(_estimator)
            if not _HAS_MLX:
                warnings.warn("MLX not available; falling back to Metal bridge")

        def fit(self, X, y=None, **kwargs):
            X, _ = self._validate_data(X, y)
            if not self._should_use_gpu(X):
                return self._fallback_fit(X, y, **kwargs)

            n, p = X.shape
            n_components = self._estimator.n_components
            if n_components is None:
                n_components = min(n, p)
            else:
                n_components = min(n_components, n, p)

            S, Vt = self._randomized_svd_mlx(X, n_components)

            self._estimator.components_ = np.array(Vt)
            self._estimator.singular_values_ = np.array(S)
            self._estimator.n_components_ = n_components
            self._estimator.n_features_in_ = p
            self._estimator.n_samples_seen_ = n

            explained_variance = (np.array(S, dtype=np.float64) ** 2) / (n - 1)
            total_var = np.var(X, axis=0, ddof=1).sum()
            self._estimator.explained_variance_ = explained_variance
            self._estimator.explained_variance_ratio_ = (
                explained_variance / total_var if total_var > 0 else explained_variance
            )
            self._fitted = True
            return self

        def _randomized_svd_mlx(self, X, n_components, n_oversamples=10, n_iter=2):
            """Randomized SVD with MLX GPU SVD."""
            n, p = X.shape
            r = min(n_components + n_oversamples, p)

            rng = np.random.RandomState(1999)
            Omega = rng.randn(p, r).astype(np.float32, order="C")

            # Y = X @ Omega  (n, r)
            Y = _mlx_matmul(X, Omega)

            # M = Y.T @ Y  (r, r) — small matrix, can do on CPU
            M = _mlx_matmul(Y, Y, trans_A=True)
            M_np = np.array(M, dtype=np.float64)

            # Cholesky QR: M = R.T @ R
            R = linalg.cholesky(M_np, lower=False).astype(np.float32)
            R_inv_T = np.linalg.inv(R).T.astype(np.float32, order="C")

            # Q = Y @ R_inv_T  (n, r)
            Q = _mlx_matmul(Y, R_inv_T)

            # B = Q.T @ X  (r, p)
            B = _mlx_matmul(Q, X, trans_A=True)
            B_np = np.array(B, dtype=np.float64)

            # SVD of small matrix B — use MLX SVD if available, else SciPy
            if hasattr(mx.linalg, "svd"):
                B_mx = mx.array(B_np)
                # SVD not supported on GPU, run on CPU
                _, s_B, Vt_B = mx.linalg.svd(B_mx, compute_uv=True, stream=mx.cpu)
                s_B = np.array(s_B)
                Vt_B = np.array(Vt_B)
            else:
                _, s_B, Vt_B = linalg.svd(B_np, full_matrices=False)

            S = s_B[:n_components].astype(np.float32)
            Vt = Vt_B[:n_components].astype(np.float32)
            _, Vt = svd_flip(None, Vt, u_based_decision=False)

            return S, Vt

        def transform(self, X):
            X = self._validate_data(X)[0]
            if not self._should_use_gpu(X) or not self._fitted:
                return self._fallback_transform(X)
            return _mlx_matmul(np.ascontiguousarray(X), np.ascontiguousarray(self._estimator.components_.T))
