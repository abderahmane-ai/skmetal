import numpy as np
from scipy import linalg
from sklearn.utils.extmath import svd_flip
from ._base import BaseGPUEstimator
from .._bridge import gemm


class MetalTruncatedSVD(BaseGPUEstimator):
    """GPU-accelerated TruncatedSVD via randomized SVD (Cholesky QR, all BLAS-3).
    """

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

        S, Vt = self._randomized_svd(X, n_components)

        self._estimator.components_ = Vt
        self._estimator.singular_values_ = S
        self._estimator.n_components_ = n_components
        self._estimator.n_features_in_ = p
        self._estimator.n_samples_seen_ = n

        explained_variance = (S.astype(np.float64) ** 2) / (n - 1)
        total_var = np.var(X, axis=0, ddof=1).sum()
        self._estimator.explained_variance_ = explained_variance
        self._estimator.explained_variance_ratio_ = (
            explained_variance / total_var if total_var > 0 else explained_variance
        )
        self._fitted = True
        return self

    def _randomized_svd(self, X, n_components, n_oversamples=10, n_iter=2):
        """Randomized SVD without centering — all BLAS-3 on GPU."""
        n, p = X.shape
        r = min(n_components + n_oversamples, p)

        rng = np.random.RandomState(1999)
        Omega = rng.randn(p, r).astype(np.float32, order="C")

        Y = gemm(X, Omega)
        M = gemm(Y, Y, trans_A=True)
        M_np = np.array(M, dtype=np.float64)
        R = linalg.cholesky(M_np, lower=False).astype(np.float32)
        R_inv_T = np.linalg.inv(R).T.astype(np.float32, order="C")
        Q = gemm(Y, R_inv_T)
        B = gemm(Q, X, trans_A=True)
        B_np = np.array(B, dtype=np.float64)
        U_B, s_B, Vt_B = linalg.svd(B_np, full_matrices=False)
        U_B_sub = U_B[:, :n_components].astype(np.float32, order="C")
        V = gemm(Q, U_B_sub)
        S = s_B[:n_components].astype(np.float32)
        Vt = Vt_B[:n_components].astype(np.float32)
        _, Vt = svd_flip(None, Vt, u_based_decision=False)

        return S, Vt

    def transform(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_transform(X)
        return gemm(np.ascontiguousarray(X), np.ascontiguousarray(self._estimator.components_.T))
