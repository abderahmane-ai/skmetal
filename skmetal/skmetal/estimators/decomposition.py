import numpy as np
from scipy import linalg
from sklearn.utils.extmath import svd_flip
from ._base import BaseGPUEstimator
from .._bridge import svd as metal_svd, gemm, scaler_fit, center_columns


class MetalPCA(BaseGPUEstimator):
    def _randomized_pca(self, X, n_components, n_oversamples=10, n_iter=2):
        """Randomized PCA via Cholesky QR, all BLAS-3 on GPU."""
        n, p = X.shape
        r = min(n_components + n_oversamples, p)

        # 1. Random projection matrix
        rng = np.random.RandomState(1999)
        Omega = rng.randn(p, r).astype(np.float32, order='C')

        # 2. Y = X @ Omega on GPU (BLAS-3)
        Y = gemm(X, Omega)

        # 3. Cholesky QR: M = Y^T @ Y, R = cholesky(M), Q = Y @ inv(R^T)
        M = gemm(Y, Y, trans_A=True)
        M_np = np.array(M, dtype=np.float64)
        R = linalg.cholesky(M_np, lower=False).astype(np.float32)
        R_inv_T = np.linalg.inv(R).T.astype(np.float32, order='C')
        Q = gemm(Y, R_inv_T)

        # 4. B = Q^T @ X on GPU (BLAS-3)
        B = gemm(Q, X, trans_A=True)

        # 5. SVD of small B (r × p) on CPU
        B_np = np.array(B, dtype=np.float64)
        U_B, s_B, Vt_B = linalg.svd(B_np, full_matrices=False)

        # 6. V = Q @ U_B on GPU (BLAS-3)
        U_B_sub = U_B[:, :n_components].astype(np.float32, order='C')
        V = gemm(Q, U_B_sub)

        # Singular values and components
        S = s_B[:n_components].astype(np.float32)
        Vt = Vt_B[:n_components].astype(np.float32)
        _, Vt = svd_flip(None, Vt, u_based_decision=False)

        return S, Vt

    def fit(self, X, y=None, **kwargs):
        X, _ = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n_samples, n_features = X.shape
        n_components = self._estimator.n_components
        if n_components is None:
            n_components = min(n_samples, n_features)
        else:
            n_components = min(n_components, n_samples, n_features)

        mean = np.empty(n_features, dtype=np.float32)
        var_out = np.empty(n_features, dtype=np.float32)
        Xc = np.ascontiguousarray(X, dtype=np.float32)
        scaler_fit(Xc, mean, var_out)
        center_columns(Xc, mean)

        m, n = Xc.shape
        k = n_components

        # Use randomized PCA for n_components up to 80% of min(n, p), else full via covariance
        if k <= 0.8 * min(m, n):
            S, Vt = self._randomized_pca(Xc, k)
        elif n <= m:
            C = gemm(Xc, Xc, trans_A=True) / (m - 1)
            C_np = np.array(C, dtype=np.float64)
            eigenvalues, eigenvectors = linalg.eigh(C_np)
            eigenvalues = eigenvalues[::-1]
            eigenvectors = eigenvectors[:, ::-1]
            S = np.sqrt(eigenvalues[:k] * (m - 1)).astype(np.float32)
            Vt = eigenvectors[:, :k].T.astype(np.float32)
            _, Vt = svd_flip(None, Vt, u_based_decision=False)
        else:
            U = np.empty((m, k), dtype=np.float32, order="C")
            S = np.empty(k, dtype=np.float32)
            Vt = np.empty((k, n), dtype=np.float32, order="C")
            metal_svd(Xc, U, S, Vt, m, n, k)

        self._estimator.components_ = Vt
        self._estimator.singular_values_ = S
        self._estimator.mean_ = mean
        self._estimator.n_components_ = k
        self._estimator.n_features_in_ = n_features
        self._estimator.n_samples_seen_ = n_samples

        explained_variance = (S.astype(np.float64) ** 2) / (n_samples - 1)
        total_var = (var_out * n_samples / (n_samples - 1)).sum()
        self._estimator.explained_variance_ = explained_variance
        self._estimator.explained_variance_ratio_ = explained_variance / total_var if total_var > 0 else explained_variance
        self._estimator.noise_variance_ = max(0, total_var - explained_variance.sum())

        self._fitted = True
        return self

    def transform(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_transform(X)
        X_centered = np.ascontiguousarray(X - self._estimator.mean_)
        return gemm(X_centered, np.ascontiguousarray(self._estimator.components_.T))


class MetalTruncatedSVD(BaseGPUEstimator):
    """GPU-accelerated TruncatedSVD via randomized SVD (Cholesky QR, all BLAS-3).

    Same as MetalPCA but without centering — operates on sparse-compatible
    raw data using only matrix operations.
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
