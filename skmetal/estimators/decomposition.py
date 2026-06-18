import numpy as np
from scipy import linalg
from sklearn.utils.extmath import svd_flip
from ._base import BaseGPUEstimator
from .._bridge import gemm


class MetalTruncatedSVD(BaseGPUEstimator):
    """GPU-accelerated TruncatedSVD via randomized SVD (Cholesky QR, all BLAS-3)."""

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

    def _randomized_svd(self, X, n_components, n_oversamples=10, n_iter=4):
        """Randomized SVD without centering — all BLAS-3 on GPU.

        Uses power iterations to improve accuracy: Y = (X @ Xᵀ)^n_iter @ X @ Omega
        Each power iteration is two GEMM calls on GPU, followed by CPU QR for
        re-orthogonalization (matches sklearn's randomized SVD algorithm).
        Defaults: n_oversamples=10, n_iter=4 (same as sklearn).
        """
        n, p = X.shape
        r = min(n_components + n_oversamples, p)

        rng = np.random.RandomState(self._estimator.random_state or 1999)
        Omega = rng.randn(p, r).astype(np.float32, order="C")

        # Stage 1: Y = X @ Omega
        Y = gemm(X, Omega)

        # Stage 2: Power iterations — sharpen the singular value spectrum.
        # Matches sklearn's randomized_svd. For n > p:
        #   Q = QR(Xᵀ @ Q)  → (p, r), then Q = QR(X @ Q) → (n, r).
        for _ in range(n_iter):
            # Half-iteration 1: Y = Xᵀ @ Y (p × r), then QR
            Y = gemm(X, Y, trans_A=True)
            Y_np = np.array(Y, dtype=np.float64)
            Y, _ = linalg.qr(Y_np, mode="economic")
            Y = Y.astype(np.float32, order="C")

            # Half-iteration 2: Y = X @ Y (n × r), then QR
            Y = gemm(X, Y)
            Y_np = np.array(Y, dtype=np.float64)
            Y, _ = linalg.qr(Y_np, mode="economic")
            Y = Y.astype(np.float32, order="C")

        # Stage 3: Cholesky QR — compute orthonormal basis Q for Y (n × r)
        M = gemm(Y, Y, trans_A=True)
        M_np = np.array(M, dtype=np.float64)
        R = linalg.cholesky(M_np, lower=False).astype(np.float32)
        R_inv_T = np.linalg.inv(R).T.astype(np.float32, order="C")
        Q = gemm(Y, R_inv_T)

        # Stage 4: B = Qᵀ @ X (small matrix, r × p)
        B = gemm(Q, X, trans_A=True)
        B_np = np.array(B, dtype=np.float64)

        # Stage 5: SVD of small B on CPU
        U_B, s_B, Vt_B = linalg.svd(B_np, full_matrices=False)
        S = s_B[:n_components].astype(np.float32)
        Vt = Vt_B[:n_components].astype(np.float32)
        _, Vt = svd_flip(None, Vt, u_based_decision=False)

        return S, Vt

    def transform(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_transform(X)
        return gemm(np.ascontiguousarray(X), np.ascontiguousarray(self._estimator.components_.T))


class MetalPCA(BaseGPUEstimator):
    """GPU-accelerated PCA via randomized SVD on centered data.

    PCA = center columns → randomized SVD. The heavy operations (GEMM,
    randomized projection, Cholesky QR) run on GPU via MPS. Mean
    computation and small-SVD are on CPU where they are already fast.
    """

    def fit(self, X, y=None, **kwargs):
        X, _ = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n, p = X.shape
        n_components = self._estimator.n_components
        if n_components is None:
            n_components = min(n, p)
        elif isinstance(n_components, float):
            # sklearn PCA: n_components as float means "explained variance ratio target"
            # Fall back to CPU for this case (requires full SVD to accumulate variance).
            return self._fallback_fit(X, y, **kwargs)
        else:
            n_components = min(int(n_components), n, p)

        # 1. Compute column means (CPU — O(nd), cheap compared to SVD)
        mean_ = X.mean(axis=0, dtype=np.float32)

        # 2. Center data
        X_centered = X - mean_[np.newaxis, :]

        # 3. Randomized SVD on centered data (GPU-heavy)
        S, Vt = self._randomized_svd(X_centered, n_components)

        # 4. Store fitted attributes
        self._estimator.components_ = Vt
        self._estimator.singular_values_ = S
        self._estimator.mean_ = mean_
        self._estimator.n_components_ = n_components
        self._estimator.n_features_in_ = p
        self._estimator.n_samples_seen_ = n

        # 5. Explained variance (on centered data)
        explained_variance = (S.astype(np.float64) ** 2) / (n - 1)
        total_var = np.var(X_centered, axis=0, ddof=1).sum()
        self._estimator.explained_variance_ = explained_variance
        self._estimator.explained_variance_ratio_ = (
            explained_variance / total_var if total_var > 0 else explained_variance
        )

        self._fitted = True
        return self

    def _randomized_svd(self, X, n_components, **kwargs):
        """Shared randomized SVD — delegates to MetalTruncatedSVD's implementation."""
        # Use the parent class's implementation which has the full power-iteration
        # algorithm with QR re-orthogonalization (identical to sklearn's approach).
        return MetalTruncatedSVD._randomized_svd(self, X, n_components, **kwargs)

    def transform(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_transform(X)
        # (X - mean_) @ components_.T
        X_centered = X - self._estimator.mean_[np.newaxis, :]
        return gemm(
            np.ascontiguousarray(X_centered),
            np.ascontiguousarray(self._estimator.components_.T),
        )

    def inverse_transform(self, X_reduced):
        X_reduced = self._validate_data(X_reduced)[0]
        if not self._fitted:
            try:
                return self._estimator.inverse_transform(X_reduced)
            except Exception:
                raise RuntimeError("PCA must be fitted before inverse_transform.")
        # X_reduced @ components_ + mean_
        proj = gemm(
            np.ascontiguousarray(X_reduced),
            np.ascontiguousarray(self._estimator.components_),
        )
        return proj + self._estimator.mean_[np.newaxis, :]
