import numpy as np
from ._base import BaseGPUEstimator
from .._bridge import row_norm_sq, rbf_kernel_square, rbf_kernel_cross
from sklearn.svm import SVC as _SKSVC
from sklearn.svm import SVR as _SKSVR


def _resolve_gamma(gamma, n_features, X_var):
    if isinstance(gamma, str):
        if gamma == "scale":
            return 1.0 / (n_features * max(X_var, 1e-12))
        elif gamma == "auto":
            return 1.0 / n_features
        else:
            return 1.0
    return float(gamma)


class _BaseMetalSVM(BaseGPUEstimator):
    def __init__(self, _estimator=None):
        super().__init__(_estimator)
        self._X_train = None
        self._X_train_norm = None
        self._saved_kernel = None
        self._saved_gamma = None
        self._n_features = 0

    def _compute_test_kernel(self, X):
        n_test, d = X.shape
        gamma = self._saved_gamma
        if isinstance(gamma, str):
            gamma = _resolve_gamma(gamma, d, X.var())
        gamma = float(gamma)

        X_test_norm = np.empty(n_test, dtype=np.float32)
        row_norm_sq(X, X_test_norm)

        K_test = np.empty((n_test, self._X_train.shape[0]), dtype=np.float32, order="C")
        rbf_kernel_cross(X, X_test_norm, self._X_train, self._X_train_norm, K_test, gamma)
        return K_test

    def score(self, X, y, **kwargs):
        if not self._fitted or self._X_train is None:
            return self._estimator.score(X, y, **kwargs)
        X = self._validate_data(X)[0]
        K_test = self._compute_test_kernel(X)
        return self._estimator.score(K_test, y, **kwargs)


class MetalSVC(_BaseMetalSVM):
    def fit(self, X, y, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n, d = X.shape
        if self._estimator.kernel != "rbf":
            return self._fallback_fit(X, y, **kwargs)

        gamma = _resolve_gamma(self._estimator.gamma, d, X.var())

        X_norm = np.empty(n, dtype=np.float32)
        row_norm_sq(X, X_norm)

        K = np.empty((n, n), dtype=np.float32, order="C")
        rbf_kernel_square(X, X_norm, K, gamma)

        self._X_train = X.copy()
        self._X_train_norm = X_norm.copy()
        self._saved_kernel = self._estimator.kernel
        self._saved_gamma = self._estimator.gamma
        self._n_features = d
        self._estimator.kernel = "precomputed"
        self._estimator.gamma = gamma

        self._estimator.fit(K, y, **kwargs)
        self._fitted = True
        return self

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted or self._X_train is None:
            return self._fallback_predict(X)

        K_test = self._compute_test_kernel(X)
        return self._estimator.predict(K_test)

    def predict_proba(self, X):
        if not self._fitted or self._X_train is None:
            return self._fallback_predict_proba(X)
        X = self._validate_data(X)[0]
        K_test = self._compute_test_kernel(X)
        return self._estimator.predict_proba(K_test)

    def decision_function(self, X):
        X = self._validate_data(X)[0]
        if not self._fitted or self._X_train is None:
            return self._estimator.decision_function(X)

        K_test = self._compute_test_kernel(X)
        return self._estimator.decision_function(K_test)


class MetalSVR(_BaseMetalSVM):
    def fit(self, X, y, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n, d = X.shape
        if self._estimator.kernel != "rbf":
            return self._fallback_fit(X, y, **kwargs)

        gamma = _resolve_gamma(self._estimator.gamma, d, X.var())

        X_norm = np.empty(n, dtype=np.float32)
        row_norm_sq(X, X_norm)

        K = np.empty((n, n), dtype=np.float32, order="C")
        rbf_kernel_square(X, X_norm, K, gamma)

        self._X_train = X.copy()
        self._X_train_norm = X_norm.copy()
        self._saved_kernel = self._estimator.kernel
        self._saved_gamma = self._estimator.gamma
        self._n_features = d
        self._estimator.kernel = "precomputed"
        self._estimator.gamma = gamma

        self._estimator.fit(K, y, **kwargs)
        self._fitted = True
        return self

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted or self._X_train is None:
            return self._fallback_predict(X)

        K_test = self._compute_test_kernel(X)
        return self._estimator.predict(K_test)


class MetalSVR(BaseGPUEstimator):
    def __init__(self, _estimator=None):
        super().__init__(_estimator)
        self._X_train = None
        self._X_train_norm = None
        self._saved_kernel = None
        self._saved_gamma = None

    def fit(self, X, y, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n, d = X.shape
        if self._estimator.kernel != "rbf":
            return self._fallback_fit(X, y, **kwargs)

        gamma = _resolve_gamma(self._estimator.gamma, d, X.var())

        X_norm = np.empty(n, dtype=np.float32)
        row_norm_sq(X, X_norm)

        K = np.empty((n, n), dtype=np.float32, order="C")
        rbf_kernel_square(X, X_norm, K, gamma)

        self._X_train = X.copy()
        self._X_train_norm = X_norm.copy()
        self._saved_kernel = self._estimator.kernel
        self._saved_gamma = self._estimator.gamma
        self._estimator.kernel = "precomputed"
        self._estimator.gamma = gamma

        self._estimator.fit(K, y, **kwargs)
        self._fitted = True
        return self

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted or self._X_train is None:
            return self._fallback_predict(X)

        n_test, d = X.shape
        gamma = self._saved_gamma
        if isinstance(gamma, str):
            gamma = _resolve_gamma(gamma, d, X.var())
        gamma = float(gamma)

        X_test_norm = np.empty(n_test, dtype=np.float32)
        row_norm_sq(X, X_test_norm)

        K_test = np.empty((n_test, self._X_train.shape[0]), dtype=np.float32, order="C")
        rbf_kernel_cross(X, X_test_norm, self._X_train, self._X_train_norm, K_test, gamma)

        return self._estimator.predict(K_test)
