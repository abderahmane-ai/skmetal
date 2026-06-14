import numpy as np
from ._base import BaseGPUEstimator
from .._bridge import rbf_kernel_square, rbf_kernel_cross, svc_predict_binary


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
        self._saved_kernel = None
        self._saved_gamma = None
        self._n_features = 0

    def _compute_test_kernel(self, X):
        n_test, d = X.shape
        gamma = self._saved_gamma
        if isinstance(gamma, str):
            gamma = _resolve_gamma(gamma, d, X.var())
        gamma = float(gamma)

        K_test = np.empty((n_test, self._X_train.shape[0]), dtype=np.float32, order="C")
        rbf_kernel_cross(X, self._X_train, K_test, gamma)
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

        K = np.empty((n, n), dtype=np.float32, order="C")
        rbf_kernel_square(X, K, gamma)

        self._X_train = X.copy()
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

        n_classes = len(self._estimator.classes_)
        if n_classes == 2 and self._saved_kernel == "rbf":
            # Matrix-free binary predict: avoid materializing full Gram matrix
            d = X.shape[1]
            gamma = _resolve_gamma(self._saved_gamma, d, X.var())
            sv_idx = self._estimator.support_
            X_sv = self._X_train[sv_idx]
            dual_coef = self._estimator.dual_coef_.ravel().astype(np.float32)
            intercept = np.asarray(self._estimator.intercept_, dtype=np.float32)
            decisions = np.empty(X.shape[0], dtype=np.float32)
            svc_predict_binary(X, X_sv, dual_coef, intercept, decisions, gamma)
            return self._estimator.classes_[(decisions > 0).astype(int)]
        else:
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

        n_classes = len(self._estimator.classes_)
        if n_classes == 2 and self._saved_kernel == "rbf":
            d = X.shape[1]
            gamma = _resolve_gamma(self._saved_gamma, d, X.var())
            sv_idx = self._estimator.support_
            X_sv = self._X_train[sv_idx]
            dual_coef = self._estimator.dual_coef_.ravel().astype(np.float32)
            intercept = np.asarray(self._estimator.intercept_, dtype=np.float32)
            decisions = np.empty(X.shape[0], dtype=np.float32)
            svc_predict_binary(X, X_sv, dual_coef, intercept, decisions, gamma)
            return decisions
        else:
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

        K = np.empty((n, n), dtype=np.float32, order="C")
        rbf_kernel_square(X, K, gamma)

        self._X_train = X.copy()
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

        if self._saved_kernel == "rbf":
            d = X.shape[1]
            gamma = _resolve_gamma(self._saved_gamma, d, X.var())
            sv_idx = self._estimator.support_
            X_sv = self._X_train[sv_idx]
            dual_coef = self._estimator.dual_coef_.ravel().astype(np.float32)
            intercept = np.asarray(self._estimator.intercept_, dtype=np.float32)
            pred = np.empty(X.shape[0], dtype=np.float32)
            svc_predict_binary(X, X_sv, dual_coef, intercept, pred, gamma)
            return pred
        else:
            K_test = self._compute_test_kernel(X)
            return self._estimator.predict(K_test)
