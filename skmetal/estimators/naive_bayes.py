import numpy as np
from ._base import BaseGPUEstimator
from .._bridge import scaler_fit


class MetalGaussianNB(BaseGPUEstimator):
    """GPU-accelerated GaussianNB via GPU mean/variance per class."""
    def fit(self, X, y, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n, d = X.shape
        classes = np.unique(y)
        n_classes = len(classes)

        theta = np.empty((n_classes, d), dtype=np.float32)
        var = np.empty((n_classes, d), dtype=np.float32)
        class_count = np.empty(n_classes, dtype=np.float32)

        for i, c in enumerate(classes):
            mask = y == c
            X_c = np.ascontiguousarray(X[mask])
            class_count[i] = float(X_c.shape[0])
            scaler_fit(X_c, theta[i], var[i])

        class_prior = class_count / class_count.sum()

        self._estimator.classes_ = classes
        self._estimator.class_count_ = class_count
        self._estimator.class_prior_ = class_prior
        self._estimator.theta_ = theta
        self._estimator.var_ = var
        self._estimator.epsilon_ = np.finfo(np.float32).eps
        self._fitted = True
        return self

    def _joint_log_likelihood(self, X):
        """Compute joint log-likelihood for all classes."""
        theta = self._estimator.theta_
        var = self._estimator.var_
        class_prior = self._estimator.class_prior_
        eps = self._estimator.epsilon_

        jll = []
        for i in range(len(self._estimator.classes_)):
            jointi = np.log(class_prior[i])
            jointi -= 0.5 * np.sum(np.log(2.0 * np.pi * (var[i] + eps)))
            jointi -= 0.5 * np.sum((X - theta[i]) ** 2 / (var[i] + eps), axis=1)
            jll.append(jointi)

        return np.column_stack(jll)

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        jll = self._joint_log_likelihood(X)
        return self._estimator.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict_proba(X)
        jll = self._joint_log_likelihood(X)
        log_prob = jll - np.max(jll, axis=1, keepdims=True)
        prob = np.exp(log_prob)
        return prob / prob.sum(axis=1, keepdims=True)
