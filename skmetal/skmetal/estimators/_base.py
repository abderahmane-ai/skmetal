import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from .._config import get_config


class BaseGPUEstimator(BaseEstimator):
    _estimator: BaseEstimator

    def __init__(self, _estimator=None):
        self._estimator = _estimator
        self._fitted = False

    def _validate_data(self, X, y=None):
        X = check_array(X, dtype=np.float32, order="C", ensure_min_samples=2)
        if y is not None:
            y = check_array(y, dtype=np.float32, order="C", ensure_2d=False)
        return X, y

    def _should_use_gpu(self, X):
        config = get_config()
        if config.device == "cpu":
            return False
        if hasattr(X, "nnz"):
            return False
        if X.dtype != np.float32:
            return False
        n, d = X.shape
        if n * d < config.threshold:
            return False
        name = type(self._estimator).__name__ if self._estimator else type(self).__name__
        if name in config.thresholds:
            min_rows, min_cols = config.thresholds[name]
            if n < min_rows or d < min_cols:
                return False
        return True

    def _fallback_fit(self, X, y, **kwargs):
        self._estimator.fit(X, y, **kwargs)
        self._fitted = True
        return self

    def _fallback_predict(self, X):
        return self._estimator.predict(X)

    def _fallback_transform(self, X):
        return self._estimator.transform(X)

    def _fallback_predict_proba(self, X):
        return self._estimator.predict_proba(X)

    def score(self, X, y, **kwargs):
        """Delegate score to the underlying estimator."""
        return self._estimator.score(X, y, **kwargs)

    def predict(self, X):
        return self._fallback_predict(X)

    def transform(self, X):
        return self._fallback_transform(X)

    def __sklearn_is_fitted__(self):
        return self._fitted

    def __getattr__(self, name):
        if name in ('_estimator', '_fitted'):
            raise AttributeError(name)
        if self._estimator is not None and hasattr(self._estimator, name):
            return getattr(self._estimator, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
