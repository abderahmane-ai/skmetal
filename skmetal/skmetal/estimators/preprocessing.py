import numpy as np
from ._base import BaseGPUEstimator
from .._bridge import scaler_fit, column_minmax, column_transform


class MetalStandardScaler(BaseGPUEstimator):
    def fit(self, X, y=None, **kwargs):
        X, _ = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n_samples, n_features = X.shape
        mean_out = np.empty(n_features, dtype=np.float32)
        var_out = np.empty(n_features, dtype=np.float32)

        scaler_fit(X, mean_out, var_out)

        self._estimator.mean_ = mean_out
        self._estimator.var_ = var_out
        self._estimator.scale_ = np.sqrt(var_out)
        self._estimator.scale_[self._estimator.scale_ < 1e-15] = 1.0
        self._estimator.n_features_in_ = n_features
        self._fitted = True
        return self

    def transform(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_transform(X)
        return (X - self._estimator.mean_) / self._estimator.scale_


class MetalMinMaxScaler(BaseGPUEstimator):
    def fit(self, X, y=None, **kwargs):
        X, _ = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n_features = X.shape[1]
        feature_range = self._estimator.feature_range
        data_min = np.empty(n_features, dtype=np.float32)
        data_max = np.empty(n_features, dtype=np.float32)
        column_minmax(X, data_min, data_max)
        data_range = data_max - data_min

        self._estimator.data_min_ = data_min
        self._estimator.data_max_ = data_max
        self._estimator.data_range_ = data_range
        self._estimator.n_features_in_ = n_features

        scale = np.where(data_range == 0, 1.0, 1.0 / data_range)
        min_adj = feature_range[0] - data_min * scale
        self._estimator.scale_ = scale
        self._estimator.min_ = min_adj
        self._fitted = True
        return self

    def transform(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_transform(X)
        return X * self._estimator.scale_ + self._estimator.min_


class MetalRobustScaler(BaseGPUEstimator):
    def fit(self, X, y=None, **kwargs):
        X, _ = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n_features = X.shape[1]
        q1 = np.empty(n_features, dtype=np.float32)
        med = np.empty(n_features, dtype=np.float32)
        q3 = np.empty(n_features, dtype=np.float32)

        for j in range(n_features):
            col = X[:, j]
            q1[j], med[j], q3[j] = np.percentile(col, [25, 50, 75])

        iqr = q3 - q1
        iqr[iqr < 1e-15] = 1.0

        self._estimator.center_ = med
        self._estimator.scale_ = iqr
        self._estimator.n_features_in_ = n_features
        self._fitted = True
        return self

    def transform(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_transform(X)

        n, d = X.shape
        scale = 1.0 / self._estimator.scale_
        output = np.empty_like(X)
        column_transform(X, output, self._estimator.center_, scale)
        return output

    def inverse_transform(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_transform(X)

        return X * self._estimator.scale_ + self._estimator.center_
