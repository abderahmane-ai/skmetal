import numpy as np
from ._base import BaseGPUEstimator
from .._bridge import (
    knn_tiled_kneighbors,
    knn_vote_classify, knn_vote_regress,
)


class MetalKNeighborsMixin:
    _k_neighbors: int = 5
    _tile_size: int = 4096

    def _kneighbors(self, X, k=None):
        if k is None:
            k = self._k_neighbors

        values, indices = knn_tiled_kneighbors(
            X, self._estimator._fit_X, k, self._tile_size,
        )
        return values, indices

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if X is None:
            X = self._estimator._fit_X
        if n_neighbors is None:
            n_neighbors = self._k_neighbors

        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X):
            return self._estimator.kneighbors(X, n_neighbors, return_distance)

        distances, indices = self._kneighbors(X, n_neighbors)
        if return_distance:
            return np.sqrt(distances), indices
        return indices


class MetalNearestNeighbors(MetalKNeighborsMixin, BaseGPUEstimator):
    def fit(self, X, y=None, **kwargs):
        X, _ = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        self._estimator.fit(X)
        self._k_neighbors = getattr(self._estimator, 'n_neighbors', 5)
        self._fitted = True
        return self


class MetalKNeighborsClassifier(MetalKNeighborsMixin, BaseGPUEstimator):
    def fit(self, X, y, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        self._estimator.fit(X, y)
        self._k_neighbors = getattr(self._estimator, 'n_neighbors', 5)
        self._fitted = True
        return self

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)

        distances, indices = self._kneighbors(X)
        n_test = X.shape[0]
        k = self._k_neighbors

        predictions = np.empty(n_test, dtype=np.float32)
        train_labels = self._estimator._y.astype(np.float32).ravel()
        knn_vote_classify(indices, train_labels, predictions, n_test, k, len(train_labels))

        classes = self._estimator.classes_
        pred_classes = classes[np.round(predictions).astype(int)]
        return pred_classes

    def predict_proba(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict_proba(X)

        distances, indices = self._kneighbors(X)
        n_test = X.shape[0]
        k = self._k_neighbors
        classes = self._estimator.classes_
        n_classes = len(classes)

        proba = np.zeros((n_test, n_classes), dtype=np.float32)
        for i in range(n_test):
            neighbor_labels = self._estimator._y[indices[i]]
            for lbl in neighbor_labels:
                idx = np.where(classes == lbl)[0]
                if len(idx) > 0:
                    proba[i, idx[0]] += 1.0
            proba[i] /= k

        return proba


class MetalKNeighborsRegressor(MetalKNeighborsMixin, BaseGPUEstimator):
    def fit(self, X, y, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        self._estimator.fit(X, y)
        self._k_neighbors = getattr(self._estimator, 'n_neighbors', 5)
        self._fitted = True
        return self

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)

        distances, indices = self._kneighbors(X)
        n_test = X.shape[0]
        k = self._k_neighbors

        predictions = np.empty(n_test, dtype=np.float32)
        train_targets = self._estimator._y.astype(np.float32).ravel()
        knn_vote_regress(indices, train_targets, predictions, n_test, k, len(train_targets))

        return predictions

    def predict_proba(self, X):
        return self._fallback_predict_proba(X)
