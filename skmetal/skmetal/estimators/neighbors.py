import numpy as np
from ._base import BaseGPUEstimator
from .._bridge import (
    knn_tiled_kneighbors,
    knn_vote_classify, knn_vote_regress,
    knn_vote_classify_weighted, knn_vote_regress_weighted,
)


class MetalKNeighborsMixin:
    _k_neighbors: int = 5
    _tile_size: int = 4096

    def _kneighbors(self, X, k=None):
        if k is None:
            k = self._k_neighbors

        metric = getattr(self._estimator, "metric", "euclidean")
        values, indices = knn_tiled_kneighbors(
            X, self._estimator._fit_X, k, self._tile_size,
            metric=metric,
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
            metric = getattr(self._estimator, "metric", "euclidean")
            if metric == "euclidean":
                distances = np.sqrt(distances)
            return distances, indices
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

        weights = getattr(self._estimator, "weights", "uniform")
        if weights == "distance":
            knn_vote_classify_weighted(indices, np.sqrt(distances),
                                        train_labels, predictions,
                                        n_test, k, len(train_labels))
        else:
            knn_vote_classify(indices, train_labels, predictions,
                              n_test, k, len(train_labels))

        classes = self._estimator.classes_
        pred_classes = classes[np.round(predictions).astype(int)]
        return pred_classes

    def predict_proba(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict_proba(X)

        distances, indices = self._kneighbors(X)
        k = self._k_neighbors
        classes = self._estimator.classes_

        weights = getattr(self._estimator, "weights", "uniform")

        # Resolve neighbor labels for all test instances simultaneously
        # ravel() ensures indexing is safe even if _y is shape (n, 1)
        neighbor_labels = self._estimator._y.ravel()[indices]  # shape (n_test, k)

        # Broadcast comparison against classes (n_test, k, n_classes)
        mask = (neighbor_labels[:, :, np.newaxis] == classes[np.newaxis, np.newaxis, :])

        if weights == "distance":
            d_safe = np.sqrt(np.maximum(distances, 1e-10))
            w = 1.0 / d_safe
            # Weighted vote: sum weights where mask is true
            proba = (mask * w[:, :, np.newaxis]).sum(axis=1)
            # Normalize probabilities per row
            row_sums = proba.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            proba /= row_sums
        else:
            proba = mask.sum(axis=1).astype(np.float32) / k

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

        weights = getattr(self._estimator, "weights", "uniform")
        if weights == "distance":
            knn_vote_regress_weighted(indices, np.sqrt(distances),
                                       train_targets, predictions,
                                       n_test, k, len(train_targets))
        else:
            knn_vote_regress(indices, train_targets, predictions,
                             n_test, k, len(train_targets))

        return predictions

    def predict_proba(self, X):
        return self._fallback_predict_proba(X)
