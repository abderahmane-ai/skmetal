import numpy as np
from sklearn.utils.validation import check_random_state
from ._base import BaseGPUEstimator
from .._bridge import (
    pairwise_distance,
    kmeans_assign,
    kmeans_batch_fused,
    compute_mindists,
)


class MetalKMeans(BaseGPUEstimator):
    def fit(self, X, y=None, **kwargs):
        X, _ = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n, d = X.shape
        k = self._estimator.n_clusters
        max_iter = self._estimator.max_iter

        centroids = self._kmeans_plusplus_init(X, k)
        assignments = np.empty(n, dtype=np.uint32)
        num_groups = max(1, (n + 255) // 256)

        kmeans_batch_fused(X, centroids, assignments,
                           n, d, k, num_groups, max_iter)

        self._estimator.cluster_centers_ = centroids
        self._estimator.labels_ = assignments.astype(np.int32)
        self._estimator.inertia_ = self._compute_inertia(X, assignments, centroids)
        self._estimator.n_iter_ = max_iter
        self._estimator.n_features_in_ = d
        self._fitted = True
        return self

    def _kmeans_plusplus_init(self, X, k):
        rng = check_random_state(self._estimator.random_state)
        n, d = X.shape
        centroids = np.empty((k, d), dtype=np.float32)
        centroids[0] = X[rng.randint(n)]
        assignments = np.empty(n, dtype=np.uint32)
        dists = np.empty(n, dtype=np.float32)
        c_view = np.empty((k, d), dtype=np.float32)

        for i in range(1, k):
            c_view[:i] = centroids[:i]
            kmeans_assign(X, c_view[:i], assignments, n, d, i)
            compute_mindists(X, c_view[:i], assignments, dists, n, d, i)
            dists = np.maximum(dists, 1e-30)
            probs = dists / dists.sum()
            centroids[i] = X[rng.choice(n, p=probs)]
        return centroids

    def _compute_inertia(self, X, assignments, centroids):
        diff = X - centroids[assignments]
        return float(np.sum(diff * diff))

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        n, d = X.shape
        k = self._estimator.cluster_centers_.shape[0]
        assignments = np.empty(n, dtype=np.uint32)
        kmeans_assign(X, self._estimator.cluster_centers_, assignments, n, d, k)
        return assignments.astype(np.int32)

    def _ensure_centroids_contiguous(self, centroids):
        if not centroids.flags['C_CONTIGUOUS']:
            return np.ascontiguousarray(centroids)
        return centroids

    def transform(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_transform(X)
        return pairwise_distance(X)
