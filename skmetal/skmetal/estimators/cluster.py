import numpy as np
from sklearn.utils.validation import check_random_state
from ._base import BaseGPUEstimator
from .._bridge import (
    pairwise_distance,
    kmeans_assign,
    kmeans_batch_fused,
    compute_mindists,
    sv_init, sv_hook, sv_shortcut,
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


class MetalDBSCAN(BaseGPUEstimator):
    def _gpu_sv_connected_components(self, core_indices, n_total, neighbor_mask):
        """GPU Shiloach-Vishkin connected components on the core-point subgraph.

        Returns core_labels: int32 array of length len(core_indices) with
        consecutive cluster labels 0..n_comp-1.
        """
        n_core = len(core_indices)
        if n_core <= 1:
            return np.zeros(n_core, dtype=np.int32)

        core_set = set(core_indices)
        edges_list = []
        for i in core_indices:
            for j in core_indices:
                if j > i and neighbor_mask[i, j]:
                    edges_list.append(i)
                    edges_list.append(j)

        if not edges_list:
            return np.zeros(n_core, dtype=np.int32)

        edges = np.array(edges_list, dtype=np.int32)

        parent = np.empty(n_total, dtype=np.int32)
        sv_init(parent, n_total)

        num_iters = int(np.ceil(np.log2(max(n_total, 2))))
        for _ in range(num_iters):
            sv_hook(edges, parent)
            sv_shortcut(parent)

        parent_to_label = {}
        next_label = 0
        core_labels = np.empty(n_core, dtype=np.int32)
        for k, idx in enumerate(core_indices):
            p = parent[idx]
            if p not in parent_to_label:
                parent_to_label[p] = next_label
                next_label += 1
            core_labels[k] = parent_to_label[p]

        return core_labels

    def fit(self, X, y=None, **kwargs):
        X, _ = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n, d = X.shape
        eps = self._estimator.eps
        min_samples = self._estimator.min_samples

        D = pairwise_distance(X)
        eps_sq = eps * eps

        neighbor_mask = D <= eps_sq
        neighbor_counts = neighbor_mask.sum(axis=1)

        core_mask = neighbor_counts >= min_samples
        core_indices = np.where(core_mask)[0]
        n_core = len(core_indices)

        if n_core == 0:
            labels = -np.ones(n, dtype=np.int32)
            self._estimator.core_sample_indices_ = np.array([], dtype=np.int32)
            self._estimator.components_ = np.empty((0, d), dtype=np.float32)
            self._estimator.labels_ = labels
            self._fitted = True
            return self

        core_labels = self._gpu_sv_connected_components(core_indices, n, neighbor_mask)

        labels = -np.ones(n, dtype=np.int32)
        for k, idx in enumerate(core_indices):
            labels[idx] = core_labels[k]

        for i in range(n):
            if not core_mask[i]:
                core_neighbors = np.where(neighbor_mask[i] & core_mask)[0]
                if len(core_neighbors) > 0:
                    labels[i] = labels[core_neighbors[0]]

        self._estimator.core_sample_indices_ = core_indices.astype(np.int32)
        self._estimator.components_ = X[core_indices].copy()
        self._estimator.labels_ = labels
        self._fitted = True
        return self
