import numpy as np
from sklearn.utils.validation import check_random_state
from ._base import BaseGPUEstimator
from .._bridge import (
    pairwise_distance,
    kmeans_assign,
    compute_mindists,
    kmeans_batch_fused,
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
        tol = self._estimator.tol
        n_init = self._estimator.n_init
        if n_init == "auto":
            n_init = 10 if n > 10000 else 5

        best_inertia = np.inf
        best_centroids = None
        best_assignments = None
        rng = check_random_state(self._estimator.random_state)
        num_groups = max(1, (n + 255) // 256)

        for _init_attempt in range(n_init):
            centroids = self._kmeans_plusplus_init(X, k, rng)
            centroids = np.ascontiguousarray(centroids, dtype=np.float32)
            assignments = np.empty(n, dtype=np.uint32)

            actual_iters = self._run_kmeans_batched(
                X, centroids, assignments, n, d, k, num_groups, max_iter, tol
            )

            diff = X - centroids[assignments]
            inertia = float(np.sum(diff * diff))
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()
                best_assignments = assignments.copy()

        self._estimator.cluster_centers_ = best_centroids
        self._estimator.labels_ = best_assignments.astype(np.int32)
        self._estimator.inertia_ = float(best_inertia)
        self._estimator.n_iter_ = actual_iters
        self._estimator.n_features_in_ = d
        self._fitted = True
        return self

    def _run_kmeans_batched(self, X, centroids, assignments,
                              n, d, k, num_groups, max_iter, tol):
        batch_size = max(1, min(5, (max_iter + 9) // 10))
        total_iters = 0
        for batch_start in range(0, max_iter, batch_size):
            remaining = min(batch_size, max_iter - batch_start)
            old = centroids.copy()
            kmeans_batch_fused(X, centroids, assignments,
                               n, d, k, num_groups, remaining)
            total_iters += remaining
            shift = np.sqrt(np.square(centroids - old).sum(axis=1)).max()
            if shift < tol:
                break
        return total_iters

    def _kmeans_plusplus_init(self, X, k, rng):
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

        # For high-D data, sklearn's tree-based DBSCAN is faster than GPU O(n²)
        if d > 6:
            return self._fallback_fit(X, y, **kwargs)

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
