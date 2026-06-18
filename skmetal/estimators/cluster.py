import numpy as np
from sklearn.utils.validation import check_random_state
from ._base import BaseGPUEstimator
from .._bridge import (
    gemm,
    pairwise_distance,
    kmeans_assign,
    kmeans_inertia,
    compute_mindists,
    kmeans_batch_fused,
    sv_init,
    sv_hook,
    sv_shortcut,
)
from .._config import get_config


class MetalKMeans(BaseGPUEstimator):
    """GPU-accelerated KMeans via fused command buffer (Lloyd iterations on GPU)."""
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
            init = getattr(self._estimator, "init", "k-means++")
            n_init = 1 if init == "k-means++" else (10 if n > 10000 else 5)

        best_inertia = np.inf
        best_centroids = None
        best_assignments = None
        rng = check_random_state(self._estimator.random_state)
        num_groups = max(1, (n + 255) // 256)

        for _init_attempt in range(n_init):
            centroids = self._kmeans_parallel_init(X, k, rng)
            centroids = np.ascontiguousarray(centroids, dtype=np.float32)
            assignments = np.empty(n, dtype=np.uint32)

            actual_iters = self._run_kmeans_batched(X, centroids, assignments, n, d, k, num_groups, max_iter, tol)

            if assignments.min() < 0 or assignments.max() >= k:
                if get_config().verbose:
                    print("[skmetal] KMeans: GPU output invalid, falling back to CPU")
                centroids, assignments = self._kmeans_cpu_fallback(X, k, max_iter, tol, rng)
                actual_iters = max_iter

            inertia = kmeans_inertia(X, centroids, assignments, n, d, k)
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

    def _run_kmeans_batched(self, X, centroids, assignments, n, d, k, num_groups, max_iter, tol):
        return kmeans_batch_fused(X, centroids, assignments, n, d, k, num_groups, max_iter, tol)

    def _kmeans_parallel_init(self, X, k, rng):
        """k-means|| (parallel) initialization — O(log k) rounds vs k serial rounds.

        Each round samples O(k) candidates proportional to distance² using GPU
        distance computation. Final set of ~k·log(k) candidates is pruned to k
        via a single weighted k-means iteration on CPU.
        """
        n, d = X.shape
        centroids = np.empty((k, d), dtype=np.float32)
        centroids[0] = X[rng.randint(n)]

        assignments = np.empty(n, dtype=np.uint32)
        dists = np.empty(n, dtype=np.float32)

        n_rounds = max(1, int(2 + np.log(k)))
        candidates = [centroids[0]]
        candidate_set = {0}  # track indices already added

        for _ in range(n_rounds):
            c_arr = np.ascontiguousarray(np.array(candidates, dtype=np.float32))
            ck = c_arr.shape[0]

            kmeans_assign(X, c_arr, assignments, n, d, ck)
            compute_mindists(X, c_arr, assignments, dists, n, d, ck)
            dists = np.maximum(dists, 1e-30)
            probs = dists / dists.sum()

            n_sample = min(k, n)
            idx = rng.choice(n, size=n_sample, p=probs, replace=False)
            for i in idx:
                if i not in candidate_set:
                    candidate_set.add(i)
                    candidates.append(X[i])

        # Prune candidates to k via weighted k-means (1 iteration on CPU)
        all_cand = np.array(candidates, dtype=np.float32)
        if all_cand.shape[0] <= k:
            centroids[: all_cand.shape[0]] = all_cand
            for i in range(all_cand.shape[0], k):
                centroids[i] = candidates[i % len(candidates)]
            return centroids

        from sklearn.cluster import KMeans as SklearnKMeans

        km = SklearnKMeans(n_clusters=k, init=all_cand[:k], n_init=1, max_iter=1, random_state=rng)
        km.fit(all_cand)
        return km.cluster_centers_.astype(np.float32)

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        n, d = X.shape
        k = self._estimator.cluster_centers_.shape[0]
        assignments = np.empty(n, dtype=np.uint32)
        kmeans_assign(X, self._estimator.cluster_centers_, assignments, n, d, k)
        return assignments.astype(np.int32)

    def _kmeans_cpu_fallback(self, X, k, max_iter, tol, rng):
        from sklearn.cluster import KMeans as SklearnKMeans

        km = SklearnKMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=max_iter, tol=tol, random_state=rng)
        km.fit(X)
        return km.cluster_centers_.astype(np.float32), km.labels_.astype(np.uint32)

    def transform(self, X):
        """Return distances from each sample to each cluster centre (n_samples × n_clusters)."""
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_transform(X)
        centers = self._estimator.cluster_centers_  # (k, d)
        # ||x_i - c_j||^2 = ||x_i||^2 + ||c_j||^2 - 2 * x_i · c_j
        X_norms = np.einsum("ij,ij->i", X, X)[:, np.newaxis]  # (n, 1)
        C_norms = np.einsum("ij,ij->i", centers, centers)[np.newaxis, :]  # (1, k)
        cross = gemm(X, np.ascontiguousarray(centers.T))  # (n, k)
        dists_sq = np.maximum(X_norms + C_norms - 2.0 * cross, 0.0)
        return np.sqrt(dists_sq)


class MetalDBSCAN(BaseGPUEstimator):
    """GPU-accelerated DBSCAN via Shiloach-Vishkin connected components on GPU."""
    def _should_use_gpu(self, X):
        if not super()._should_use_gpu(X):
            return False
        if X.shape[1] > 6:
            if get_config().verbose:
                print(f"[skmetal] DBSCAN: d={X.shape[1]} > 6, tree-based CPU faster than GPU O(n²)")
            return False
        return True

    def _gpu_sv_connected_components(self, core_indices, n_total, neighbor_mask):
        """GPU Shiloach-Vishkin connected components on the core-point subgraph.

        Returns core_labels: int32 array of length len(core_indices) with
        consecutive cluster labels 0..n_comp-1.
        """
        n_core = len(core_indices)
        if n_core <= 1:
            return np.zeros(n_core, dtype=np.int32)

        # Vectorised edge extraction — avoids O(n_core²) Python loop.
        # sub[i,j] == True means core_indices[i] is a neighbour of core_indices[j]
        sub = neighbor_mask[np.ix_(core_indices, core_indices)]
        # Upper-triangular pairs only (undirected graph)
        r, c = np.where(np.triu(sub, k=1))
        if r.size == 0:
            return np.zeros(n_core, dtype=np.int32)
        # Map local indices back to global node ids
        src = core_indices[r].astype(np.int32)
        dst = core_indices[c].astype(np.int32)
        edges = np.empty(src.size + dst.size, dtype=np.int32)
        edges[0::2] = src
        edges[1::2] = dst

        parent = np.empty(n_total, dtype=np.int32)
        sv_init(parent)

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
