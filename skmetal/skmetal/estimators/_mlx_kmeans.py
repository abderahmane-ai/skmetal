"""MLX-accelerated KMeans — uses MLX GPU ops for Lloyd iterations.

Requires ``pip install skmetal[mlx]`` (``mlx>=0.31``).
Falls back to ``MetalKMeans`` (Metal bridge) or sklearn CPU automatically.
"""

import numpy as np
from ._base import BaseGPUEstimator

_HAS_MLX = False
try:
    import mlx.core as mx

    _HAS_MLX = True
except ImportError:
    pass


if _HAS_MLX:

    def _centroid_update_scatter(X: mx.array, labels: mx.array, k: int) -> mx.array:
        """Update centroids via scatter-add (avoids O(K·N) one-hot matrix)."""
        d = X.shape[1]
        ids = labels.astype(mx.uint32)
        n = X.shape[0]
        sums = mx.zeros((k, d), dtype=mx.float32)
        sums = sums.at[ids].add(X)
        counts = mx.zeros((k,), dtype=mx.float32)
        counts = counts.at[ids].add(mx.ones(n, dtype=mx.float32))
        return sums / mx.maximum(counts[:, None], 1.0)

    @mx.compile
    def _fused_lloyd_iter(X: mx.array, centroids: mx.array) -> tuple[mx.array, mx.array]:
        k = centroids.shape[0]
        x_norm = (X * X).sum(axis=1, keepdims=True)
        c_norm = (centroids * centroids).sum(axis=1)
        dists = x_norm + c_norm - 2.0 * (X @ centroids.T)
        labels = mx.argmin(dists, axis=1)
        new_centroids = _centroid_update_scatter(X, labels, k)
        return new_centroids, labels

    @mx.compile
    def _fused_lloyd_batch_10(X: mx.array, centroids: mx.array) -> tuple[mx.array, mx.array]:
        """Run 10 Lloyd iterations in a single compiled graph. 10x fewer CPU round-trips."""
        k = centroids.shape[0]
        x_norm = (X * X).sum(axis=1, keepdims=True)
        for _ in range(10):
            c_norm = (centroids * centroids).sum(axis=1)
            dists = x_norm + c_norm - 2.0 * (X @ centroids.T)
            labels = mx.argmin(dists, axis=1)
            centroids = _centroid_update_scatter(X, labels, k)
        return centroids, labels

    class MetalKMeansMLX(BaseGPUEstimator):
        def fit(self, X, y=None, **kwargs):
            X, _ = self._validate_data(X, y)
            if not self._should_use_gpu(X):
                return self._fallback_fit(X, y, **kwargs)

            n, d = X.shape
            k = self._estimator.n_clusters
            max_iter = self._estimator.max_iter
            tol = self._estimator.tol
            n_init = self._estimator.n_init
            init = getattr(self._estimator, "init", "k-means++")

            if n_init == "auto":
                n_init = 1 if init == "k-means++" else (10 if n > 10000 else 5)

            X_mx = mx.array(X)

            best_inertia = np.inf
            best_centroids = None
            best_labels = None
            best_n_iter = max_iter

            rng = np.random.RandomState(self._estimator.random_state)

            for _ in range(n_init):
                centroids_mx = self._mlx_init(X_mx, k, init, rng, n)
                centroids_mx, labels_mx, n_iter = self._mlx_lloyd_loop(
                    X_mx, centroids_mx, k, max_iter, tol
                )
                inertia = self._mlx_inertia(X_mx, centroids_mx, labels_mx)

                if inertia < best_inertia:
                    best_inertia = inertia
                    best_centroids = centroids_mx
                    best_labels = labels_mx
                    best_n_iter = n_iter

            self._estimator.cluster_centers_ = np.array(best_centroids)
            self._estimator.labels_ = np.array(best_labels).astype(np.int32)
            self._estimator.inertia_ = best_inertia
            self._estimator.n_iter_ = best_n_iter
            self._estimator.n_features_in_ = d
            self._fitted = True
            return self

        def _mlx_init(self, X, k, init, rng, n):
            if init == "random":
                idx = rng.choice(n, size=k, replace=False)
                return mx.array(np.ascontiguousarray(X[idx]))
            # k-means++
            centroids = mx.zeros((k, X.shape[1]), dtype=X.dtype)
            centroids[0] = X[rng.randint(n)]
            for c in range(1, k):
                x_norm = (X * X).sum(axis=1, keepdims=True)
                c_norm = (centroids[:c] * centroids[:c]).sum(axis=1)
                dot = X @ centroids[:c].T
                d2 = x_norm + c_norm - 2.0 * dot
                min_d2 = d2.min(axis=1)
                probs = np.array(min_d2 / min_d2.sum()).ravel()
                probs = np.maximum(probs, 1e-30)
                idx = rng.choice(n, p=probs / probs.sum())
                centroids[c] = X[idx]
            return centroids

        def _mlx_lloyd_loop(self, X, centroids, k, max_iter, tol):
            i = 0
            while i + 10 <= max_iter:
                old = centroids
                centroids, labels = _fused_lloyd_batch_10(X, centroids)
                shift = mx.sqrt(((centroids - old) ** 2).sum(axis=1)).max()
                if shift.item() < tol:
                    break
                i += 10
            # Tail loop: run up to 10 single iterations to catch convergence
            # between batches (when max_iter % 10 == 0).
            for _ in range(10):
                if i >= max_iter:
                    break
                old = centroids
                centroids, labels = _fused_lloyd_iter(X, centroids)
                shift = mx.sqrt(((centroids - old) ** 2).sum(axis=1)).max()
                i += 1
                if shift.item() < tol:
                    break
            return centroids, labels, min(i, max_iter)

        def _mlx_inertia(self, X, centroids, labels):
            x_norm = (X * X).sum(axis=1)
            c_norm = (centroids * centroids).sum(axis=1)
            assigned_c_norm = c_norm[labels]
            dot = (X * centroids[labels]).sum(axis=1)
            return float((x_norm + assigned_c_norm - 2.0 * dot).sum().item())

        def predict(self, X):
            X = self._validate_data(X)[0]
            if not self._should_use_gpu(X) or not self._fitted:
                return self._fallback_predict(X)
            X_mx = mx.array(X)
            centroids_mx = mx.array(self._estimator.cluster_centers_)
            x_norm = (X_mx * X_mx).sum(axis=1, keepdims=True)
            c_norm = (centroids_mx * centroids_mx).sum(axis=1)
            dists = x_norm + c_norm - 2.0 * (X_mx @ centroids_mx.T)
            labels = mx.argmin(dists, axis=1)
            return np.array(labels).astype(np.int32)

        def transform(self, X):
            X = self._validate_data(X)[0]
            if not self._should_use_gpu(X) or not self._fitted:
                return self._fallback_transform(X)
            X_mx = mx.array(X)
            centroids_mx = mx.array(self._estimator.cluster_centers_)
            x_norm = (X_mx * X_mx).sum(axis=1, keepdims=True)
            c_norm = (centroids_mx * centroids_mx).sum(axis=1)
            dists = x_norm + c_norm - 2.0 * (X_mx @ centroids_mx.T)
            return np.sqrt(np.maximum(np.array(dists), 0.0))

        def _kmeans_cpu_fallback(self, X, k, max_iter, tol, rng):
            from sklearn.cluster import KMeans as SklearnKMeans
            km = SklearnKMeans(
                n_clusters=k, init="k-means++", n_init=1,
                max_iter=max_iter, tol=tol, random_state=rng,
            )
            km.fit(X)
            return km.cluster_centers_.astype(np.float32), km.labels_.astype(np.uint32)
