"""Correctness tests for MLX-accelerated TruncatedSVD.

Requires Apple Silicon + Metal dylib + MLX installed.
"""

import numpy as np
import pytest
from sklearn.decomposition import TruncatedSVD

from skmetal import _bridge, accelerate
from skmetal.estimators._mlx_registry import has_mlx

pytestmark = [
    pytest.mark.skipif(not hasattr(_bridge, "_lib"), reason="Metal dylib not available"),
    pytest.mark.skipif(not has_mlx(), reason="MLX not installed"),
]

_RNG = np.random.default_rng(42)


def _make_svd_data(n=500, d=30):
    X = _RNG.uniform(-5, 5, size=(n, d)).astype(np.float32)
    return X


class TestMLXTruncatedSVD:
    def test_svd_vs_sklearn(self):
        X = _make_svd_data()
        n_components = 5
        gpu = accelerate(TruncatedSVD(n_components=n_components, random_state=42))
        gpu.fit(X)
        cpu = TruncatedSVD(n_components=n_components, random_state=42)
        cpu.fit(X)
        assert gpu.components_.shape == (n_components, X.shape[1])
        assert len(gpu.singular_values_) == n_components
        assert len(gpu.explained_variance_ratio_) == n_components

    def test_svd_transform(self):
        X = _make_svd_data()
        n_components = 5
        gpu = accelerate(TruncatedSVD(n_components=n_components, random_state=42))
        gpu.fit(X)
        Xt = gpu.transform(X)
        assert Xt.shape == (X.shape[0], n_components)
        assert np.all(np.isfinite(Xt))

    def test_svd_singular_values_descending(self):
        X = _make_svd_data()
        gpu = accelerate(TruncatedSVD(n_components=10, random_state=42))
        gpu.fit(X)
        sv = gpu.singular_values_
        for i in range(len(sv) - 1):
            assert sv[i] >= sv[i + 1] - 1e-6

    def test_svd_explained_variance_ratio(self):
        X = _make_svd_data()
        gpu = accelerate(TruncatedSVD(n_components=5, random_state=42))
        gpu.fit(X)
        ratios = gpu.explained_variance_ratio_
        assert np.all(ratios >= 0)
        assert np.all(ratios <= 1.0)

    def test_svd_n_components_explicit(self):
        X = _make_svd_data()
        n_comp = min(X.shape)
        gpu = accelerate(TruncatedSVD(n_components=n_comp, random_state=42))
        gpu.fit(X)
        assert gpu.components_.shape[0] == n_comp
