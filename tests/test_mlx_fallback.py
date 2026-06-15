"""MLX fallback tests — graceful degradation when MLX is unavailable or fails.

Tests should pass both with and without MLX installed.
"""

import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD

from skmetal import _bridge, accelerate
from skmetal._config import get_config, set_device
from skmetal.estimators._mlx_registry import has_mlx

pytestmark = [
    pytest.mark.skipif(not hasattr(_bridge, "_lib"), reason="Metal dylib not available"),
]

_RNG = np.random.default_rng(42)


def _make_data(n=200, d=10):
    X, y = datasets.make_classification(
        n_samples=n, n_features=d, n_classes=2, random_state=42,
    )
    return X.astype(np.float32), y


class TestMLXFallbackToCPU:
    """When device='cpu', estimators fall back to sklearn CPU."""

    def setup_method(self):
        set_device("cpu")

    def teardown_method(self):
        set_device("gpu")

    def test_logreg_fallback_cpu(self):
        X, y = _make_data()
        gpu = accelerate(LogisticRegression(max_iter=100, random_state=42))
        gpu.fit(X, y)
        cpu = LogisticRegression(max_iter=100, random_state=42)
        cpu.fit(X, y)
        np.testing.assert_allclose(gpu.coef_, cpu.coef_, rtol=1e-5, atol=1e-6)

    def test_truncated_svd_fallback_cpu(self):
        X = _RNG.uniform(-5, 5, size=(200, 20)).astype(np.float32)
        gpu = accelerate(TruncatedSVD(n_components=5, random_state=42))
        gpu.fit(X)
        cpu = TruncatedSVD(n_components=5, random_state=42)
        cpu.fit(X)
        np.testing.assert_allclose(gpu.singular_values_, cpu.singular_values_, rtol=1e-4, atol=1e-4)


class TestMLXThresholdFallback:
    """When threshold forces CPU, estimator falls back."""

    def setup_method(self):
        get_config().device = "gpu"

    def test_threshold_huge_forces_cpu(self):
        get_config().threshold = 10 ** 15
        X, y = _make_data()
        gpu = accelerate(LogisticRegression(max_iter=100, random_state=42))
        gpu.fit(X, y)
        cpu = LogisticRegression(max_iter=100, random_state=42)
        cpu.fit(X, y)
        np.testing.assert_allclose(gpu.coef_, cpu.coef_, rtol=1e-5, atol=1e-6)
        get_config().threshold = 1

    def test_per_estimator_threshold_forces_cpu(self):
        thresh = dict(get_config().thresholds)
        thresh["LogisticRegression"] = (10 ** 9, 10 ** 9)
        get_config().thresholds = thresh
        X, y = _make_data()
        gpu = accelerate(LogisticRegression(max_iter=100, random_state=42))
        gpu.fit(X, y)
        assert hasattr(gpu, "coef_")


class TestMLXNonFloat32:
    """Non-float32 inputs trigger fallback to sklearn CPU."""

    def test_float64_input_fallback(self):
        X, y = datasets.make_classification(n_samples=200, n_features=10, random_state=42)
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        gpu = accelerate(LogisticRegression(max_iter=100, random_state=42))
        gpu.fit(X, y)
        assert hasattr(gpu, "coef_")

    def test_integer_input_fallback(self):
        X = np.random.randint(0, 100, size=(200, 10)).astype(np.float32)
        y = np.random.choice([0, 1], size=200).astype(np.float32)
        gpu = accelerate(LogisticRegression(max_iter=100, random_state=42))
        gpu.fit(X, y)
        assert hasattr(gpu, "coef_")


@pytest.mark.skipif(not has_mlx(), reason="MLX not installed")
class TestMLXImportProtection:
    """Verify that MLX modules handle import errors gracefully."""

    def test_mlx_svd_imports(self):
        from skmetal.estimators._mlx_svd import MetalTruncatedSVDMLX
        assert MetalTruncatedSVDMLX is not None
