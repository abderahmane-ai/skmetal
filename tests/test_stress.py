"""Edge-case and stress tests for the skmetal bridge and estimator wrappers.

Tests dtype rejection, shape validation, contiguity checks, and estimator-level
edge cases (single sample, single feature, constant data, duplicate rows).
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from skmetal import _bridge
from skmetal._bridge import gemm

pytestmark = [
    pytest.mark.skipif(
        not hasattr(_bridge, "_lib"),
        reason="SkMetalBridge dylib not available",
    ),
]


# ===========================================================================
# dtype rejection
# ===========================================================================

class TestDtypeRejection:
    def test_gemm_f64_rejected(self):
        A = np.ones((2, 3), dtype=np.float64)
        B = np.ones((3, 2), dtype=np.float64)
        with pytest.raises(TypeError, match="must be float32"):
            gemm(A, B)

    def test_gemm_mixed_dtype_rejected(self):
        A = np.ones((2, 3), dtype=np.float32)
        B = np.ones((3, 2), dtype=np.float64)
        with pytest.raises(TypeError, match="must be float32"):
            gemm(A, B)


# ===========================================================================
# Contiguity rejection
# ===========================================================================

class TestContiguityRejection:
    def test_gemm_non_contiguous_rejected(self):
        A = np.ones((2, 3), dtype=np.float32).T
        B = np.ones((3, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="C-contiguous"):
            gemm(A, B)


# ===========================================================================
# Shape validation
# ===========================================================================

class TestShapeValidation:
    def test_gemm_incompatible_dims(self):
        A = np.ones((2, 3), dtype=np.float32)
        B = np.ones((5, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="Incompatible dimensions"):
            gemm(A, B)





# ===========================================================================
# Estimator-level edge cases (via accelerate)
# ===========================================================================

class TestEstimatorEdgeCases:
    def test_single_feature_regression(self):
        X, y = make_regression(n_samples=200, n_features=1, noise=0.1, random_state=42)
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        from skmetal import accelerate
        model = accelerate(LinearRegression())
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)
        assert np.isfinite(preds).all()

    def test_two_samples_regression(self):
        X = np.array([[1.0], [2.0]], dtype=np.float32)
        y = np.array([1.0, 2.0], dtype=np.float32)
        from skmetal import accelerate
        model = accelerate(LinearRegression())
        model.fit(X, y)
        preds = model.predict(X)
        assert np.isfinite(preds).all()

    def test_constant_features_regression(self):
        X = np.ones((100, 5), dtype=np.float32)
        y = np.random.randn(100).astype(np.float32)
        from skmetal import accelerate
        model = accelerate(Ridge(alpha=1.0))
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_duplicate_rows(self):
        rng = np.random.RandomState(42)
        X = np.tile(rng.randn(1, 10).astype(np.float32), (100, 1))
        y = np.ones(100, dtype=np.float32)
        from skmetal import accelerate
        model = accelerate(LinearRegression())
        model.fit(X, y)
        preds = model.predict(X)
        assert np.allclose(preds, preds[0], rtol=1e-5)

    def test_large_p_small_n(self):
        X = np.random.randn(10, 500).astype(np.float32)
        y = np.ones(10, dtype=np.float32)
        from skmetal import accelerate
        model = accelerate(Ridge(alpha=10.0))
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (10,)

    def test_binary_classification(self):
        X, y = make_classification(n_samples=200, n_features=10, n_classes=2,
                                    random_state=42)
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        from sklearn.linear_model import LogisticRegression
        from skmetal import accelerate
        model = accelerate(LogisticRegression(random_state=42))
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)
        # Both classes should appear
        assert len(np.unique(preds)) == 2


# ===========================================================================
# Accelerator edge cases
# ===========================================================================

class TestAcceleratorEdgeCases:
    def test_accelerate_in_pipeline_works(self):
        from sklearn.pipeline import Pipeline
        from skmetal import accelerate
        pipe = accelerate(Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearRegression()),
        ]))
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        pipe.fit(X, y)
        assert pipe.predict(X).shape == (100,)

    def test_fit_twice_same_instance(self):
        from skmetal import accelerate
        model = accelerate(LinearRegression())
        X1, y1 = make_regression(n_samples=100, n_features=5, random_state=1)
        X2, y2 = make_regression(n_samples=100, n_features=5, random_state=2)
        model.fit(X1.astype(np.float32), y1.astype(np.float32))
        coef1 = model._estimator.coef_.copy()
        model.fit(X2.astype(np.float32), y2.astype(np.float32))
        coef2 = model._estimator.coef_
        assert not np.allclose(coef1, coef2), "Re-fit should change coefficients"

    def test_set_device_cpu_fallback(self):
        import skmetal
        skmetal.set_device("cpu")
        from sklearn.linear_model import LogisticRegression
        model = skmetal.accelerate(LogisticRegression(random_state=42))
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model.fit(X.astype(np.float32), y.astype(np.float32))
        assert model._fitted
        skmetal.set_device("gpu")


# ===========================================================================
# Regression: known issues that should not reappear
# ===========================================================================

class TestRegressionBugs:
    def test_binary_predict_not_all_class_zero(self):
        """Binary LogisticRegression predict should not always return class 0."""
        from sklearn.linear_model import LogisticRegression
        from skmetal import accelerate
        rng = np.random.RandomState(42)
        X = np.hstack([rng.randn(50, 5), np.ones((50, 1)) * 10])  # separable-ish
        X = X.astype(np.float32)
        y = np.array([0] * 25 + [1] * 25, dtype=np.float32)
        model = accelerate(LogisticRegression(random_state=42))
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.sum() > 0, "Binary predict should predict some class 1"
        assert preds.sum() < 50, "Binary predict should predict some class 0"
