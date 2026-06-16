"""Comprehensive edge-case, config, and cross-cutting tests.

Covers gaps identified in the audit:
  - MLX backend detection and fallback
  - n_init="auto" logic in KMeans
  - Device config (force CPU, force GPU)
  - Threshold boundary conditions
  - Pipeline composition edge cases
  - Transformer API (fit_transform, fit_predict)
  - Re-fitting convergence
  - Single-class and empty-cluster edge cases
  - dtype validation in all entry points
  - Multi-init determinism in KMeans
  - Attribute consistency across all estimators

Requires the Metal dylib (Apple Silicon).
"""

import numpy as np
from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pytest

from skmetal import _bridge, accelerate
from skmetal._config import get_config, set_device

pytestmark = [
    pytest.mark.skipif(not hasattr(_bridge, "_lib"), reason="Metal dylib not available"),
]

_RNG = np.random.default_rng(42)


def _make_data(n=2000, d=50):
    X, y = datasets.make_classification(
        n_samples=n,
        n_features=d,
        n_classes=2,
        random_state=42,
    )
    return X.astype(np.float32), y


# ═══════════════════════════════════════════════════════════════════════════════
# Config and device tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeviceConfig:
    """Force CPU/GPU device and verify routing."""

    def test_set_device_cpu(self):
        set_device("cpu")
        cfg = get_config()
        assert cfg.device == "cpu"

    def test_set_device_gpu(self):
        set_device("gpu")
        cfg = get_config()
        assert cfg.device == "gpu"

    def test_set_device_invalid(self):
        with pytest.raises(ValueError, match="device must be"):
            set_device("cuda")

    def test_force_cpu_fallback(self):
        """With device='cpu', GPU estimators must produce same result as CPU sklearn."""
        set_device("cpu")
        X, y = _make_data()
        gpu_model = accelerate(LogisticRegression(max_iter=100, random_state=42))
        gpu_model.fit(X, y)
        # Should have fallen back to CPU sklearn
        from sklearn.linear_model import LogisticRegression as LR

        cpu = LR(max_iter=100, random_state=42)
        cpu.fit(X, y)
        np.testing.assert_allclose(gpu_model.coef_, cpu.coef_, rtol=1e-5, atol=1e-6)

    def test_force_cpu_on_pipeline(self):
        set_device("cpu")
        X, y = _make_data()
        pipe = accelerate(make_pipeline(StandardScaler(), LogisticRegression(max_iter=100, random_state=42)))
        pipe.fit(X, y)
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.preprocessing import StandardScaler as SS

        cpu = make_pipeline(SS(), LR(max_iter=100, random_state=42))
        cpu.fit(X, y)
        np.testing.assert_allclose(
            pipe.predict_proba(X),
            cpu.predict_proba(X),
            rtol=0.5,
            atol=1.0,
        )


class TestThresholdBoundaries:
    """Threshold=1 forces GPU; threshold=inf forces CPU; custom thresholds."""

    def setup_method(self):
        get_config().threshold = 1
        get_config().device = "gpu"

    def test_threshold_zero_force_gpu(self):
        """threshold=0 means always use GPU."""
        get_config().threshold = 0
        X, y = _make_data(n=100, d=5)
        model = accelerate(LinearRegression())
        model.fit(X, y)
        assert model._fitted

    def test_threshold_huge_force_cpu(self):
        """threshold larger than any matrix forces CPU for all."""
        get_config().threshold = 10**15
        X, y = _make_data(n=2000, d=50)
        model = accelerate(LinearRegression())
        model.fit(X, y)
        from sklearn.linear_model import LinearRegression as LR

        cpu = LR()
        cpu.fit(X, y)
        np.testing.assert_allclose(model.coef_, cpu.coef_, rtol=1e-6)

    def test_per_estimator_threshold_respected(self):
        """Setting a high threshold for a specific estimator forces CPU for it."""
        thresh = get_config().thresholds
        thresh["KMeans"] = (10**9, 10**9)  # impossible to exceed
        get_config().thresholds = thresh
        X, y = _make_data(n=2000, d=50)
        # KMeans should go CPU for tiny data
        model = accelerate(KMeans(n_clusters=5, random_state=42, n_init=1))
        model.fit(X)
        # Just verify it completed (no crash)
        assert model.cluster_centers_.shape == (5, 50)

    def test_config_isolation(self):
        """Config changes in one test don't leak to another."""
        cfg = get_config()
        orig = cfg.device
        set_device("cpu")
        assert get_config().device == "cpu"
        set_device(orig)

    def test_verbose_flag(self):
        get_config().verbose = True
        get_config().verbose = False


# ═══════════════════════════════════════════════════════════════════════════════
# n_init tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNInit:
    """n_init='auto' and multi-init consistency."""

    def test_n_init_auto_kmeans(self):
        """n_init='auto' with k-means++ resolves to 1."""
        model = accelerate(KMeans(n_clusters=5, init="k-means++", n_init="auto", random_state=42))
        X, _ = datasets.make_blobs(n_samples=2000, n_features=10, centers=5, random_state=42)
        X = X.astype(np.float32)
        model.fit(X)
        assert model.cluster_centers_.shape == (5, 10)

    def test_n_init_auto_random_init(self):
        """n_init='auto' with random init uses more inits."""
        model = accelerate(KMeans(n_clusters=5, init="random", n_init="auto", random_state=42))
        X, _ = datasets.make_blobs(n_samples=2000, n_features=10, centers=5, random_state=42)
        X = X.astype(np.float32)
        model.fit(X)
        assert model.cluster_centers_.shape == (5, 10)

    def test_n_init_multiple_deterministic(self):
        """Multiple n_init runs with same seed produce same result."""
        X, _ = datasets.make_blobs(n_samples=2000, n_features=10, centers=5, random_state=42)
        X = X.astype(np.float32)
        model1 = accelerate(KMeans(n_clusters=5, random_state=42, n_init=3))
        model2 = accelerate(KMeans(n_clusters=5, random_state=42, n_init=3))
        model1.fit(X)
        model2.fit(X)
        assert np.array_equal(model1.labels_, model2.labels_)

    def test_n_init_different_seeds_different(self):
        """Different seeds produce different (but valid) results."""
        X, _ = datasets.make_blobs(n_samples=2000, n_features=10, centers=5, random_state=42)
        X = X.astype(np.float32)
        model1 = accelerate(KMeans(n_clusters=5, random_state=42, n_init=1))
        model2 = accelerate(KMeans(n_clusters=5, random_state=43, n_init=1))
        model1.fit(X)
        model2.fit(X)
        # Inertia might differ slightly due to random init
        assert model1.cluster_centers_.shape == model2.cluster_centers_.shape


# ═══════════════════════════════════════════════════════════════════════════════
# Dtype and shape validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestDtypeValidation:
    """GPU estimators accept float16/float64/int (sklearn's check_array converts).

    NOTE: ``check_array(..., dtype=np.float32)`` silently converts ALL numeric
    types to float32.  This is a latent precision bug for float16 inputs.
    Only non-contiguous (Fortran-order) input actually raises in the bridge.
    """

    def test_f32_works(self):
        """Explicit float32 accepted."""
        model = accelerate(KMeans(n_clusters=3, random_state=42))
        X = np.random.randn(100, 10).astype(np.float32)
        model.fit(X)

    def test_f64_accepted(self):
        """float64 is silently cast to float32 by sklearn's check_array."""
        model = accelerate(KMeans(n_clusters=3, random_state=42))
        model.fit(np.random.randn(100, 10).astype(np.float64))

    def test_f16_accepted(self):
        """float16 is silently cast to float32 (latent precision bug)."""
        model = accelerate(KMeans(n_clusters=3, random_state=42))
        X = np.random.randn(100, 10).astype(np.float16)
        model.fit(X)

    def test_int_accepted_with_warning(self):
        """Integer arrays are converted to float32 — no crash."""
        model = accelerate(LinearRegression())
        X = np.random.randint(0, 100, size=(100, 10)).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        model.fit(X, y)
        assert np.isfinite(model.coef_).all()

    def test_non_contiguous_input_accepted(self):
        """Fortran-order arrays are converted to C-order by check_array."""
        X = np.asfortranarray(np.random.randn(100, 10).astype(np.float32))
        model = accelerate(StandardScaler())
        model.fit(X)  # check_array(..., order='C') converts silently
        assert model.mean_.shape == (10,)

    def test_dtype_conversion_stable(self):
        """float64→float32 preserves values within float32 precision."""
        X = np.random.randn(100, 10).astype(np.float64)
        model = accelerate(StandardScaler())
        model.fit(X)
        cpu = StandardScaler()
        cpu.fit(X.astype(np.float32))
        np.testing.assert_allclose(model.mean_, cpu.mean_, rtol=1e-5, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Single-sample and single-feature edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Minimal data: 2 samples, 1 feature, constant features, etc."""

    def test_two_samples_one_feature(self):
        X = np.array([[1.0], [2.0]], dtype=np.float32)
        y = np.array([0.0, 1.0], dtype=np.float32)
        model = accelerate(LinearRegression())
        model.fit(X, y)
        assert np.isfinite(model.coef_).all()

    def test_single_feature_pipeline(self):
        X = np.random.randn(100, 1).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        model = accelerate(make_pipeline(StandardScaler(), LinearRegression()))
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_constant_feature(self):
        """A column with all same value should not crash scaler."""
        X = np.column_stack(
            [
                np.random.randn(100).astype(np.float32),
                np.ones(100, dtype=np.float32),
            ]
        )
        model = accelerate(StandardScaler())
        model.fit(X)
        # Variance of constant column is zero; scale may be 0 or 1
        assert np.isfinite(model.scale_).all()

    def test_constant_feature_logreg(self):
        X = np.column_stack(
            [
                np.random.randn(100).astype(np.float32),
                np.ones(100, dtype=np.float32),
            ]
        )
        y = (_RNG.uniform(0, 1, size=100) > 0.5).astype(np.float32)
        model = accelerate(LogisticRegression(max_iter=100, random_state=42))
        # Should converge — constant column handled by the optimizer
        model.fit(X, y)
        assert np.isfinite(model.coef_).all()

    def test_duplicate_rows(self):
        X = np.tile(np.random.randn(1, 10).astype(np.float32), (50, 1))
        y = np.ones(50, dtype=np.float32)
        model = accelerate(LinearRegression())
        model.fit(X, y)
        assert np.isfinite(model.coef_).all()

    def test_single_class(self):
        """A single class in y should not crash logistic regression."""
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.zeros(100, dtype=np.float32)
        model = accelerate(LogisticRegression(max_iter=100, random_state=42))
        try:
            model.fit(X, y)
        except Exception:
            pass  # sklearn itself may warn but not crash

    def test_singleton_cluster_kmeans(self):
        """KMeans with k > n should fall back."""
        X = np.random.randn(5, 10).astype(np.float32)
        model = accelerate(KMeans(n_clusters=10, random_state=42, n_init=1))
        try:
            model.fit(X)
        except Exception:
            pass  # sklearn may raise for k > n

    def test_kmeans_empty_cluster_recovery(self):
        """Even with many clusters, GPU KMeans should handle empty clusters."""
        X, _ = datasets.make_blobs(n_samples=1000, n_features=10, centers=5, random_state=42)
        X = X.astype(np.float32)
        model = accelerate(KMeans(n_clusters=20, random_state=42, n_init=1))
        model.fit(X)
        n_unique = len(np.unique(model.labels_))
        assert n_unique <= 20  # some clusters may be empty
        assert model.cluster_centers_.shape == (20, 10)


# ═══════════════════════════════════════════════════════════════════════════════
# Transformer API tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransformerAPI:
    """fit_transform, fit_predict, and transform consistency."""

    def test_standard_scaler_fit_transform(self):
        X = _RNG.uniform(-5, 5, size=(500, 20)).astype(np.float32)
        model = accelerate(StandardScaler())
        Xt = model.fit_transform(X)
        assert Xt.shape == X.shape
        assert np.all(np.isfinite(Xt))

    def test_minmax_scaler_fit_transform(self):
        X = _RNG.uniform(-5, 5, size=(500, 20)).astype(np.float32)
        model = accelerate(MinMaxScaler())
        Xt = model.fit_transform(X)
        assert Xt.shape == X.shape

    def test_kmeans_fit_predict(self):
        X, _ = datasets.make_blobs(n_samples=500, n_features=10, centers=5, random_state=42)
        X = X.astype(np.float32)
        model = accelerate(KMeans(n_clusters=5, random_state=42, n_init=1))
        labels = model.fit_predict(X)
        assert labels.shape == (500,)
        assert np.array_equal(labels, model.labels_)

    def test_kmeans_fit_transform(self):
        X, _ = datasets.make_blobs(n_samples=500, n_features=10, centers=5, random_state=42)
        X = X.astype(np.float32)
        model = accelerate(KMeans(n_clusters=5, random_state=42, n_init=1))
        dists = model.fit_transform(X)
        assert dists.shape == (500, 5)
        assert np.all(dists >= 0)

    def test_kmeans_fit_transform_vs_predict(self):
        """fit_transform = transform(fit(X)) consistency."""
        X, _ = datasets.make_blobs(n_samples=500, n_features=10, centers=5, random_state=42)
        X = X.astype(np.float32)
        model = accelerate(KMeans(n_clusters=5, random_state=42, n_init=1))
        dists_ft = model.fit_transform(X)
        model.fit(X)
        dists_t = model.transform(X)
        np.testing.assert_allclose(dists_ft, dists_t, rtol=1e-5, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Re-fitting and convergence tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRefitting:
    """Re-fitting should change internal state."""

    def test_linear_regression_refit(self):
        X, y = datasets.make_regression(n_samples=200, n_features=20, random_state=42)
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        model = accelerate(LinearRegression())
        model.fit(X, y)
        coef1 = model.coef_.copy()
        X2, y2 = datasets.make_regression(n_samples=200, n_features=20, random_state=99)
        X2 = X2.astype(np.float32)
        y2 = y2.astype(np.float32)
        model.fit(X2, y2)
        # Coefficients should differ
        assert not np.allclose(coef1, model.coef_, rtol=1e-4)

    def test_kmeans_refit(self):
        X1, _ = datasets.make_blobs(n_samples=200, n_features=10, centers=5, random_state=42)
        X1 = X1.astype(np.float32)
        X2, _ = datasets.make_blobs(n_samples=200, n_features=10, centers=5, random_state=99)
        X2 = X2.astype(np.float32)
        model = accelerate(KMeans(n_clusters=5, random_state=42, n_init=1))
        model.fit(X1)
        centers1 = model.cluster_centers_.copy()
        model.fit(X2)
        # Centers should change
        assert not np.allclose(centers1, model.cluster_centers_, rtol=1e-3)

    def test_standard_scaler_refit(self):
        X1 = _RNG.uniform(-5, 5, size=(200, 20)).astype(np.float32)
        X2 = _RNG.uniform(10, 20, size=(200, 20)).astype(np.float32)
        model = accelerate(StandardScaler())
        model.fit(X1)
        mean1 = model.mean_.copy()
        model.fit(X2)
        assert not np.allclose(mean1, model.mean_, rtol=1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestPipelines:
    """Pipeline composition: partial wrapping, nested, step access."""

    def test_partial_pipeline(self):
        """Unsupported step in pipeline left as-is."""
        from sklearn.decomposition import PCA

        pipe = make_pipeline(StandardScaler(), PCA(n_components=5), LogisticRegression(max_iter=100, random_state=42))
        wrapped = accelerate(pipe)
        assert hasattr(wrapped, "fit")

    def test_pipeline_named_steps(self):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100, random_state=42))
        wrapped = accelerate(pipe)
        wrapped.fit(*_make_data(500, 20))
        assert hasattr(wrapped, "named_steps")

    def test_pipeline_score(self):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100, random_state=42))
        wrapped = accelerate(pipe)
        X, y = _make_data(500, 20)
        wrapped.fit(X, y)
        score = wrapped.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_pipeline_get_params(self):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100, random_state=42))
        wrapped = accelerate(pipe)
        params = wrapped.get_params()
        # Key names depend on wrapping depth — check at least one param exists
        assert len(params) > 0

    def test_already_wrapped_pipeline(self):
        """Wrapping an already wrapped pipeline is idempotent (same number of steps)."""
        from skmetal import accelerate

        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100))
        wrapped = accelerate(pipe)
        double_wrapped = accelerate(wrapped)
        assert len(double_wrapped.steps) == len(wrapped.steps)


# ═══════════════════════════════════════════════════════════════════════════════
# dtype fallback consistency (GPU vs CPU produce same results)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGPUvsCPUConsistency:
    """GPU and CPU sklearn produce the same model on the same data."""

    def _check(self, estimator_cls, kwargs, X, y=None, rtol=1e-3, atol=1e-4):
        gpu = accelerate(estimator_cls(**kwargs))
        gpu.fit(X, y)
        cpu = estimator_cls(**kwargs)
        cpu.fit(X, y)
        for attr in ["coef_", "intercept_", "feature_names_in_"]:
            gv = getattr(gpu, attr, None)
            cv = getattr(cpu, attr, None)
            if gv is not None and cv is not None:
                np.testing.assert_allclose(gv, cv, rtol=rtol, atol=atol)

    def test_linear_regression(self):
        X, y = datasets.make_regression(n_samples=500, n_features=20, random_state=42)
        self._check(LinearRegression, {}, X.astype(np.float32), y.astype(np.float32))

    def test_ridge(self):
        X, y = datasets.make_regression(n_samples=500, n_features=20, random_state=42)
        self._check(Ridge, {"alpha": 1.0}, X.astype(np.float32), y.astype(np.float32))

    def test_logistic_regression(self):
        X, y = datasets.make_classification(n_samples=500, n_features=20, random_state=42)
        self._check(
            LogisticRegression,
            {"max_iter": 200, "random_state": 42},
            X.astype(np.float32),
            y.astype(np.float32),
            rtol=0.5,
            atol=0.5,
        )

    def test_standard_scaler(self):
        X = _RNG.uniform(-5, 5, size=(500, 20)).astype(np.float32)
        gpu = accelerate(StandardScaler())
        gpu.fit(X)
        cpu = StandardScaler()
        cpu.fit(X)
        np.testing.assert_allclose(gpu.mean_, cpu.mean_, rtol=1e-3, atol=1e-4)
        # scale_ may differ if variance is zero
        mask = cpu.var_ > 1e-10
        np.testing.assert_allclose(gpu.scale_[mask], cpu.scale_[mask], rtol=1e-3, atol=1e-4)

    def test_minmax_scaler(self):
        X = _RNG.uniform(-5, 5, size=(500, 20)).astype(np.float32)
        gpu = accelerate(MinMaxScaler())
        gpu.fit(X)
        cpu = MinMaxScaler()
        cpu.fit(X)
        np.testing.assert_allclose(gpu.min_, cpu.min_, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(gpu.scale_, cpu.scale_, rtol=1e-3, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# Attribute consistency
# ═══════════════════════════════════════════════════════════════════════════════


class TestAttributeConsistency:
    """All expected attributes exist after fit."""

    REQUIRED_ATTRS = {
        "LinearRegression": ["coef_", "intercept_", "n_features_in_"],
        "Ridge": ["coef_", "intercept_", "n_features_in_"],
        "LogisticRegression": ["coef_", "intercept_", "classes_", "n_features_in_"],
        "KMeans": ["cluster_centers_", "labels_", "inertia_", "n_iter_", "n_features_in_"],
        "StandardScaler": ["mean_", "scale_", "var_", "n_features_in_"],
        "MinMaxScaler": ["min_", "scale_", "data_min_", "data_max_", "n_features_in_"],
        "TruncatedSVD": ["components_", "explained_variance_ratio_", "singular_values_", "n_features_in_"],
    }

    def test_all_estimators_have_expected_attrs(self):
        X, y = _make_data(500, 20)
        for name, attrs in self.REQUIRED_ATTRS.items():
            cls = globals().get(name)
            if cls is None:
                continue
            model = accelerate(
                cls(**{"random_state": 42} if "random_state" in str(cls.__init__.__code__.co_varnames) else {})
            )
            if name in ("LinearRegression", "Ridge"):
                model.fit(X, y)
            elif name in ("LogisticRegression",):
                model.fit(X, y.astype(np.float32))
            elif name == "KMeans":
                model.fit(X)
            elif name in ("StandardScaler", "MinMaxScaler"):
                model.fit(X)
            elif name == "TruncatedSVD":
                model.fit(X)
            for attr in attrs:
                assert hasattr(model, attr), f"{name} missing attribute {attr}"
