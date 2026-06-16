"""Large-matrix benchmark and stress tests.

Exercises every estimator with matrix sizes ≥ 100K samples or ≥ 500 features
to validate GPU correctness and detect memory/crash bugs at scale.

Requires the Metal dylib (Apple Silicon).
"""

import time
import numpy as np
from sklearn import datasets
import pytest

from skmetal import _bridge, accelerate
from skmetal._config import get_config
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import DBSCAN

pytestmark = [
    pytest.mark.skipif(not hasattr(_bridge, "_lib"), reason="Metal dylib not available"),
]

# ── helpers ──────────────────────────────────────────────────────────────────

_RNG = np.random.default_rng(42)


def _make_classification(n, d, n_classes=2):
    X, y = datasets.make_classification(
        n_samples=n,
        n_features=d,
        n_classes=n_classes,
        n_informative=max(d // 2, 1),
        n_redundant=0,
        random_state=42,
    )
    return X.astype(np.float32), y


def _make_regression(n, d):
    X, y = datasets.make_regression(
        n_samples=n,
        n_features=d,
        noise=0.1,
        random_state=42,
    )
    return X.astype(np.float32), y.astype(np.float32)


def _make_blobs(n, d, k=10):
    X, y = datasets.make_blobs(
        n_samples=n,
        n_features=d,
        centers=k,
        random_state=42,
    )
    return X.astype(np.float32), y


# ── Large-classification tests ───────────────────────────────────────────────


class TestLargeClassification:
    """100K samples, 100 features — exercises GEMM and iterative solvers."""

    N, D = 100_000, 100

    @classmethod
    def setup_class(cls):
        get_config().verbose = False
        get_config().threshold = 1  # always use GPU
        cls.X, cls.y = _make_classification(cls.N, cls.D)

    # ── LogisticRegression ───────────────────────────────────────────────
    def test_logistic_regression_fit(self):
        model = accelerate(LogisticRegression(max_iter=200, random_state=42))
        model.fit(self.X, self.y)
        # at least better than chance
        acc = (model.predict(self.X) == self.y).mean()
        assert acc > 0.7, f"LogReg accuracy too low: {acc:.3f}"

    def test_logistic_regression_predict_shape(self):
        model = accelerate(LogisticRegression(max_iter=100, random_state=42))
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        assert preds.shape == (self.N,)
        probas = model.predict_proba(self.X)
        assert probas.shape == (self.N, 2)

    # ── HistGradientBoostingClassifier ───────────────────────────────────
    def test_hgb_classifier_fit(self):
        model = accelerate(HistGradientBoostingClassifier(max_iter=50, random_state=42))
        model.fit(self.X, self.y)
        acc = (model.predict(self.X) == self.y).mean()
        assert acc > 0.7

    # ── GaussianNB ───────────────────────────────────────────────────────
    def test_gaussian_nb_fit(self):
        # Use easy data: 5 informative features, 2 classes, no noise
        X, y = datasets.make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=10,
            n_classes=2,
            random_state=42,
        )
        model = accelerate(GaussianNB())
        model.fit(X.astype(np.float32), y)
        acc = (model.predict(X.astype(np.float32)) == y).mean()
        assert acc > 0.8

    # ── KNeighborsClassifier ─────────────────────────────────────────────
    def test_knn_classifier_fit_predict(self):
        n_train = 10_000
        X_train, y_train = self.X[:n_train], self.y[:n_train]
        model = accelerate(KNeighborsClassifier(n_neighbors=5))
        model.fit(X_train, y_train)
        preds = model.predict(self.X[:2000])
        assert preds.shape == (2000,)
        acc = (preds == self.y[:2000]).mean()
        assert acc > 0.5

    # ── SVC ──────────────────────────────────────────────────────────────
    def test_svc_fit_predict(self):
        n_train = 2000
        X_train, y_train = self.X[:n_train], self.y[:n_train]
        model = accelerate(SVC(kernel="rbf", gamma="scale", random_state=42))
        model.fit(X_train, y_train)
        preds = model.predict(self.X[:500])
        assert preds.shape == (500,)


class TestLargeRegression:
    """100K samples, 100 features — regression estimators."""

    N, D = 100_000, 100

    @classmethod
    def setup_class(cls):
        get_config().verbose = False
        get_config().threshold = 1
        cls.X, cls.y = _make_regression(cls.N, cls.D)

    def test_linear_regression_fit(self):
        model = accelerate(LinearRegression())
        model.fit(self.X, self.y)
        score = model.score(self.X, self.y)
        assert score > 0.8, f"LinearRegression R² too low: {score:.4f}"

    def test_ridge_fit(self):
        model = accelerate(Ridge(alpha=1.0))
        model.fit(self.X, self.y)
        score = model.score(self.X, self.y)
        assert score > 0.8

    def test_lasso_fit(self):
        model = accelerate(Lasso(alpha=0.01, max_iter=200))
        model.fit(self.X, self.y)
        score = model.score(self.X, self.y)
        assert score > 0.5

    def test_knn_regression_fit_predict(self):
        n_train = 10_000
        X_train, y_train = self.X[:n_train], self.y[:n_train]
        model = accelerate(KNeighborsRegressor(n_neighbors=5))
        model.fit(X_train, y_train)
        preds = model.predict(self.X[:2000])
        assert preds.shape == (2000,)

    def test_hgb_regressor_fit(self):
        model = accelerate(HistGradientBoostingRegressor(max_iter=50, random_state=42))
        model.fit(self.X, self.y)
        score = model.score(self.X, self.y)
        assert score > 0.5

    def test_svr_fit_predict(self):
        n_train = 1000
        X_train, y_train = self.X[:n_train], self.y[:n_train]
        model = accelerate(SVR(kernel="rbf", gamma="scale"))
        model.fit(X_train, y_train)
        preds = model.predict(self.X[:200])
        assert preds.shape == (200,)


# ── Large-unsupervised tests ─────────────────────────────────────────────────


class TestLargeUnsupervised:
    """100K samples, 50 features — clustering and decomposition."""

    N, D, K = 100_000, 50, 20

    @classmethod
    def setup_class(cls):
        get_config().verbose = False
        get_config().threshold = 1
        cls.X, cls.y = _make_blobs(cls.N, cls.D, cls.K)

    def test_kmeans_fit(self):
        model = accelerate(KMeans(n_clusters=self.K, random_state=42, n_init=1))
        model.fit(self.X)
        assert len(np.unique(model.labels_)) == self.K
        assert model.cluster_centers_.shape == (self.K, self.D)
        assert np.all(np.isfinite(model.cluster_centers_))

    def test_kmeans_transform(self):
        model = accelerate(KMeans(n_clusters=self.K, random_state=42, n_init=1))
        model.fit(self.X)
        dists = model.transform(self.X[:1000])
        assert dists.shape == (1000, self.K)
        assert np.all(dists >= 0)

    def test_truncated_svd_fit(self):
        model = accelerate(TruncatedSVD(n_components=10, random_state=42))
        model.fit(self.X)
        assert model.components_.shape == (10, self.D)
        Xt = model.transform(self.X[:1000])
        assert Xt.shape == (1000, 10)

    def test_dbscan_fit(self):
        X_small, y_true = _make_blobs(500, 5, 5)
        model = accelerate(DBSCAN(eps=2.0, min_samples=5))
        model.fit(X_small)
        n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
        assert n_clusters >= 3, f"DBSCAN found only {n_clusters} clusters (expected ~5)"


# ── Scaler tests (very large — 1M rows) ──────────────────────────────────────


class TestLargeScalers:
    N, D = 1_000_000, 100

    @classmethod
    def setup_class(cls):
        get_config().verbose = False
        get_config().threshold = 1
        cls.X = _RNG.uniform(-10, 10, size=(cls.N, cls.D)).astype(np.float32)

    def test_standard_scaler_fit(self):
        model = accelerate(StandardScaler())
        model.fit(self.X)
        assert model.mean_.shape == (self.D,)
        assert model.scale_.shape == (self.D,)
        assert np.all(np.isfinite(model.mean_))
        assert np.all(np.isfinite(model.scale_))

    def test_standard_scaler_transform(self):
        model = accelerate(StandardScaler())
        model.fit(self.X)
        Xt = model.transform(self.X[:5000])
        assert Xt.shape == (5000, self.D)
        assert np.all(np.isfinite(Xt))

    def test_minmax_scaler_fit(self):
        model = accelerate(MinMaxScaler())
        model.fit(self.X)
        assert model.min_.shape == (self.D,)
        assert model.scale_.shape == (self.D,)

    def test_robust_scaler_fit(self):
        model = accelerate(RobustScaler())
        model.fit(self.X)
        assert model.center_.shape == (self.D,)


# ── Fat-matrix tests (p >> n) ────────────────────────────────────────────────


class TestFatMatrices:
    """Wide matrices — 500 features, only 2000 samples."""

    N, D = 2000, 500

    @classmethod
    def setup_class(cls):
        get_config().verbose = False
        get_config().threshold = 1
        cls.X_reg, cls.y_reg = _make_regression(cls.N, cls.D)

    def test_linear_regression_wide(self):
        model = accelerate(LinearRegression())
        model.fit(self.X_reg, self.y_reg)
        score = model.score(self.X_reg, self.y_reg)
        assert score > 0.5

    def test_ridge_wide(self):
        model = accelerate(Ridge(alpha=10.0))
        model.fit(self.X_reg, self.y_reg)
        score = model.score(self.X_reg, self.y_reg)
        assert score > 0.5

    def test_truncated_svd_wide(self):
        X, _ = _make_blobs(self.N, self.D, 10)
        model = accelerate(TruncatedSVD(n_components=20, random_state=42))
        model.fit(X)
        assert model.components_.shape == (20, self.D)


# ── Tall-skinny tests (n >> d) ───────────────────────────────────────────────


class TestTallSkinny:
    """Very tall matrices — 500K samples, 10 features."""

    N, D = 500_000, 10

    @classmethod
    def setup_class(cls):
        get_config().verbose = False
        get_config().threshold = 1
        cls.X, cls.y = _make_regression(cls.N, cls.D)

    def test_standard_scaler_tall(self):
        model = accelerate(StandardScaler())
        model.fit(self.X)
        assert model.mean_.shape == (self.D,)

    def test_kmeans_tall(self):
        model = accelerate(KMeans(n_clusters=5, random_state=42, n_init=1))
        model.fit(self.X)
        assert model.cluster_centers_.shape == (5, self.D)

    def test_linear_regression_tall(self):
        model = accelerate(LinearRegression())
        model.fit(self.X, self.y)
        score = model.score(self.X, self.y)
        assert score > 0.8

    def test_ridge_tall(self):
        model = accelerate(Ridge(alpha=1.0))
        model.fit(self.X, self.y)
        score = model.score(self.X, self.y)
        assert score > 0.8


# ── Full benchmark (not assertions, just timing report) ──────────────────────


class TestBenchmarkReport:
    """Report GPU timings for large matrices (no pass/fail, just info)."""

    @pytest.mark.parametrize(
        "estimator_cls,kwargs,n,d,has_y",
        [
            (LinearRegression, {}, 200_000, 500, True),
            (Ridge, {"alpha": 1.0}, 200_000, 500, True),
            (LogisticRegression, {"max_iter": 100, "random_state": 42}, 100_000, 200, True),
            (KMeans, {"n_clusters": 50, "random_state": 42, "n_init": 1}, 500_000, 100, False),
        ],
    )
    def test_gpu_timing(self, estimator_cls, kwargs, n, d, has_y, capsys):
        """Fit on GPU and report elapsed time (no speedup assertion)."""
        X = _RNG.uniform(-1, 1, size=(n, d)).astype(np.float32)
        if has_y:
            y = (
                _RNG.uniform(-1, 1, size=n).astype(np.float32)
                if "Logistic" not in estimator_cls.__name__
                else ((_RNG.uniform(-1, 1, size=n) > 0).astype(np.float32))
            )
        else:
            y = None

        model = accelerate(estimator_cls(**kwargs))
        t0 = time.perf_counter()
        model.fit(X, y)
        elapsed = time.perf_counter() - t0
        # Just print timing; no speedup assertion since CI may vary
        with capsys.disabled():
            print(f"  {estimator_cls.__name__}: {n}×{d} → {elapsed:.3f}s")
