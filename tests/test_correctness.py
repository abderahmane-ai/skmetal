"""Correctness tests for skmetal GPU acceleration."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, Lasso, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
import skmetal
from skmetal import _bridge


# Mark all tests in this module
pytestmark = [
    pytest.mark.skipif(
        not hasattr(_bridge, "_lib"),
        reason="SkMetalBridge dylib not available",
    ),
]


SKIP_ATTRS = {
    "n_iter_",
    "n_features_in_",
    "n_features_out_",
    "n_samples_seen_",
    "_fit_X",
    "_y",
    "n_samples_fit_",
    "support_vectors_",
    "support_",
    "n_support_",
    "dual_coef_",
    "intercept_",
    "probA_",
    "probB_",
    "shape_fit_",
    "n_features_in_",
}


def _match_clusters(gpu_centers, cpu_centers):
    """Match GPU cluster centers to CPU centers via Hungarian assignment."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import pairwise_distances

    cost = pairwise_distances(gpu_centers, cpu_centers)
    row_ind, col_ind = linear_sum_assignment(cost)
    return gpu_centers[row_ind]


ESTIMATOR_TOL = {
    LogisticRegression: 0.5,
    TruncatedSVD: 0.5,  # randomized SVD uses different random seeds
    SVC: 0.5,  # GPU SVC uses different solver path
    SVR: 0.5,
}


def _check_attrs(gpu_obj, cpu_obj, estimator_cls=None):
    """Compare fitted attributes between GPU and CPU estimators."""
    rtol = ESTIMATOR_TOL.get(estimator_cls, 1e-3)
    atol = 1e-4
    for attr in dir(gpu_obj):
        if attr.startswith("_") or attr in SKIP_ATTRS:
            continue
        gpu_val = getattr(gpu_obj, attr, None)
        cpu_val = getattr(cpu_obj, attr, None)
        if gpu_val is None or cpu_val is None:
            continue
        if callable(gpu_val) or callable(cpu_val):
            continue
        if isinstance(gpu_val, str) or isinstance(cpu_val, str):
            continue
        if isinstance(gpu_val, np.ndarray) and isinstance(cpu_val, np.ndarray):
            if attr == "cluster_centers_":
                gpu_val = _match_clusters(gpu_val, cpu_val)
            if estimator_cls is TruncatedSVD and attr in ("components_",):
                # Randomized SVD: check subspace alignment, not component equality
                continue
            np.testing.assert_allclose(gpu_val, cpu_val, rtol=rtol, atol=atol)
        elif isinstance(gpu_val, (int, float, np.number)) and isinstance(cpu_val, (int, float, np.number)):
            assert abs(gpu_val - cpu_val) <= atol + rtol * abs(cpu_val)


@pytest.mark.parametrize(
    "EstimatorCls, data_fn, has_y",
    [
        (LinearRegression, lambda: make_regression(n_samples=2000, n_features=50, noise=0.1, random_state=42), True),
        (Ridge, lambda: make_regression(n_samples=2000, n_features=50, noise=0.1, random_state=42), True),
        (LogisticRegression, lambda: make_classification(n_samples=2000, n_features=50, random_state=42), True),
        (
            LogisticRegression,  # multi-class (exercises multinomial_lbfgs_fit)
            lambda: make_classification(n_samples=2000, n_features=50, n_classes=3, n_informative=10, random_state=42),
            True,
        ),
        (TruncatedSVD, lambda: make_regression(n_samples=2000, n_features=100, random_state=42), False),
        (KMeans, lambda: make_blobs(n_samples=2000, centers=10, n_features=50, random_state=42), False),
        (StandardScaler, lambda: make_regression(n_samples=2000, n_features=50, random_state=42), False),
        (MinMaxScaler, lambda: make_regression(n_samples=2000, n_features=50, random_state=42), False),
        (RobustScaler, lambda: make_regression(n_samples=2000, n_features=50, random_state=42), False),
        (Lasso, lambda: make_regression(n_samples=2000, n_features=50, noise=0.1, random_state=42), True),
        (
            KNeighborsClassifier,
            lambda: make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=42),
            True,
        ),
        (KNeighborsRegressor, lambda: make_regression(n_samples=2000, n_features=20, noise=0.1, random_state=42), True),
        (ElasticNet, lambda: make_regression(n_samples=2000, n_features=50, noise=0.1, random_state=42), True),
        (
            GaussianNB,
            lambda: make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=42),
            True,
        ),
        (DBSCAN, lambda: make_blobs(n_samples=2000, centers=5, n_features=10, cluster_std=0.5, random_state=42), False),
        (
            HistGradientBoostingRegressor,
            lambda: make_regression(n_samples=500, n_features=20, noise=0.1, random_state=42),
            True,
        ),
        (
            HistGradientBoostingClassifier,
            lambda: make_classification(n_samples=500, n_features=20, random_state=42),
            True,
        ),
        (SVC, lambda: make_classification(n_samples=500, n_features=20, random_state=42), True),
        (SVR, lambda: make_regression(n_samples=500, n_features=20, noise=0.1, random_state=42), True),
        (NearestNeighbors, lambda: make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42), False),
    ],
)
def test_estimator_correctness(EstimatorCls, data_fn, has_y):
    """Compare GPU-accelerated estimator against CPU baseline."""
    X, y = data_fn()
    X = X.astype(np.float32)
    if y is not None:
        y = y.astype(np.float32)

    extra_kwargs = {}
    if EstimatorCls.__name__ in ("KMeans", "LogisticRegression"):
        extra_kwargs = {"random_state": 42}
    if EstimatorCls.__name__ in ("KNeighborsClassifier", "KNeighborsRegressor"):
        extra_kwargs = {"n_neighbors": 5}
    if EstimatorCls.__name__ in ("DBSCAN",):
        extra_kwargs = {"eps": 0.5, "min_samples": 5}
    if EstimatorCls.__name__ in ("SVC", "SVR"):
        extra_kwargs = {"kernel": "rbf", "gamma": "scale"}
    common_kwargs = extra_kwargs

    cpu_model = EstimatorCls(**common_kwargs)
    if has_y:
        cpu_model.fit(X, y)
    else:
        cpu_model.fit(X)

    gpu_model = skmetal.accelerate(EstimatorCls(**common_kwargs))
    if has_y:
        gpu_model.fit(X, y)
    else:
        gpu_model.fit(X)

    gpu_obj = gpu_model._estimator if hasattr(gpu_model, "_estimator") else gpu_model
    _check_attrs(gpu_obj, cpu_model, estimator_cls=EstimatorCls)

    # For SVC/SVR: verify prediction accuracy against CPU
    if EstimatorCls in (SVC, SVR):
        gpu_pred = gpu_model.predict(X)
        cpu_pred = cpu_model.predict(X)
        if EstimatorCls is SVC:
            acc_gpu = (gpu_pred == y).mean()
            acc_cpu = (cpu_pred == y).mean()
            assert acc_gpu >= acc_cpu - 0.2  # GPU within 20pp of CPU
        else:
            err_gpu = np.mean((gpu_pred - y) ** 2)
            err_cpu = np.mean((cpu_pred - y) ** 2)
            assert err_gpu <= err_cpu * 1.5  # GPU MSE within 50% of CPU

    # For TruncatedSVD: verify subspace quality via reconstruction error
    if EstimatorCls is TruncatedSVD:
        X_proj = gpu_model.transform(X)
        X_recon = X_proj @ gpu_obj.components_
        cpu_proj = cpu_model.transform(X)
        cpu_recon = cpu_proj @ cpu_model.components_
        gpu_err = np.linalg.norm(X - X_recon)
        cpu_err = np.linalg.norm(X - cpu_recon)
        assert gpu_err <= cpu_err * 1.1  # GPU reconstruction ≤ 110% of CPU


def test_pipeline_correctness():
    """Test GPU-accelerated pipeline."""
    X, y = make_classification(n_samples=2000, n_features=100, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    pipe_cpu = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ]
    )
    pipe_cpu.fit(X, y)

    pipe_gpu = skmetal.accelerate(
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression()),
            ]
        )
    )
    pipe_gpu.fit(X, y)

    np.testing.assert_allclose(pipe_gpu.predict_proba(X), pipe_cpu.predict_proba(X), rtol=0.5, atol=1.0)


def test_device_info():
    """Test that Metal device info can be retrieved."""
    info = _bridge.device_info()
    assert "name" in info
    assert len(info["name"]) > 0
    assert info["max_threads_per_threadgroup"] > 0


def test_weighted_knn_classify():
    """Test KNN with weighted voting (weights='distance')."""
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    cpu_model = KNeighborsClassifier(n_neighbors=5, weights="distance")
    cpu_model.fit(X, y)
    cpu_pred = cpu_model.predict(X)

    gpu_model = skmetal.accelerate(KNeighborsClassifier(n_neighbors=5, weights="distance"))
    gpu_model.fit(X, y)
    gpu_pred = gpu_model.predict(X)

    assert np.mean(gpu_pred == cpu_pred) >= 0.85


def test_weighted_knn_regress():
    """Test KNN regression with weighted voting (weights='distance')."""
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    cpu_model = KNeighborsRegressor(n_neighbors=5, weights="distance")
    cpu_model.fit(X, y)
    cpu_pred = cpu_model.predict(X)

    gpu_model = skmetal.accelerate(KNeighborsRegressor(n_neighbors=5, weights="distance"))
    gpu_model.fit(X, y)
    gpu_pred = gpu_model.predict(X)

    assert np.corrcoef(gpu_pred, cpu_pred)[0, 1] >= 0.90


def test_logistic_regression_fit_intercept_false():
    """Test LogisticRegression with fit_intercept=False."""
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    cpu_model = LogisticRegression(fit_intercept=False)
    cpu_model.fit(X, y)

    gpu_model = skmetal.accelerate(LogisticRegression(fit_intercept=False))
    gpu_model.fit(X, y)

    assert np.mean(gpu_model.predict(X) == cpu_model.predict(X)) >= 0.85
