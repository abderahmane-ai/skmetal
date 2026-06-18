"""Estimator registry — single source of truth for all GPU implementations.

Each entry maps an sklearn class to a (module_path, class_name) tuple.
``_dispatch.py`` reads this directly — no second map to maintain.
"""

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, Lasso, ElasticNet
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.svm import SVC, SVR

from ._mlx_registry import has_mlx, has_flash_kmeans

_HAS_MLX = has_mlx()
_HAS_FLASH_KMEANS = has_flash_kmeans()

# Maps sklearn class -> (python_module, gpu_class_name).
# This is the ONLY place where this mapping lives. _dispatch.py consumes it directly.
GPU_REGISTRY: dict[type, tuple[str, str]] = {
    LinearRegression: ("skmetal.estimators.linear_model", "MetalLinearRegression"),
    Ridge: ("skmetal.estimators.linear_model", "MetalRidge"),
    LogisticRegression: ("skmetal.estimators.linear_model", "MetalLogisticRegression"),
    Lasso: ("skmetal.estimators.linear_model", "MetalLasso"),
    ElasticNet: ("skmetal.estimators.linear_model", "MetalElasticNet"),
    TruncatedSVD: ("skmetal.estimators.decomposition", "MetalTruncatedSVD"),
    PCA: ("skmetal.estimators.decomposition", "MetalPCA"),
    DBSCAN: ("skmetal.estimators.cluster", "MetalDBSCAN"),
    GaussianNB: ("skmetal.estimators.naive_bayes", "MetalGaussianNB"),
    StandardScaler: ("skmetal.estimators.preprocessing", "MetalStandardScaler"),
    MinMaxScaler: ("skmetal.estimators.preprocessing", "MetalMinMaxScaler"),
    RobustScaler: ("skmetal.estimators.preprocessing", "MetalRobustScaler"),
    KNeighborsClassifier: ("skmetal.estimators.neighbors", "MetalKNeighborsClassifier"),
    KNeighborsRegressor: ("skmetal.estimators.neighbors", "MetalKNeighborsRegressor"),
    NearestNeighbors: ("skmetal.estimators.neighbors", "MetalNearestNeighbors"),
    HistGradientBoostingRegressor: ("skmetal.estimators.ensemble", "MetalHistGradientBoostingRegressor"),
    HistGradientBoostingClassifier: ("skmetal.estimators.ensemble", "MetalHistGradientBoostingClassifier"),
    SVC: ("skmetal.estimators.svm", "MetalSVC"),
    SVR: ("skmetal.estimators.svm", "MetalSVR"),
}

# Opt-in MLX backends — only registered when they actually accelerate.
# KMeans without MLX is slower than CPU (0.1×); only register with flash-kmeans.
# TruncatedSVD MLX path uses GPU SVD via mx.linalg.svd (3× speedup).
if _HAS_MLX:
    GPU_REGISTRY[TruncatedSVD] = ("skmetal.estimators._mlx_svd", "MetalTruncatedSVDMLX")
if _HAS_FLASH_KMEANS:
    GPU_REGISTRY[KMeans] = ("skmetal.estimators._mlx_kmeans", "MetalKMeansMLX")
