"""Estimator registry — single source of truth for all GPU implementations.

Each entry maps an sklearn class to a (module_path, class_name) tuple.
``_dispatch.py`` reads this directly — no second map to maintain.
"""

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, Lasso, ElasticNet
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.svm import SVC, SVR

_HAS_MLX = False
try:
    import mlx.core  # noqa: F401
    _HAS_MLX = True
except ImportError:
    pass

_KMEANS_CLASS = "MetalKMeansMLX" if _HAS_MLX else "MetalKMeans"

# Maps sklearn class → (python_module, gpu_class_name).
# This is the ONLY place where this mapping lives. _dispatch.py consumes it directly.
GPU_REGISTRY: dict[type, tuple[str, str]] = {
    LinearRegression:                ("skmetal.estimators.linear_model", "MetalLinearRegression"),
    Ridge:                           ("skmetal.estimators.linear_model", "MetalRidge"),
    LogisticRegression:              ("skmetal.estimators.linear_model", "MetalLogisticRegression"),
    Lasso:                           ("skmetal.estimators.linear_model", "MetalLasso"),
    ElasticNet:                      ("skmetal.estimators.linear_model", "MetalElasticNet"),

    TruncatedSVD:                    ("skmetal.estimators.decomposition", "MetalTruncatedSVD"),
    KMeans:                          ("skmetal.estimators.cluster",       _KMEANS_CLASS),
    DBSCAN:                          ("skmetal.estimators.cluster",       "MetalDBSCAN"),
    GaussianNB:                      ("skmetal.estimators.naive_bayes",   "MetalGaussianNB"),

    StandardScaler:                  ("skmetal.estimators.preprocessing", "MetalStandardScaler"),
    MinMaxScaler:                    ("skmetal.estimators.preprocessing", "MetalMinMaxScaler"),
    RobustScaler:                    ("skmetal.estimators.preprocessing", "MetalRobustScaler"),

    KNeighborsClassifier:            ("skmetal.estimators.neighbors",    "MetalKNeighborsClassifier"),
    KNeighborsRegressor:             ("skmetal.estimators.neighbors",    "MetalKNeighborsRegressor"),
    NearestNeighbors:                ("skmetal.estimators.neighbors",    "MetalNearestNeighbors"),

    HistGradientBoostingRegressor:   ("skmetal.estimators.ensemble",     "MetalHistGradientBoostingRegressor"),
    HistGradientBoostingClassifier:  ("skmetal.estimators.ensemble",     "MetalHistGradientBoostingClassifier"),

    SVC:                             ("skmetal.estimators.svm",          "MetalSVC"),
    SVR:                             ("skmetal.estimators.svm",          "MetalSVR"),
}

# Derived map: sklearn class → GPU class name. Computed from GPU_REGISTRY.
GPU_ESTIMATORS: dict[type, str] = {cls: name for cls, (_, name) in GPU_REGISTRY.items()}
