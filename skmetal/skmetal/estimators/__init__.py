"""GPU-accelerated estimators for scikit-learn."""

from ._base import BaseGPUEstimator  # noqa: F401
from ._registry import GPU_REGISTRY, GPU_ESTIMATORS  # noqa: F401
from .linear_model import MetalLinearRegression, MetalRidge, MetalLogisticRegression, MetalLasso, MetalElasticNet  # noqa: F401,E501
from .cluster import MetalKMeans, MetalDBSCAN  # noqa: F401
from .naive_bayes import MetalGaussianNB  # noqa: F401
from .preprocessing import MetalStandardScaler, MetalMinMaxScaler, MetalRobustScaler  # noqa: F401
from .neighbors import MetalKNeighborsClassifier, MetalKNeighborsRegressor, MetalNearestNeighbors  # noqa: F401,E501
from .ensemble import MetalHistGradientBoostingRegressor, MetalHistGradientBoostingClassifier  # noqa: F401,E501
from .decomposition import MetalTruncatedSVD  # noqa: F401
from .svm import MetalSVC, MetalSVR  # noqa: F401

try:
    from ._mlx_svd import MetalTruncatedSVDMLX  # noqa: F401
    _HAS_MLX_SVD = True
except ImportError:
    _HAS_MLX_SVD = False
    MetalTruncatedSVDMLX = None  # type: ignore

__all__ = [
    "BaseGPUEstimator", "GPU_REGISTRY", "GPU_ESTIMATORS",
    "MetalLinearRegression", "MetalRidge", "MetalLogisticRegression",
    "MetalLasso", "MetalElasticNet", "MetalKMeans", "MetalDBSCAN",
    "MetalGaussianNB", "MetalStandardScaler", "MetalMinMaxScaler",
    "MetalRobustScaler", "MetalKNeighborsClassifier",
    "MetalKNeighborsRegressor", "MetalNearestNeighbors",
    "MetalHistGradientBoostingRegressor", "MetalHistGradientBoostingClassifier",
    "MetalSVC", "MetalSVR", "MetalTruncatedSVD",
]

if _HAS_MLX_SVD:
    __all__.append("MetalTruncatedSVDMLX")
