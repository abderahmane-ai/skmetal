"""GPU-accelerated estimators for scikit-learn."""

from ._base import BaseGPUEstimator
from ._registry import GPU_REGISTRY, GPU_ESTIMATORS
from .linear_model import MetalLinearRegression, MetalRidge, MetalLogisticRegression, MetalLasso, MetalElasticNet

from .cluster import MetalKMeans, MetalDBSCAN
from .naive_bayes import MetalGaussianNB
from .preprocessing import MetalStandardScaler, MetalMinMaxScaler, MetalRobustScaler
from .neighbors import MetalKNeighborsClassifier, MetalKNeighborsRegressor, MetalNearestNeighbors
from .ensemble import MetalHistGradientBoostingRegressor, MetalHistGradientBoostingClassifier
from .svm import MetalSVC, MetalSVR

__all__ = [
    "BaseGPUEstimator",
    "GPU_REGISTRY",
    "GPU_ESTIMATORS",

    "MetalLinearRegression",
    "MetalRidge",
    "MetalLogisticRegression",
    "MetalLasso",
    "MetalElasticNet",

    "MetalKMeans",
    "MetalDBSCAN",
    "MetalGaussianNB",
    "MetalStandardScaler",
    "MetalMinMaxScaler",
    "MetalRobustScaler",
    "MetalKNeighborsClassifier",
    "MetalKNeighborsRegressor",
    "MetalNearestNeighbors",
    "MetalHistGradientBoostingRegressor",
    "MetalHistGradientBoostingClassifier",

    "MetalSVC",
    "MetalSVR",
]
