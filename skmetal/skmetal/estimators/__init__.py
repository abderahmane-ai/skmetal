"""GPU-accelerated estimators for scikit-learn."""

from ._base import BaseGPUEstimator
from ._registry import GPU_ESTIMATORS, PIPELINE_PATTERNS
from .linear_model import MetalLinearRegression, MetalRidge, MetalLogisticRegression, MetalLasso, MetalElasticNet
from .decomposition import MetalPCA
from .cluster import MetalKMeans, MetalDBSCAN
from .naive_bayes import MetalGaussianNB
from .preprocessing import MetalStandardScaler, MetalMinMaxScaler, MetalRobustScaler
from .neighbors import MetalKNeighborsClassifier, MetalKNeighborsRegressor, MetalNearestNeighbors

__all__ = [
    "BaseGPUEstimator",
    "GPU_ESTIMATORS",
    "PIPELINE_PATTERNS",
    "MetalLinearRegression",
    "MetalRidge",
    "MetalLogisticRegression",
    "MetalLasso",
    "MetalElasticNet",
    "MetalPCA",
    "MetalKMeans",
    "MetalDBSCAN",
    "MetalGaussianNB",
    "MetalStandardScaler",
    "MetalMinMaxScaler",
    "MetalRobustScaler",
    "MetalKNeighborsClassifier",
    "MetalKNeighborsRegressor",
    "MetalNearestNeighbors",
]