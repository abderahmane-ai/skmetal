"""GPU-accelerated estimators for scikit-learn."""

from ._base import BaseGPUEstimator
from ._registry import GPU_ESTIMATORS, PIPELINE_PATTERNS
from .linear_model import MetalLinearRegression, MetalRidge, MetalLogisticRegression
from .decomposition import MetalPCA
from .cluster import MetalKMeans
from .preprocessing import MetalStandardScaler, MetalMinMaxScaler

__all__ = [
    "BaseGPUEstimator",
    "GPU_ESTIMATORS",
    "PIPELINE_PATTERNS",
    "MetalLinearRegression",
    "MetalRidge",
    "MetalLogisticRegression",
    "MetalPCA",
    "MetalKMeans",
    "MetalStandardScaler",
    "MetalMinMaxScaler",
]