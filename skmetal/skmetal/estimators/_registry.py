"""Estimator registry mapping sklearn classes to GPU implementations."""

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Mapping from sklearn class -> GPU implementation class name
GPU_ESTIMATORS = {
    LinearRegression: "MetalLinearRegression",
    Ridge: "MetalRidge",
    LogisticRegression: "MetalLogisticRegression",
    PCA: "MetalPCA",
    TruncatedSVD: "MetalTruncatedSVD",
    KMeans: "MetalKMeans",
    StandardScaler: "MetalStandardScaler",
    MinMaxScaler: "MetalMinMaxScaler",
}

# Pipeline support: map by step name patterns
PIPELINE_PATTERNS = {
    "linearregression": LinearRegression,
    "ridge": Ridge,
    "logisticregression": LogisticRegression,
    "pca": PCA,
    "truncatedsvd": TruncatedSVD,
    "kmeans": KMeans,
    "standardscaler": StandardScaler,
    "minmaxscaler": MinMaxScaler,
}