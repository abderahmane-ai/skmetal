"""Estimator registry mapping sklearn classes to GPU implementations."""

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, Lasso, ElasticNet
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

# Mapping from sklearn class -> GPU implementation class name
GPU_ESTIMATORS = {
    LinearRegression: "MetalLinearRegression",
    Ridge: "MetalRidge",
    LogisticRegression: "MetalLogisticRegression",
    Lasso: "MetalLasso",
    ElasticNet: "MetalElasticNet",

    TruncatedSVD: "MetalTruncatedSVD",
    KMeans: "MetalKMeans",
    DBSCAN: "MetalDBSCAN",
    GaussianNB: "MetalGaussianNB",
    StandardScaler: "MetalStandardScaler",
    MinMaxScaler: "MetalMinMaxScaler",
    RobustScaler: "MetalRobustScaler",
    KNeighborsClassifier: "MetalKNeighborsClassifier",
    KNeighborsRegressor: "MetalKNeighborsRegressor",
    NearestNeighbors: "MetalNearestNeighbors",
    HistGradientBoostingRegressor: "MetalHistGradientBoostingRegressor",
    HistGradientBoostingClassifier: "MetalHistGradientBoostingClassifier",
}

# Pipeline support: map by step name patterns
PIPELINE_PATTERNS = {
    "linearregression": LinearRegression,
    "ridge": Ridge,
    "logisticregression": LogisticRegression,
    "lasso": Lasso,
    "elasticnet": ElasticNet,

    "truncatedsvd": TruncatedSVD,
    "kmeans": KMeans,
    "dbscan": DBSCAN,
    "gaussiannb": GaussianNB,
    "standardscaler": StandardScaler,
    "minmaxscaler": MinMaxScaler,
    "robustscaler": RobustScaler,
    "kneighborsclassifier": KNeighborsClassifier,
    "kneighborsregressor": KNeighborsRegressor,
    "nearestneighbors": NearestNeighbors,
    "histgradientboostingregressor": HistGradientBoostingRegressor,
    "histgradientboostingclassifier": HistGradientBoostingClassifier,
}
