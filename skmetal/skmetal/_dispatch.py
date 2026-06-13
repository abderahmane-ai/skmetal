"""Estimator registry and dispatch logic for skmetal."""

import warnings
from sklearn.pipeline import Pipeline
from .estimators._registry import GPU_ESTIMATORS, PIPELINE_PATTERNS
from ._config import get_config


def _is_supported(estimator) -> bool:
    return type(estimator) in GPU_ESTIMATORS


def _get_gpu_impl(estimator):
    """Import and return GPU implementation class."""
    gpu_name = GPU_ESTIMATORS[type(estimator)]
    module_map = {
        "MetalLinearRegression": ("skmetal.estimators.linear_model", "MetalLinearRegression"),
        "MetalRidge": ("skmetal.estimators.linear_model", "MetalRidge"),
        "MetalLogisticRegression": ("skmetal.estimators.linear_model", "MetalLogisticRegression"),
        "MetalLasso": ("skmetal.estimators.linear_model", "MetalLasso"),
        "MetalElasticNet": ("skmetal.estimators.linear_model", "MetalElasticNet"),
        "MetalPCA": ("skmetal.estimators.decomposition", "MetalPCA"),
        "MetalTruncatedSVD": ("skmetal.estimators.decomposition", "MetalTruncatedSVD"),
        "MetalKMeans": ("skmetal.estimators.cluster", "MetalKMeans"),
        "MetalDBSCAN": ("skmetal.estimators.cluster", "MetalDBSCAN"),
        "MetalGaussianNB": ("skmetal.estimators.naive_bayes", "MetalGaussianNB"),
        "MetalStandardScaler": ("skmetal.estimators.preprocessing", "MetalStandardScaler"),
        "MetalMinMaxScaler": ("skmetal.estimators.preprocessing", "MetalMinMaxScaler"),
        "MetalRobustScaler": ("skmetal.estimators.preprocessing", "MetalRobustScaler"),
        "MetalKNeighborsClassifier": ("skmetal.estimators.neighbors", "MetalKNeighborsClassifier"),
        "MetalKNeighborsRegressor": ("skmetal.estimators.neighbors", "MetalKNeighborsRegressor"),
        "MetalNearestNeighbors": ("skmetal.estimators.neighbors", "MetalNearestNeighbors"),
    }
    mod_name, cls_name = module_map[gpu_name]
    mod = __import__(mod_name, fromlist=[cls_name])
    return getattr(mod, cls_name)


def _wrap_estimator(estimator):
    """Wrap a single estimator with GPU implementation."""
    if not _is_supported(estimator):
        if get_config().verbose:
            warnings.warn(f"No GPU accelerator for {type(estimator).__name__}. Using CPU.")
        return estimator

    GPUImpl = _get_gpu_impl(estimator)
    return GPUImpl(estimator)


def _wrap_pipeline(pipeline):
    """Wrap pipeline steps that have GPU implementations."""
    steps = []
    for name, est in pipeline.steps:
        matched = False
        for pattern, cls in PIPELINE_PATTERNS.items():
            if pattern in name.lower():
                if _is_supported(est):
                    GPUImpl = _get_gpu_impl(est)
                    steps.append((name, GPUImpl(est)))
                    matched = True
                    break
        if not matched:
            steps.append((name, _wrap_estimator(est)))

    return Pipeline(steps, memory=pipeline.memory, verbose=pipeline.verbose)