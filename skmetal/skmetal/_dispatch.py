"""Estimator registry and dispatch logic for skmetal."""

import warnings
from sklearn.pipeline import Pipeline
from .estimators._registry import GPU_REGISTRY
from ._config import get_config


def _is_supported(estimator) -> bool:
    return type(estimator) in GPU_REGISTRY


def _get_gpu_impl(estimator):
    """Import and return the GPU implementation class for *estimator*."""
    mod_name, cls_name = GPU_REGISTRY[type(estimator)]
    try:
        mod = __import__(mod_name, fromlist=[cls_name])
        return getattr(mod, cls_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"GPU_REGISTRY entry for {type(estimator).__name__} points to "
            f"'{mod_name}.{cls_name}' but could not import/retrieve it: {e}"
        ) from e


def _wrap_estimator(estimator):
    """Wrap a single estimator with its GPU implementation.

    If no GPU implementation exists the estimator is returned unchanged.
    """
    if not _is_supported(estimator):
        if get_config().verbose:
            warnings.warn(
                f"No GPU accelerator for {type(estimator).__name__}. Using CPU.",
                stacklevel=3,
            )
        return estimator

    GPUImpl = _get_gpu_impl(estimator)
    return GPUImpl(estimator)


def _wrap_pipeline(pipeline):
    """Return a new Pipeline where every step that has a GPU backend is wrapped.

    Dispatch is purely type-based (via GPU_REGISTRY) — step *names* are never
    inspected.  This avoids the substring-match brittleness of the old
    PIPELINE_PATTERNS approach and means adding a new estimator only requires
    an entry in ``_registry.py``.
    """
    steps = [(name, _wrap_estimator(est)) for name, est in pipeline.steps]
    return Pipeline(steps, memory=pipeline.memory, verbose=pipeline.verbose)
