"""CPU fallback utilities for skmetal."""

import warnings
import numpy as np
from sklearn.utils.validation import check_array


def validate_and_convert(X, dtype=np.float32, order="C", ensure_min_samples=2):
    """Validate and convert input array for GPU processing."""
    X = check_array(X, dtype=dtype, order=order, ensure_min_samples=ensure_min_samples)
    return X


def should_fallback(X, config, estimator_name: str) -> bool:
    """Determine whether to fall back to CPU."""
    if config.device == "cpu":
        if config.verbose:
            warnings.warn(f"{estimator_name}: CPU device forced, using CPU fallback")
        return True
    
    if hasattr(X, "nnz"):
        if config.verbose:
            warnings.warn(f"{estimator_name}: Sparse input not supported, using CPU fallback")
        return True
    
    if X.dtype != np.float32:
        if config.verbose:
            warnings.warn(f"{estimator_name}: Non-float32 dtype ({X.dtype}), using CPU fallback")
        return True
    
    n_samples, n_features = X.shape
    if n_samples * n_features < config.threshold:
        if config.verbose:
            warnings.warn(
                f"{estimator_name}: Problem size ({n_samples}x{n_features}) "
                f"below threshold ({config.threshold}), using CPU fallback"
            )
        return True
    
    return False


def get_fallback_warning(config) -> bool:
    return config.fallback_warn