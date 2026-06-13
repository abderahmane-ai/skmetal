"""
skmetal: Apple Silicon GPU acceleration for scikit-learn.

Usage:
    import skmetal
    from sklearn.linear_model import LinearRegression

    @skmetal.accelerate
    def model():
        return LinearRegression()

    m = model()
    m.fit(X, y)
    m.predict(X_test)
"""

from ._about import __version__, __version_info__
from ._config import get_config, set_device, set_threshold, set_dtype, set_verbose, set_thresholds, update_threshold
from .accelerate import accelerate, accelerate_context
from ._bridge import device_info
__all__ = [
    "accelerate",
    "accelerate_context",
    "get_config",
    "set_device",
    "set_threshold",
    "set_dtype",
    "set_verbose",
    "set_thresholds",
    "update_threshold",
    "device_info",
]