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

from ._config import get_config, set_device, set_threshold, set_dtype, set_verbose
from .accelerate import accelerate, accelerate_context
from ._bridge import device_info

__version__ = "0.1.0"
__all__ = [
    "accelerate",
    "accelerate_context",
    "get_config",
    "set_device",
    "set_threshold",
    "set_dtype",
    "set_verbose",
    "device_info",
]