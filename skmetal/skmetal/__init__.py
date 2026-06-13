"""
skmetal: Apple Silicon GPU acceleration for scikit-learn.

Usage::

    import skmetal
    from sklearn.linear_model import LinearRegression

    @skmetal.accelerate
    def model():
        return LinearRegression()

    m = model()
    m.fit(X, y)
    m.predict(X_test)

On non-Apple-Silicon machines skmetal imports cleanly and all estimators
transparently fall back to scikit-learn CPU implementations.
Check ``skmetal.METAL_AVAILABLE`` to detect GPU support at runtime.
"""

from ._about import __version__, __version_info__
from ._config import get_config, set_device, set_threshold, set_dtype, set_verbose, set_thresholds, update_threshold
from .accelerate import accelerate, accelerate_context
from ._bridge import METAL_AVAILABLE

if METAL_AVAILABLE:
    from ._bridge import device_info
else:
    def device_info() -> dict:  # type: ignore[misc]
        """Returns empty info when Metal is unavailable."""
        raise RuntimeError(
            "skmetal: device_info() requires Apple Silicon + macOS 14+. "
            "Metal is not available on this device."
        )

__all__ = [
    "__version__",
    "__version_info__",
    "METAL_AVAILABLE",
    "accelerate",
    "accelerate_context",
    "device_info",
    "get_config",
    "set_device",
    "set_threshold",
    "set_dtype",
    "set_verbose",
    "set_thresholds",
    "update_threshold",
]

