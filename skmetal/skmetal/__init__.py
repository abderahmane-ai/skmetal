"""
skmetal: Apple Silicon GPU acceleration for scikit-learn.

Drop-in GPU acceleration for scikit-learn estimators on Apple Silicon
(M1-M5). Decorate any estimator-returning function with ``@accelerate``
and ``fit()``/``predict()`` run on the Metal GPU — no code changes.

Quick start::

    import skmetal
    from sklearn.linear_model import LinearRegression

    @skmetal.accelerate
    def model():
        return LinearRegression()

    m = model()
    m.fit(X_train, y_train)
    m.predict(X_test)

Installation::

    pip install skmetal

macOS 14+ and Apple Silicon required. No Xcode needed for the pip package.

Configuration::

    import skmetal

    skmetal.set_device("cpu")             # force CPU fallback globally
    skmetal.set_verbose(True)             # log dispatch decisions
    skmetal.set_threshold(100_000)        # global min rows for GPU
    skmetal.update_threshold("KMeans",    # per-estimator override
                             min_rows=100_000, min_cols=50)
    skmetal.reset_thresholds()            # restore defaults

    config = skmetal.get_config()
    print(config)

Transparent fallback: On non-Apple-Silicon machines skmetal imports
cleanly and all estimators fall back to scikit-learn CPU implementations.
Check ``skmetal.METAL_AVAILABLE`` at runtime to detect GPU support.
"""

from ._about import __version__, __version_info__
from ._config import get_config, set_device, set_threshold, set_dtype, set_verbose, set_thresholds, update_threshold, reset_thresholds
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
    "reset_thresholds",
]

