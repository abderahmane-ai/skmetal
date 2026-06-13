"""Configuration API for skmetal."""

import threading


PER_ESTIMATOR_THRESHOLDS = {
    "StandardScaler":    (1_000,   10),
    "MinMaxScaler":      (500_000, 10),
    "RobustScaler":      (100_000, 10),

    "TruncatedSVD":      (5_000,   20),
    "Ridge":             (100_000, 50),
    "Lasso":             (50_000,  50),
    "ElasticNet":        (50_000,  50),
    "LinearRegression":  (50_000,  50),
    "LogisticRegression":(20_000,  20),
    "KMeans":            (5_000,   5),
    "DBSCAN":            (1_000,   2),
    "KNeighborsClassifier":  (5_000,   10),
    "KNeighborsRegressor":   (5_000,   10),
    "HistGradientBoostingClassifier":   (10_000, 10),
    "HistGradientBoostingRegressor":    (10_000, 10),
    "GaussianNB":        (10_000,  10),
}


class Config:
    def __init__(self, device: str = "gpu", threshold: int = 1, dtype: str = "float32", verbose: bool = False, thresholds: dict = None):
        self._device = device
        self.threshold = threshold
        self.dtype = dtype
        self.verbose = verbose
        self.thresholds = thresholds if thresholds is not None else dict(PER_ESTIMATOR_THRESHOLDS)

    @property
    def device(self) -> str:
        val = getattr(_thread_local, "device", _UNSET)
        return self._device if val is _UNSET else val

    @device.setter
    def device(self, val: str):
        self._device = val


_config = Config()
_lock = threading.Lock()

# Thread-local store for the `device` override set by accelerate_context.
# Each thread starts with _UNSET (inherits the global default).
_thread_local = threading.local()
_UNSET = object()


def get_config() -> Config:
    return _config


def _get_device() -> str:
    """Return the device for the current thread.

    If accelerate_context has set a per-thread override, that takes precedence;
    otherwise the global Config value is returned.
    """
    val = getattr(_thread_local, "device", _UNSET)
    return _config._device if val is _UNSET else val


def _set_thread_device(device: str | None) -> None:
    """Set or clear the per-thread device override used by accelerate_context."""
    if device is None:
        if hasattr(_thread_local, "device"):
            del _thread_local.device
    else:
        _thread_local.device = device


def set_device(device: str) -> None:
    with _lock:
        if device not in ("gpu", "cpu", "auto"):
            raise ValueError("device must be 'gpu', 'cpu', or 'auto'")
        _config._device = device
        # Also update the thread-local so callers that set global device and
        # then read via get_config() still see a consistent value.
        _thread_local.device = device



def set_threshold(threshold: int) -> None:
    with _lock:
        _config.threshold = int(threshold)


def set_dtype(dtype: str) -> None:
    with _lock:
        if dtype == "float64":
            raise ValueError("Only float32 is supported for GPU operations.")
        if dtype != "float32":
            raise ValueError("dtype must be 'float32'")
        _config.dtype = dtype


def set_verbose(verbose: bool) -> None:
    with _lock:
        _config.verbose = bool(verbose)


def set_thresholds(thresholds: dict) -> None:
    with _lock:
        _config.thresholds = dict(thresholds)


def update_threshold(name: str, min_rows: int, min_cols: int) -> None:
    with _lock:
        _config.thresholds[name] = (min_rows, min_cols)
