"""Configuration API for skmetal."""

import threading


PER_ESTIMATOR_THRESHOLDS: dict[str, tuple[int, int]] = {
    # --- GPU winners (benchmarked at 200K×500 / 100K×200 / 1M×100) ---
    "StandardScaler": (1_000, 10),  # 6.05× GPU (1M×100)
    "LinearRegression": (50_000, 50),  # 8.15× GPU (200K×500)
    "TruncatedSVD": (5_000, 20),  # 3.59× GPU (100K×500)
    "ElasticNet": (50_000, 50),  # 1.53× GPU (100K×200)
    "Lasso": (50_000, 50),  # 1.40× GPU (100K×200)
    "LogisticRegression": (500_000, 500),  # 0.92× tied; likely wins for p>500
    # --- CPU wins at benchmark sizes (keep on CPU) ---
    "Ridge": (10_000_000, 10_000),  # 0.91× CPU (Accelerate sub-ms, dispatch overhead)
    "MinMaxScaler": (500_000, 10),  # 1.11× GPU (1M×100) — marginal, high threshold
    "KNeighborsClassifier": (5_000_000, 10_000),  # 0.62× CPU (100K×200)
    "KNeighborsRegressor": (5_000_000, 10_000),  # 0.62× CPU (same kernel)
    "KMeans": (10_000, 10),  # 0.67× CPU on Apple GPU; 94-517× via MLX backend (M3 Ultra)
    # --- Conservative defaults (not benchmarked) ---
    "RobustScaler": (100_000, 10),
    "DBSCAN": (1_000, 2),
    "HistGradientBoostingClassifier": (10_000, 10),
    "HistGradientBoostingRegressor": (10_000, 10),
    "GaussianNB": (10_000, 10),
    # --- SVM (uses RBF Gram on GPU; matrix-free predict) ---
    "SVC": (5_000, 10),
    "SVR": (5_000, 10),
    # --- NearestNeighbors (unsupervised; GPU pairwise distance) ---
    "NearestNeighbors": (5_000, 10),
}


class Config:
    def __init__(
        self,
        device: str = "gpu",
        threshold: int = 1,
        dtype: str = "float32",
        verbose: bool = False,
        thresholds: dict = None,
    ):
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

    def __repr__(self):
        n_gpu = sum(1 for v in self.thresholds.values() if v[0] < 1e8)
        n_cpu = len(self.thresholds) - n_gpu
        return (
            f"device={self.device}, threshold={self.threshold}, verbose={self.verbose}\n"
            f"  estimators routed to GPU: {n_gpu}  |  estimators routed to CPU: {n_cpu}"
        )


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
        if dtype != "float32":
            raise ValueError("Only float32 is supported for GPU operations.")
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


def reset_thresholds() -> None:
    """Restore per-estimator thresholds to default values."""
    with _lock:
        _config.thresholds = dict(PER_ESTIMATOR_THRESHOLDS)
