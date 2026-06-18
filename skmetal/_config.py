"""Configuration API for skmetal."""

import threading


PER_ESTIMATOR_THRESHOLDS: dict[str, tuple[int, int]] = {
    # --- GPU winners (benchmarked at 200K×500 / 100K×200 / 1M×100) ---
    "StandardScaler": (1_000, 10),  # 9.5× GPU (1M×100)
    "LinearRegression": (50_000, 50),  # 10.0× GPU (200K×500)
    "TruncatedSVD": (5_000, 20),  # 3.1× GPU (100K×500)
    "PCA": (5_000, 10),  # GPU randomized SVD (centering on CPU, GEMM on GPU)
    "ElasticNet": (50_000, 50),  # 1.53× GPU (100K×200)
    "Lasso": (50_000, 50),  # 1.40× GPU (100K×200)
    "LogisticRegression": (500_000, 500),  # 0.92× tied; likely wins for p>500
    # --- CPU wins at benchmark sizes (keep on CPU) ---
    "Ridge": (10_000_000, 10_000),  # 0.91× CPU (Accelerate sub-ms, dispatch overhead)
    "MinMaxScaler": (500_000, 10),  # 1.11× GPU (1M×100) — marginal, high threshold
    "KNeighborsClassifier": (5_000_000, 10_000),  # 0.62× CPU (100K×200)
    "KNeighborsRegressor": (5_000_000, 10_000),  # 0.62× CPU (same kernel)
    "KMeans": (1_000, 10),  # 7.8× GPU via flash-kmeans-mlx (MLX backend); n_init=1 matches k-means++ quality
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
    """Global configuration for skmetal GPU dispatch.

    Thread-safe via ``threading.Lock``. Holds device preference, global
    dispatch threshold, compute dtype, verbosity flag, and per-estimator
    (min_rows, min_cols) overrides.

    Attributes
    ----------
    device : str
        ``"gpu"`` (Metal), ``"cpu"`` (sklearn fallback), or ``"auto"``.
    threshold : int
        Minimum ``n_samples * n_features`` for GPU dispatch.
    dtype : str
        Compute dtype. Only ``"float32"`` is supported (Apple GPU lacks float64).
    verbose : bool
        When True, log GPU/CPU dispatch decisions to stderr.
    thresholds : dict
        Per-estimator overrides mapping class name → (min_rows, min_cols).
    """

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
    """Return the current thread-local configuration.

    Returns a ``Config`` instance. Use ``config.device`` to read the
    device setting for the current thread (may differ from the global
    default when inside an ``accelerate_context`` block).
    """
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
    """Set the global GPU device preference.

    ``"gpu"`` routes estimators to Metal (default).
    ``"cpu"`` forces every estimator to use scikit-learn CPU.
    ``"auto"`` uses per-estimator thresholds to decide.

    Per-thread override via ``accelerate_context(enabled=False)`` takes
    precedence over this global setting.
    """
    with _lock:
        if device not in ("gpu", "cpu", "auto"):
            raise ValueError("device must be 'gpu', 'cpu', or 'auto'")
        _config._device = device
        # Also update the thread-local so callers that set global device and
        # then read via get_config() still see a consistent value.
        _thread_local.device = device


def set_threshold(threshold: int) -> None:
    """Set the global minimum ``n * d`` for GPU dispatch.

    Estimators whose input has fewer than *threshold* elements fall back
    to CPU.  Larger values keep more computation on the CPU.
    """
    with _lock:
        _config.threshold = int(threshold)


def set_dtype(dtype: str) -> None:
    """Set the compute dtype for GPU operations.

    Only ``"float32"`` is supported — Apple GPUs lack native float64.
    """
    with _lock:
        if dtype != "float32":
            raise ValueError("Only float32 is supported for GPU operations.")
        _config.dtype = dtype


def set_verbose(verbose: bool) -> None:
    """Enable or disable dispatch-decision logging.

    When True, every GPU/CPU routing decision prints to stderr, showing
    which estimator was wrapped and why GPU was or was not selected.
    """
    with _lock:
        _config.verbose = bool(verbose)


def set_thresholds(thresholds: dict) -> None:
    """Replace all per-estimator thresholds at once.

    *thresholds* maps estimator class name → ``(min_rows, min_cols)``.
    ``reset_thresholds()`` restores the built-in defaults.
    """
    with _lock:
        _config.thresholds = dict(thresholds)


def update_threshold(name: str, min_rows: int, min_cols: int) -> None:
    """Override the threshold for a single estimator.

    *name* is the sklearn class name (e.g. ``"LinearRegression"``).
    GPU dispatch is skipped unless ``n >= min_rows`` and ``d >= min_cols``.
    """
    with _lock:
        _config.thresholds[name] = (min_rows, min_cols)


def reset_thresholds() -> None:
    """Restore per-estimator thresholds to default values."""
    with _lock:
        _config.thresholds = dict(PER_ESTIMATOR_THRESHOLDS)
