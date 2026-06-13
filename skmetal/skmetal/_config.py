"""Configuration API for skmetal."""

import threading
from dataclasses import dataclass


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


@dataclass
class Config:
    device: str = "gpu"
    threshold: int = 1
    dtype: str = "float32"
    verbose: bool = False
    thresholds: dict = None

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = dict(PER_ESTIMATOR_THRESHOLDS)


_config = Config()
_lock = threading.Lock()


def get_config() -> Config:
    return _config


def set_device(device: str) -> None:
    with _lock:
        if device not in ("gpu", "cpu", "auto"):
            raise ValueError("device must be 'gpu', 'cpu', or 'auto'")
        _config.device = device


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