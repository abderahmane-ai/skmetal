"""Configuration API for skmetal."""

import threading
from dataclasses import dataclass


@dataclass
class Config:
    device: str = "gpu"
    threshold: int = 500_000
    dtype: str = "float32"
    verbose: bool = False


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
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        _config.dtype = dtype


def set_verbose(verbose: bool) -> None:
    with _lock:
        _config.verbose = bool(verbose)