"""Utility functions for skmetal."""

import numpy as np


def ensure_f32_contiguous(arr, gpu_threshold=1024 * 1024):
    """Ensure array is float32 C-contiguous, using GPU for conversion of large arrays.

    GPU path is used when arr.nbytes >= gpu_threshold (default 1MB).
    Returns the converted array (same array if already f32 C-contiguous).
    """
    if arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]:
        return arr

    n = arr.size

    if n < gpu_threshold // max(arr.dtype.itemsize, 4):
        return np.ascontiguousarray(arr, dtype=np.float32)

    from ._bridge import transpose_f32

    needs_cast = arr.dtype == np.float64
    needs_transpose = not arr.flags["C_CONTIGUOUS"]

    if needs_cast:
        arr = np.ascontiguousarray(arr, dtype=np.float32)

    if needs_transpose and arr.ndim == 2:
        rows, cols = arr.shape
        out = np.empty((cols, rows), dtype=np.float32)
        transpose_f32(arr, out)
        return out

    return arr


def get_device_info():
    """Get Metal device information."""
    from ._bridge import device_info
    return device_info()


def is_metal_available():
    """Check if Metal is available on this system."""
    try:
        from ._bridge import init
        return init() == 0
    except Exception:
        return False
