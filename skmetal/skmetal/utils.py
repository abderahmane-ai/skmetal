"""Utility functions for skmetal."""

import numpy as np


def ensure_c_contiguous_float32(arr):
    """Ensure array is C-contiguous float32."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
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