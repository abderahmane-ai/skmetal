import importlib

__version__ = "0.1.0"


def __getattr__(name):
    """Lazy-load attributes from the inner skmetal package."""
    inner = importlib.import_module("skmetal.skmetal")
    return getattr(inner, name)


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
