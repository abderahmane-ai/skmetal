"""MLX backend registry — centralized MLX availability, version, and capability detection."""
from sklearn.decomposition import TruncatedSVD

_HAS_MLX = False
_MLX_VERSION = None
_MLX_CAPABILITIES = {
    "has_svd": False,
    "has_compile": False,
}

try:
    import mlx.core as mx
    _HAS_MLX = True
    _MLX_VERSION = getattr(mx, "__version__", "unknown")
    _MLX_CAPABILITIES["has_compile"] = hasattr(mx, "compile")
    _MLX_CAPABILITIES["has_svd"] = hasattr(mx.linalg, "svd")
except ImportError:
    pass


def has_mlx() -> bool:
    return _HAS_MLX


def mlx_version() -> str:
    return _MLX_VERSION or "unavailable"


def mlx_capabilities() -> dict:
    return _MLX_CAPABILITIES.copy()


MLX_REGISTRY: dict[type, tuple[str, str]] = {
    TruncatedSVD: ("skmetal.estimators._mlx_svd", "MetalTruncatedSVDMLX"),
}

_MLX_CLASS_NAMES: dict[type, str] = {
    TruncatedSVD: "MetalTruncatedSVDMLX",
}

__all__ = [
    "has_mlx",
    "mlx_version",
    "mlx_capabilities",
]
