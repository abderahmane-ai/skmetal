"""MLX backend registry — centralized MLX availability, version, and capability detection."""

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

_HAS_MLX = False
_HAS_FLASH_KMEANS = False
_MLX_VERSION = None
_MLX_CAPABILITIES = {
    "has_svd": False,
    "has_compile": False,
    "has_flash_kmeans": False,
}

try:
    import mlx.core as mx

    _HAS_MLX = True
    _MLX_VERSION = getattr(mx, "__version__", "unknown")
    _MLX_CAPABILITIES["has_compile"] = hasattr(mx, "compile")
    _MLX_CAPABILITIES["has_svd"] = hasattr(mx.linalg, "svd")
except ImportError:
    pass

if _HAS_MLX:
    try:
        from flash_kmeans_mlx import batch_kmeans_Euclid  # noqa: F401

        _HAS_FLASH_KMEANS = True
        _MLX_CAPABILITIES["has_flash_kmeans"] = True
    except ImportError:
        pass


def has_mlx() -> bool:
    return _HAS_MLX


def has_flash_kmeans() -> bool:
    return _HAS_FLASH_KMEANS


def mlx_version() -> str:
    return _MLX_VERSION or "unavailable"


def mlx_capabilities() -> dict:
    return _MLX_CAPABILITIES.copy()


MLX_REGISTRY: dict[type, tuple[str, str]] = {
    TruncatedSVD: ("skmetal.estimators._mlx_svd", "MetalTruncatedSVDMLX"),
}
if _HAS_FLASH_KMEANS:
    MLX_REGISTRY[KMeans] = ("skmetal.estimators._mlx_kmeans", "MetalKMeansMLX")

_MLX_CLASS_NAMES: dict[type, str] = {
    TruncatedSVD: "MetalTruncatedSVDMLX",
}
if _HAS_FLASH_KMEANS:
    _MLX_CLASS_NAMES[KMeans] = "MetalKMeansMLX"

__all__ = [
    "has_mlx",
    "has_flash_kmeans",
    "mlx_version",
    "mlx_capabilities",
]
