"""MLX registry and detection tests — no dylib required.

Tests for MLX availability detection, capability flags, and registry structure.
Runs on any platform.
"""

from sklearn.decomposition import TruncatedSVD

from skmetal.estimators._mlx_registry import (
    has_mlx,
    mlx_version,
    mlx_capabilities,
    MLX_REGISTRY,
    _MLX_CLASS_NAMES,
)


class TestMLXDetection:
    def test_has_mlx_returns_bool(self):
        assert isinstance(has_mlx(), bool)

    def test_mlx_version_returns_str(self):
        v = mlx_version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_capabilities_is_dict(self):
        caps = mlx_capabilities()
        assert isinstance(caps, dict)

    def test_capabilities_structure(self):
        caps = mlx_capabilities()
        for key in ("has_compile", "has_svd"):
            assert key in caps, f"Missing capability key: {key}"
            assert isinstance(caps[key], bool)

    def test_compile_capability(self):
        caps = mlx_capabilities()
        if has_mlx():
            assert caps["has_compile"] is True


class TestMLXRegistry:
    def test_truncated_svd_is_mapped(self):
        assert TruncatedSVD in MLX_REGISTRY

    def test_registry_values_are_tuples(self):
        for cls, entry in MLX_REGISTRY.items():
            assert isinstance(entry, tuple), f"Entry for {cls} is not a tuple"
            assert len(entry) == 2, f"Entry for {cls} should have (module, class)"

    def test_registry_module_paths_are_skmetal(self):
        for cls, (mod, _) in MLX_REGISTRY.items():
            assert mod.startswith("skmetal.estimators._mlx"), f"Bad module: {mod}"

    def test_class_names_have_mlx_suffix(self):
        for cls, name in _MLX_CLASS_NAMES.items():
            assert name.startswith("Metal"), f"Missing Metal prefix: {name}"
            assert name.endswith("MLX"), f"Missing MLX suffix: {name}"

    def test_registry_no_duplicate_class_names(self):
        names = list(_MLX_CLASS_NAMES.values())
        assert len(names) == len(set(names)), "Duplicate class names in registry"

    def test_class_names_match_registry(self):
        for cls, name in _MLX_CLASS_NAMES.items():
            assert cls in MLX_REGISTRY, f"{cls} in _MLX_CLASS_NAMES but not MLX_REGISTRY"
            assert MLX_REGISTRY[cls][1] == name, f"Class name mismatch for {cls}"
