"""Tests for skmetal configuration API."""
import pytest
from skmetal._config import get_config, set_device, set_threshold, set_dtype, set_verbose, set_thresholds


class TestConfig:
    def test_defaults(self):
        cfg = get_config()
        assert cfg.device == "gpu"
        assert cfg.threshold == 1
        assert cfg.dtype == "float32"
        assert cfg.verbose is False

    def test_set_device_gpu(self):
        set_device("gpu")
        assert get_config().device == "gpu"

    def test_set_device_cpu(self):
        set_device("cpu")
        assert get_config().device == "cpu"

    def test_set_device_auto(self):
        set_device("auto")
        assert get_config().device == "auto"

    def test_set_device_invalid(self):
        with pytest.raises(ValueError, match="device must be"):
            set_device("invalid")

    def test_set_threshold(self):
        set_threshold(500)
        assert get_config().threshold == 500

    def test_set_threshold_zero(self):
        set_threshold(0)
        assert get_config().threshold == 0

    def test_set_threshold_string(self):
        set_threshold("42")
        assert get_config().threshold == 42

    def test_set_dtype_float32(self):
        set_dtype("float32")
        assert get_config().dtype == "float32"

    def test_set_dtype_float64_rejected(self):
        with pytest.raises(ValueError, match="Only float32"):
            set_dtype("float64")

    def test_set_dtype_unknown_rejected(self):
        with pytest.raises(ValueError, match="Only float32"):
            set_dtype("float16")

    def test_set_verbose_true(self):
        set_verbose(True)
        assert get_config().verbose is True

    def test_set_verbose_false(self):
        set_verbose(False)
        assert get_config().verbose is False

    def test_set_thresholds(self):
        custom = {"LinearRegression": (1000, 50), "KMeans": (500, 10)}
        set_thresholds(custom)
        assert get_config().thresholds["LinearRegression"] == (1000, 50)
        assert get_config().thresholds["KMeans"] == (500, 10)

    def test_set_thresholds_immutable_copy(self):
        original = {"Test": (100, 10)}
        set_thresholds(original)
        original["Test"] = (999, 999)
        assert get_config().thresholds["Test"] == (100, 10)
