"""Tests for CPU fallback behavior."""

import numpy as np
from skmetal._fallback import validate_and_convert, should_fallback
from skmetal._config import Config


def test_validate_and_convert_float64():
    """float64 arrays should be converted to float32."""
    X = np.random.randn(100, 10).astype(np.float64)
    X_out = validate_and_convert(X)
    assert X_out.dtype == np.float32
    assert X_out.flags["C_CONTIGUOUS"]


def test_validate_and_convert_float32():
    """float32 arrays should remain float32."""
    X = np.random.randn(100, 10).astype(np.float32)
    X_out = validate_and_convert(X)
    assert X_out.dtype == np.float32


def test_fallback_cpu_device():
    """Config device='cpu' should trigger fallback."""
    config = Config(device="cpu")
    X = np.random.randn(1000, 100).astype(np.float32)
    assert should_fallback(X, config, "TestEstimator") is True


def test_fallback_small_size():
    """Small problem sizes below threshold should trigger fallback."""
    config = Config(device="gpu", threshold=1_000_000)
    X = np.random.randn(100, 100).astype(np.float32)
    assert should_fallback(X, config, "TestEstimator") is True


def test_no_fallback_large_size():
    """Large problem sizes should not trigger fallback."""
    config = Config(device="gpu", threshold=100)
    X = np.random.randn(1000, 100).astype(np.float32)
    assert should_fallback(X, config, "TestEstimator") is False


def test_fallback_non_float32():
    """Non-float32 dtypes should trigger fallback."""
    config = Config(device="gpu")
    X = np.random.randn(1000, 100).astype(np.float64)
    assert should_fallback(X, config, "TestEstimator") is True
