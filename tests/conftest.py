"""Shared fixtures for all skmetal tests."""
import pytest
from skmetal._config import get_config, PER_ESTIMATOR_THRESHOLDS


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config to defaults before every test to avoid cross-test pollution."""
    cfg = get_config()
    cfg.device = "gpu"
    cfg.threshold = 1
    cfg.dtype = "float32"
    cfg.verbose = False
    cfg.thresholds = dict(PER_ESTIMATOR_THRESHOLDS)
    yield
