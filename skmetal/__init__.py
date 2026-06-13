import sys
import importlib

__version__ = "0.1.0"

# ── Canonical import of the inner _config module ──────────────────────────
# We import it by its canonical name (*.skmetal._config) and then alias it
# under the top-level namespace so that ``import skmetal._config`` and
# ``import skmetal.skmetal._config`` always refer to the *same* module
# object (and therefore the same Config singleton).
_config_mod = importlib.import_module("skmetal.skmetal._config")
sys.modules["skmetal._config"] = _config_mod

get_config = _config_mod.get_config
set_device = _config_mod.set_device
set_threshold = _config_mod.set_threshold
set_dtype = _config_mod.set_dtype
set_verbose = _config_mod.set_verbose
set_thresholds = _config_mod.set_thresholds
update_threshold = _config_mod.update_threshold

from .skmetal.accelerate import accelerate, accelerate_context
from .skmetal._bridge import device_info

# Fallback __getattr__ for any remaining inner submodule (e.g. estimators).
def __getattr__(name):
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
    "set_thresholds",
    "update_threshold",
    "device_info",
]
