import sys
import importlib
from pathlib import Path

from .skmetal._about import __version__, __version_info__

# Make the inner skmetal/skmetal/ directory discoverable as a submodule source
# so that ``from skmetal.accelerate import accelerate`` resolves correctly.
_inner = str(Path(__file__).resolve().parent / "skmetal")
if _inner not in __path__:
    __path__.append(_inner)

# Canonical import of the inner _config module.
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
    "__version__",
    "__version_info__",
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
