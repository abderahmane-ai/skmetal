"""@accelerate decorator for enabling GPU acceleration."""

import threading
import warnings
from functools import wraps
from sklearn.pipeline import Pipeline
from ._dispatch import _wrap_estimator, _wrap_pipeline
from ._config import _get_device, _set_thread_device

# Thread-local storage for nested accelerate_context stacks
_local = threading.local()


def accelerate(obj=None):
    """
    Enable GPU acceleration for scikit-learn estimators.

    Can be used as a decorator (recommended) or as a function call::

        @accelerate
        def model():
            return LinearRegression()

        @accelerate
        def pipeline():
            return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])

    Function-call form also supported::

        model = accelerate(LinearRegression())
    """
    if obj is None:
        return _Accelerator()
    return _Accelerator()(obj)


class _Accelerator:
    """Internal helper for @accelerate without arguments."""

    @staticmethod
    def _wrap_class(cls):
        """Wrap a class so its instances are accelerated."""
        orig_new = cls.__new__
        orig_init = cls.__init__

        @wraps(orig_new)
        def new_new(cls2, *args, **kwargs):
            if args and isinstance(args[0], (Pipeline,)):
                return _wrap_pipeline(args[0])
            if args and hasattr(args[0], "fit"):
                return _wrap_estimator(args[0])
            if orig_new is not object.__new__:
                return orig_new(cls2)
            return super(cls, cls2).__new__(cls2)

        @wraps(orig_init)
        def new_init(self, *args, **kwargs):
            if args and hasattr(args[0], "fit"):
                return
            orig_init(self, *args, **kwargs)

        cls.__new__ = new_new
        cls.__init__ = new_init
        return cls

    def __call__(self, obj):
        if isinstance(obj, Pipeline):
            return _wrap_pipeline(obj)
        if isinstance(obj, type):
            return self._wrap_class(obj)
        if hasattr(obj, "fit"):
            return _wrap_estimator(obj)
        if callable(obj):
            @wraps(obj)
            def wrapper(*args, **kwargs):
                result = obj(*args, **kwargs)
                if isinstance(result, Pipeline):
                    return _wrap_pipeline(result)
                if hasattr(result, "fit"):
                    return _wrap_estimator(result)
                return result
            return wrapper
        warnings.warn(f"accelerate: {type(obj).__name__} is not a recognized estimator.")
        return obj


class accelerate_context:
    """Context manager to temporarily enable/disable acceleration.

    Thread-safe: each thread maintains its own device-context stack so that
    concurrent calls from different threads cannot corrupt each other's state.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __enter__(self):
        # Per-thread stack — push current device, set new one.
        if not hasattr(_local, "device_stack"):
            _local.device_stack = []
        _local.device_stack.append(_get_device())
        _set_thread_device("gpu" if self.enabled else "cpu")
        return self

    def __exit__(self, *args):
        if hasattr(_local, "device_stack") and _local.device_stack:
            prev = _local.device_stack.pop()
            _set_thread_device(prev)
