"""@accelerate decorator for enabling GPU acceleration."""

import warnings
from functools import wraps
from sklearn.pipeline import Pipeline
from ._dispatch import _wrap_estimator, _wrap_pipeline
from ._config import get_config


def accelerate(obj=None):
    """
    Enable GPU acceleration for scikit-learn estimators.

    Can be used as a decorator (recommended) or as a function call:

        @accelerate
        def model():
            return LinearRegression()

        @accelerate
        def pipeline():
            return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])

    Function-call form also supported:

        model = accelerate(LinearRegression())
    """
    if obj is None:
        return _Accelerator()

    if isinstance(obj, Pipeline):
        return _wrap_pipeline(obj)

    if isinstance(obj, type):
        return _Accelerator._wrap_class(obj)

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

    warnings.warn(f"accelerate: {type(obj).__name__} is not a recognized estimator or pipeline.")
    return obj


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
    """Context manager to temporarily enable/disable acceleration."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.prev = None

    def __enter__(self):
        self.prev = get_config().device
        get_config().device = "gpu" if self.enabled else "cpu"
        return self

    def __exit__(self, *args):
        get_config().device = self.prev
