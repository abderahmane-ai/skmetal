"""Tests for estimator dispatch logic."""

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import skmetal
from skmetal._dispatch import _is_supported, _wrap_estimator, _wrap_pipeline
from skmetal.estimators._registry import GPU_ESTIMATORS


def test_all_registered_estimators_are_supported():
    """Every estimator in the registry should be detected as supported."""
    for sklearn_cls in GPU_ESTIMATORS:
        assert _is_supported(sklearn_cls())


def test_unregistered_estimator_not_supported():
    """An estimator not in the registry should not be supported."""

    class FakeEstimator:
        def fit(self, X, y=None):
            pass

        def predict(self, X):
            return X

    assert not _is_supported(FakeEstimator())


def test_wrap_single_estimator():
    """Wrapping a supported estimator should return a GPU wrapper."""
    lr = LinearRegression()
    wrapped = _wrap_estimator(lr)
    assert hasattr(wrapped, "_estimator")
    assert wrapped._estimator is lr


def test_wrap_unsupported_estimator():
    """Wrapping an unsupported estimator should return it unchanged."""

    class FakeEstimator:
        def fit(self, X, y=None):
            pass

        def predict(self, X):
            return X

    fake = FakeEstimator()
    wrapped = _wrap_estimator(fake)
    assert wrapped is fake


def test_wrap_pipeline():
    """Pipeline wrapping should replace supported steps."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ])
    wrapped = _wrap_pipeline(pipe)
    assert isinstance(wrapped, Pipeline)
    assert hasattr(wrapped.steps[0][1], "_estimator")
    assert hasattr(wrapped.steps[1][1], "_estimator")


def test_accelerate_decorator():
    """@accelerate should wrap estimators and pipelines."""
    lr = skmetal.accelerate(LinearRegression())
    assert hasattr(lr, "_estimator")

    pipe = skmetal.accelerate(Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ]))
    assert isinstance(pipe, Pipeline)
    assert hasattr(pipe.steps[0][1], "_estimator")


def test_accelerate_context_manager():
    """Context manager should toggle device config."""
    with skmetal.accelerate_context(enabled=False):
        from skmetal._config import get_config
        assert get_config().device == "cpu"

    from skmetal._config import get_config
    assert get_config().device == "gpu"
