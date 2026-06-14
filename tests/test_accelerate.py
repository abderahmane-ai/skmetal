"""Tests for accelerator decorator, context manager, and dispatch edge cases."""
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import skmetal
from skmetal._dispatch import _is_supported, _wrap_estimator


class TestIsSupported:
    def test_supported_estimator(self):
        assert _is_supported(LinearRegression()) is True

    def test_unsupported_estimator(self):
        rf = RandomForestClassifier()
        assert _is_supported(rf) is False

    def test_custom_class_looks_like_sklearn(self):
        class FakeEstimator:
            def fit(self, X, y=None): pass
            def predict(self, X): return X
        assert _is_supported(FakeEstimator()) is False


class TestWrapEstimator:
    def test_wrap_supported(self):
        lr = LinearRegression()
        wrapped = _wrap_estimator(lr)
        assert hasattr(wrapped, "_estimator")
        assert wrapped._estimator is lr

    def test_wrap_unsupported_returns_original(self):
        rf = RandomForestClassifier()
        wrapped = _wrap_estimator(rf)
        assert wrapped is rf
        assert not hasattr(wrapped, "_estimator")

    def test_wrap_already_wrapped_is_idempotent(self):
        lr = skmetal.accelerate(LinearRegression())
        lr2 = skmetal.accelerate(lr)
        assert lr2 is lr

    def test_wrap_double_wrap_estimator(self):
        lr = skmetal.accelerate(LinearRegression())
        lr2 = _wrap_estimator(lr)
        assert lr2 is lr


class TestAccelerateDecorator:
    def test_accelerate_linear_regression(self):
        lr = skmetal.accelerate(LinearRegression())
        assert hasattr(lr, "_estimator")
        assert hasattr(lr, "fit")
        assert hasattr(lr, "predict")

    def test_accelerate_pipeline(self):
        pipe = skmetal.accelerate(Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ]))
        assert isinstance(pipe, Pipeline)
        assert hasattr(pipe.steps[0][1], "_estimator")
        assert hasattr(pipe.steps[1][1], "_estimator")

    def test_accelerate_unsupported_returns_original(self):
        rf = skmetal.accelerate(RandomForestClassifier())
        assert not hasattr(rf, "_estimator")

    def test_accelerate_pipeline_with_unsupported(self):
        pipe = skmetal.accelerate(Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier()),
        ]))
        assert hasattr(pipe.steps[0][1], "_estimator")  # StandardScaler wrapped
        assert not hasattr(pipe.steps[1][1], "_estimator")  # RF untouched

    def test_accelerate_pipeline_all_unsupported(self):
        pipe = skmetal.accelerate(Pipeline([
            ("rf1", RandomForestClassifier()),
            ("rf2", RandomForestClassifier()),
        ]))
        # Neither step should be wrapped
        assert not hasattr(pipe.steps[0][1], "_estimator")
        assert not hasattr(pipe.steps[1][1], "_estimator")


class TestAccelerateContextManager:
    def test_context_manager_disables_gpu(self):
        with skmetal.accelerate_context(enabled=False):
            from skmetal._config import get_config
            assert get_config().device == "cpu"
        from skmetal._config import get_config
        assert get_config().device == "gpu"

    def test_context_manager_enables_gpu(self):
        skmetal.set_device("cpu")
        with skmetal.accelerate_context(enabled=True):
            from skmetal._config import get_config
            assert get_config().device == "gpu"
        from skmetal._config import get_config
        assert get_config().device == "cpu"

    def test_context_manager_exception_restores(self):
        skmetal.set_device("gpu")
        try:
            with skmetal.accelerate_context(enabled=False):
                raise ValueError("boom")
        except ValueError:
            pass
        assert skmetal.get_config().device == "gpu"

    def test_context_manager_nested(self):
        with skmetal.accelerate_context(enabled=False):
            assert skmetal.get_config().device == "cpu"
            with skmetal.accelerate_context(enabled=True):
                assert skmetal.get_config().device == "gpu"
            assert skmetal.get_config().device == "cpu"
        assert skmetal.get_config().device == "gpu"


class TestAccelerateFunctionCall:
    def test_function_call_wraps(self):
        model = skmetal.accelerate(LinearRegression())
        assert hasattr(model, "_estimator")


class TestAccelerateTypeCheck:
    def test_accelerate_wrong_type_warns(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = skmetal.accelerate("not an estimator")
            assert len(w) == 1
            assert "not a recognized estimator" in str(w[0].message)
        assert result is not None
