"""Tests that run on ANY platform — no dylib / Apple Silicon required.

These cover the Python-layer behaviour: METAL_AVAILABLE flag, the accelerate
decorator when Metal is absent, the context manager, dispatch for unsupported
estimators, and the unified registry structure.
"""

import threading

import skmetal
from skmetal._bridge import METAL_AVAILABLE
from skmetal._dispatch import _wrap_estimator, _wrap_pipeline, _is_supported
from skmetal.estimators._registry import GPU_REGISTRY, GPU_ESTIMATORS

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier  # not in GPU_REGISTRY


# ---------------------------------------------------------------------------
# 1. Basic invariants
# ---------------------------------------------------------------------------


def test_metal_available_is_bool():
    assert isinstance(METAL_AVAILABLE, bool)


def test_metal_available_exported():
    assert hasattr(skmetal, "METAL_AVAILABLE")
    assert isinstance(skmetal.METAL_AVAILABLE, bool)


def test_all_registered_estimators_are_supported():
    """Every estimator in the registry should be detected as supported."""
    for sklearn_cls in GPU_ESTIMATORS:
        assert _is_supported(sklearn_cls())


def test_registry_no_duplicate_values():
    """Every sklearn class maps to a unique (module, class) pair."""
    seen = set()
    for cls, (mod, name) in GPU_REGISTRY.items():
        key = (mod, name)
        assert key not in seen, f"Duplicate registry entry: {key}"
        seen.add(key)


def test_registry_class_names_match_metal_prefix():
    """Every GPU class name starts with 'Metal'."""
    for _, (_, name) in GPU_REGISTRY.items():
        assert name.startswith("Metal"), f"Expected 'Metal' prefix, got {name!r}"


def test_registry_module_paths_are_skmetal():
    """Every module path is inside the skmetal.estimators namespace."""
    for _, (mod, _) in GPU_REGISTRY.items():
        assert mod.startswith("skmetal.estimators."), f"Unexpected module: {mod!r}"


# ---------------------------------------------------------------------------
# 3. Dispatch: unsupported estimator is returned unchanged
# ---------------------------------------------------------------------------


def test_wrap_unsupported_estimator_returns_original():
    dt = DecisionTreeClassifier()
    result = _wrap_estimator(dt)
    assert result is dt, "Unsupported estimator should be returned as-is"


def test_is_supported_true_for_sklearn_linear_regression():
    assert _is_supported(LinearRegression())


def test_is_supported_false_for_decision_tree():
    assert not _is_supported(DecisionTreeClassifier())


# ---------------------------------------------------------------------------
# 4. Pipeline dispatch is type-safe (not name-based)
# ---------------------------------------------------------------------------


def test_wrap_pipeline_unsupported_steps_pass_through():
    """Steps without a GPU impl are returned unchanged inside a pipeline."""
    pipe = Pipeline(
        [
            ("step_named_linearregression", DecisionTreeClassifier()),  # name looks like LR but type is DT
        ]
    )
    wrapped = _wrap_pipeline(pipe)
    _, est = wrapped.steps[0]
    assert type(est) is DecisionTreeClassifier, (
        "Dispatch should be type-based, not name-based. A DecisionTreeClassifier should never be wrapped."
    )


def test_wrap_pipeline_supported_step_by_type():
    """A StandardScaler in a strangely-named step still gets wrapped by type."""
    pipe = Pipeline(
        [
            ("my_confusingly_named_linear_regression_step", StandardScaler()),
        ]
    )
    wrapped = _wrap_pipeline(pipe)
    _, est = wrapped.steps[0]
    # On Apple Silicon the type will be MetalStandardScaler;
    # on other platforms the fallback scaler is returned (still a StandardScaler
    # underneath). Either way it must NOT be a MetalLinearRegression.
    assert "LinearRegression" not in type(est).__name__, "Step dispatch must not be driven by step name substring."


# ---------------------------------------------------------------------------
# 5. accelerate_context — thread safety
# ---------------------------------------------------------------------------


def test_accelerate_context_restores_device():
    """Context manager must restore the previous device on exit."""
    from skmetal._config import get_config, set_device

    set_device("cpu")
    with skmetal.accelerate_context(enabled=True):
        assert get_config().device == "gpu"
    assert get_config().device == "cpu"


def test_accelerate_context_is_nestable():
    from skmetal._config import get_config, set_device

    set_device("cpu")
    with skmetal.accelerate_context(enabled=True):
        assert get_config().device == "gpu"
        with skmetal.accelerate_context(enabled=False):
            assert get_config().device == "cpu"
        assert get_config().device == "gpu"
    assert get_config().device == "cpu"


def test_accelerate_context_thread_isolation():
    """Two threads with different contexts must not corrupt each other."""
    from skmetal._config import _get_device

    results = {}

    def thread_gpu():
        with skmetal.accelerate_context(enabled=True):
            import time

            time.sleep(0.05)  # overlap with thread_cpu
            results["gpu"] = _get_device()

    def thread_cpu():
        with skmetal.accelerate_context(enabled=False):
            import time

            time.sleep(0.05)
            results["cpu"] = _get_device()

    t1 = threading.Thread(target=thread_gpu)
    t2 = threading.Thread(target=thread_cpu)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both threads should have seen their own setting.
    assert results.get("gpu") == "gpu"
    assert results.get("cpu") == "cpu"


# ---------------------------------------------------------------------------
# 6. accelerate decorator on an unsupported estimator
# ---------------------------------------------------------------------------


def test_accelerate_unsupported_returns_original():
    """@accelerate on an unsupported estimator returns it unchanged."""
    dt = DecisionTreeClassifier()
    wrapped = skmetal.accelerate(dt)
    assert wrapped is dt


def test_accelerate_function_wraps_result():
    """@accelerate on a factory function returns a callable."""

    @skmetal.accelerate
    def make_model():
        return LinearRegression()

    model = make_model()
    # On Apple Silicon: MetalLinearRegression; elsewhere: LinearRegression.
    # Either way it must have .fit and .predict.
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
