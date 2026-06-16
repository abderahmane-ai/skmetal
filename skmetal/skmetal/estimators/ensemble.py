import numpy as np
from ._base import BaseGPUEstimator
from .._bridge import tree_predict_all
from sklearn.ensemble import HistGradientBoostingRegressor as _SKHistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier as _SKHistGradientBoostingClassifier


# All constructor parameters shared between HGBT Regressor and Classifier.
_HGBT_PARAMS = [
    "loss", "learning_rate", "max_iter", "max_leaf_nodes", "max_depth",
    "min_samples_leaf", "l2_regularization", "max_bins",
    "categorical_features", "monotonic_cst", "interaction_cst",
    "warm_start", "early_stopping", "scoring", "validation_fraction",
    "n_iter_no_change", "tol", "verbose", "random_state",
]


def _clone_hist_estimator(estimator, target_cls):
    """Copy constructor params from wrapped estimator to a fresh CPU HGBT."""
    kwargs = {p: getattr(estimator, p) for p in _HGBT_PARAMS}
    # class_weight is Classifier-only (not on Regressor)
    if hasattr(estimator, "class_weight"):
        kwargs["class_weight"] = estimator.class_weight
    return target_cls(**kwargs)


class MetalHistGradientBoostingBase(BaseGPUEstimator):
    def __init__(self, _estimator=None):
        super().__init__(_estimator)
        self._flat_values = None
        self._flat_feature = None
        self._flat_threshold = None
        self._flat_left = None
        self._flat_right = None
        self._flat_is_leaf = None
        self._tree_offsets = None
        self._tree_n_nodes = None
        self._baseline_val = np.array([0.0], dtype=np.float32)
        self._n_trees = 0

    def _fallback_fit(self, X, y, **kwargs):
        result = super()._fallback_fit(X, y, **kwargs)
        self._baseline_val = np.array([float(self._estimator._baseline_prediction.flat[0])], dtype=np.float32)
        self._extract_trees()
        return result

    def _extract_trees(self):
        all_values, all_feature = [], []
        all_threshold, all_left, all_right = [], [], []
        all_is_leaf = []
        offsets, n_nodes_list = [], []
        offset = 0
        for stage in self._estimator._predictors:
            for tree in stage:
                nodes = tree.nodes
                n = len(nodes)
                offsets.append(offset)
                n_nodes_list.append(n)
                offset += n
                all_values.append(nodes["value"].astype(np.float32))
                all_feature.append(np.where(nodes["is_leaf"], -1, nodes["feature_idx"]).astype(np.int32))
                all_threshold.append(nodes["num_threshold"].astype(np.float32))
                all_left.append(nodes["left"].astype(np.int32))
                all_right.append(nodes["right"].astype(np.int32))
                all_is_leaf.append(nodes["is_leaf"].astype(np.uint8))
        self._flat_values = np.concatenate(all_values)
        self._flat_feature = np.concatenate(all_feature)
        self._flat_threshold = np.concatenate(all_threshold)
        self._flat_left = np.concatenate(all_left)
        self._flat_right = np.concatenate(all_right)
        self._flat_is_leaf = np.concatenate(all_is_leaf)
        self._tree_offsets = np.array(offsets, dtype=np.uint32)
        self._tree_n_nodes = np.array(n_nodes_list, dtype=np.uint32)
        self._n_trees = len(offsets)

    def _fit_hist(self, X, y, target_cls):
        """Shared fit: build CPU estimator, extract trees for GPU predict."""
        cpu_est = _clone_hist_estimator(self._estimator, target_cls)
        cpu_est.fit(X, y)

        self._estimator = cpu_est
        self._baseline_val = np.array([float(cpu_est._baseline_prediction.flat[0])], dtype=np.float32)
        self._extract_trees()
        self._fitted = True
        return self

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        if self._flat_values is None:
            return self._fallback_predict(X)

        n = X.shape[0]
        predictions = np.empty(n, dtype=np.float32)
        tree_predict_all(
            X, self._flat_values, self._flat_feature,
            self._flat_threshold, self._flat_left, self._flat_right,
            self._flat_is_leaf, self._tree_offsets, self._tree_n_nodes,
            self._baseline_val, predictions,
        )
        return predictions


class MetalHistGradientBoostingRegressor(MetalHistGradientBoostingBase):
    def fit(self, X, y, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)
        return self._fit_hist(X, y.astype(np.float64), _SKHistGradientBoostingRegressor)


class MetalHistGradientBoostingClassifier(MetalHistGradientBoostingBase):
    def fit(self, X, y, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)
        return self._fit_hist(X, y, _SKHistGradientBoostingClassifier)

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        raw = super().predict(X)
        n_classes = len(self._estimator.classes_)
        if n_classes == 2:
            proba = 1.0 / (1.0 + np.exp(-raw))
            return np.where(proba >= 0.5, self._estimator.classes_[1], self._estimator.classes_[0])
        n = len(raw)
        raw_2d = raw.reshape(n, -1)
        return self._estimator.classes_[np.argmax(raw_2d, axis=1)]

    def predict_proba(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict_proba(X)
        raw = super().predict(X)
        n_classes = len(self._estimator.classes_)
        if n_classes == 2:
            proba = 1.0 / (1.0 + np.exp(-raw))
            return np.column_stack([1 - proba, proba])
        raw_2d = raw.reshape(len(X), -1)
        exp_scores = np.exp(raw_2d - raw_2d.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
