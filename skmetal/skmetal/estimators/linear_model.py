import numpy as np
from ._base import BaseGPUEstimator
from .._bridge import ridge_fit_solve, logreg_irls_fit, logreg_lbfgs_fit, multinomial_lbfgs_fit, fista_fit, gemm


class MetalLinearRegression(BaseGPUEstimator):
    def fit(self, X, y, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n, p = X.shape
        fit_intercept = self._estimator.fit_intercept
        X = X.copy()

        X_mean = np.empty(p, dtype=np.float32)
        coef = np.empty(p, dtype=np.float32)

        ridge_fit_solve(X, y, X_mean, coef, alpha=0.0)

        self._estimator.coef_ = coef
        if fit_intercept:
            self._estimator.intercept_ = float(y.mean()) - X_mean @ coef
        else:
            self._estimator.intercept_ = 0.0
        self._fitted = True
        return self

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        w = self._estimator.coef_.reshape(-1, 1)
        return gemm(X, w).ravel() + self._estimator.intercept_


class MetalRidge(BaseGPUEstimator):
    def fit(self, X, y, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        X = X.copy()

        n, p = X.shape
        alpha = self._estimator.alpha
        fit_intercept = self._estimator.fit_intercept

        X_mean = np.empty(p, dtype=np.float32)
        coef = np.empty(p, dtype=np.float32)

        ridge_fit_solve(X, y, X_mean, coef, alpha)

        self._estimator.coef_ = coef
        if fit_intercept:
            self._estimator.intercept_ = float(y.mean()) - X_mean @ coef
        else:
            self._estimator.intercept_ = 0.0
        self._fitted = True
        return self

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        w = self._estimator.coef_.reshape(-1, 1)
        return gemm(X, w).ravel() + self._estimator.intercept_


class MetalLasso(BaseGPUEstimator):
    def fit(self, X, y, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n, p = X.shape
        alpha = self._estimator.alpha
        tol = self._estimator.tol
        max_iter = self._estimator.max_iter
        fit_intercept = self._estimator.fit_intercept

        if fit_intercept:
            X_mean = X.mean(axis=0, dtype=np.float32)
            y_mean = float(y.mean())
            Xc = X - X_mean
            yc = y - y_mean
        else:
            Xc = X
            yc = y

        coef, n_iter = fista_fit(Xc, yc, alpha, l1_ratio=1.0, tol=tol, max_iter=max_iter)

        self._estimator.coef_ = coef
        if fit_intercept:
            self._estimator.intercept_ = y_mean - X_mean @ coef
        else:
            self._estimator.intercept_ = 0.0
        self._estimator.n_iter_ = n_iter
        self._fitted = True
        return self

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        w = self._estimator.coef_.reshape(-1, 1)
        return gemm(X, w).ravel() + self._estimator.intercept_


class MetalElasticNet(BaseGPUEstimator):
    def fit(self, X, y, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n, p = X.shape
        alpha = self._estimator.alpha
        l1_ratio = self._estimator.l1_ratio
        tol = self._estimator.tol
        max_iter = self._estimator.max_iter
        fit_intercept = self._estimator.fit_intercept

        if fit_intercept:
            X_mean = X.mean(axis=0, dtype=np.float32)
            y_mean = float(y.mean())
            Xc = X - X_mean
            yc = y - y_mean
        else:
            Xc = X
            yc = y

        coef, n_iter = fista_fit(Xc, yc, alpha, l1_ratio=l1_ratio, tol=tol, max_iter=max_iter)

        self._estimator.coef_ = coef
        if fit_intercept:
            self._estimator.intercept_ = y_mean - X_mean @ coef
        else:
            self._estimator.intercept_ = 0.0
        self._estimator.n_iter_ = n_iter
        self._fitted = True
        return self

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        w = self._estimator.coef_.reshape(-1, 1)
        return gemm(X, w).ravel() + self._estimator.intercept_


class MetalLogisticRegression(BaseGPUEstimator):
    def fit(self, X, y, **kwargs):
        # Pop solver kwarg before passing through to fallback
        solver = kwargs.pop("solver", "irls")
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n, p = X.shape
        C = self._estimator.C
        tol = self._estimator.tol
        max_iter = self._estimator.max_iter
        fit_intercept = self._estimator.fit_intercept
        penalty = self._estimator.penalty

        classes = np.unique(y)
        n_classes = len(classes)
        if n_classes == 2:
            coef_, intercept_ = self._fit_binary(
                X, y, C, tol, max_iter, fit_intercept, penalty, pos_label=classes[1], solver=solver
            )
            self._estimator.coef_ = coef_.reshape(1, -1)
            self._estimator.intercept_ = np.array([intercept_])
        else:
            self._fit_multinomial(X, y, classes, n, p, n_classes, tol, max_iter, fit_intercept, penalty)

        self._estimator.classes_ = classes
        self._fitted = True
        return self

    def _fit_binary(self, X, y, C, tol, max_iter, fit_intercept, penalty, pos_label, solver="irls"):
        n = X.shape[0]

        if fit_intercept:
            ones = np.ones(n, dtype=np.float32)
            Xe = np.column_stack([X, ones])
        else:
            Xe = X

        if solver == "lbfgs":
            coef, n_iter = logreg_lbfgs_fit(Xe, y, C, tol, max_iter, fit_intercept)
        else:
            coef, n_iter = logreg_irls_fit(Xe, y, C, tol, max_iter, fit_intercept)
        self._estimator.n_iter_ = [n_iter]

        if fit_intercept:
            w_final = coef[:-1].copy()
            b_final = float(coef[-1])
        else:
            w_final = coef.copy()
            b_final = 0.0

        return w_final, b_final

    def _fit_multinomial(self, X, y, classes, n, p, n_classes, tol, max_iter, fit_intercept, penalty):
        # One-shot label encoding via np.unique (pure C, much faster than Python loop for large n)
        _, y_enc = np.unique(y, return_inverse=True)
        y_enc = y_enc.astype(np.float32)

        if fit_intercept:
            ones = np.ones((n, 1), dtype=np.float32)
            Xe = np.column_stack([X, ones])
        else:
            Xe = X

        # Full multinomial L-BFGS loop in Swift — robust to collinear features
        reg_C = self._estimator.C
        W, n_iter = multinomial_lbfgs_fit(Xe, y_enc, reg_C, tol, max_iter, n_classes)
        self._estimator.n_iter_ = [n_iter]

        if fit_intercept:
            self._estimator.coef_ = W[:-1].T.copy()
            self._estimator.intercept_ = W[-1].copy()
        else:
            self._estimator.coef_ = W.T.copy()
            self._estimator.intercept_ = np.zeros(n_classes, dtype=np.float32)

        self._estimator.classes_ = classes

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        coef = self._estimator.coef_
        scores = gemm(X, coef, trans_B=True) + self._estimator.intercept_
        if scores.shape[1] == 1:
            return self._estimator.classes_[(scores.ravel() > 0).astype(int)]
        return self._estimator.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict_proba(X)
        coef = self._estimator.coef_
        scores = gemm(X, coef, trans_B=True) + self._estimator.intercept_
        if scores.shape[1] == 1:
            prob = 1.0 / (1.0 + np.exp(-scores.ravel()))
            return np.column_stack([1 - prob, prob])
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)


