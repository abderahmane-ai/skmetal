import numpy as np
from ._base import BaseGPUEstimator
from .._bridge import gemm, sigmoid, ridge_fit, logreg_irls_iter, soft_threshold


class MetalLinearRegression(BaseGPUEstimator):
    def fit(self, X, y, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n, p = X.shape
        fit_intercept = self._estimator.fit_intercept

        if fit_intercept:
            X_mean = X.mean(axis=0, dtype=np.float32)
            y_mean = float(y.mean())
            Xc = X - X_mean
            yc = y - y_mean
        else:
            Xc = X
            yc = y

        XTX = gemm(Xc, Xc, trans_A=True)
        XTy = gemm(Xc, yc.reshape(-1, 1), trans_A=True)

        XTX_np = np.array(XTX, dtype=np.float64)
        XTy_np = np.array(XTy, dtype=np.float64).ravel()

        coef = np.linalg.lstsq(XTX_np, XTy_np, rcond=None)[0]

        self._estimator.coef_ = coef.astype(np.float32)
        if fit_intercept:
            self._estimator.intercept_ = y_mean - X_mean @ coef
        else:
            self._estimator.intercept_ = 0.0
        self._fitted = True
        return self

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        return X @ self._estimator.coef_ + self._estimator.intercept_


class MetalRidge(BaseGPUEstimator):
    def fit(self, X, y, **kwargs):
        X, y = self._validate_data(X, y)
        if not self._should_use_gpu(X):
            return self._fallback_fit(X, y, **kwargs)

        n, p = X.shape
        alpha = self._estimator.alpha
        fit_intercept = self._estimator.fit_intercept

        XTX = np.empty((p, p), dtype=np.float32)
        XTy = np.empty(p, dtype=np.float32)
        X_mean = np.empty(p, dtype=np.float32)

        ridge_fit(X, y, XTX, XTy, X_mean)

        np.fill_diagonal(XTX, XTX.diagonal() + alpha)
        coef = np.linalg.solve(XTX, XTy)

        self._estimator.coef_ = coef.astype(np.float32)
        if fit_intercept:
            self._estimator.intercept_ = float(y.mean()) - X_mean @ coef
        else:
            self._estimator.intercept_ = 0.0
        self._fitted = True
        return self


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

        XTX = gemm(Xc, Xc, trans_A=True)
        XTy = gemm(Xc, yc.reshape(-1, 1), trans_A=True).ravel()

        L = np.linalg.norm(XTX, ord=2)
        step = 1.0 / L

        x = np.zeros(p, dtype=np.float32)
        z = np.zeros(p, dtype=np.float32)
        x_temp = np.empty(p, dtype=np.float32)
        x_prev = np.empty(p, dtype=np.float32)
        t = 1.0

        for it in range(max_iter):
            np.copyto(x_prev, x)

            grad = XTX @ z - XTy
            np.copyto(x_temp, z - step * grad)

            soft_threshold(x, x_temp, step * alpha * n)

            t_prev = t
            t = (1.0 + np.sqrt(1.0 + 4.0 * t_prev * t_prev)) / 2.0

            np.copyto(x_temp, x - x_prev)
            factor = (t_prev - 1.0) / t
            np.copyto(z, x)
            z += factor * x_temp

            diff = np.max(np.abs(x - x_prev))
            if diff < tol:
                break

        self._estimator.coef_ = x.astype(np.float32)
        if fit_intercept:
            self._estimator.intercept_ = y_mean - X_mean @ x
        else:
            self._estimator.intercept_ = 0.0
        self._estimator.n_iter_ = it + 1
        self._fitted = True
        return self

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        return X @ self._estimator.coef_ + self._estimator.intercept_


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

        XTX = gemm(Xc, Xc, trans_A=True)
        XTy = gemm(Xc, yc.reshape(-1, 1), trans_A=True).ravel()

        L = np.linalg.norm(XTX, ord=2)
        step = 1.0 / L

        x = np.zeros(p, dtype=np.float32)
        z = np.zeros(p, dtype=np.float32)
        x_temp = np.empty(p, dtype=np.float32)
        x_prev = np.empty(p, dtype=np.float32)
        t = 1.0

        for it in range(max_iter):
            np.copyto(x_prev, x)

            grad = XTX @ z - XTy
            np.copyto(x_temp, z - step * grad)

            soft_threshold(x, x_temp, step * alpha * l1_ratio * n)
            x /= (1.0 + step * alpha * (1.0 - l1_ratio) * n)

            t_prev = t
            t = (1.0 + np.sqrt(1.0 + 4.0 * t_prev * t_prev)) / 2.0

            np.copyto(x_temp, x - x_prev)
            factor = (t_prev - 1.0) / t
            np.copyto(z, x)
            z += factor * x_temp

            diff = np.max(np.abs(x - x_prev))
            if diff < tol:
                break

        self._estimator.coef_ = x.astype(np.float32)
        if fit_intercept:
            self._estimator.intercept_ = y_mean - X_mean @ x
        else:
            self._estimator.intercept_ = 0.0
        self._estimator.n_iter_ = it + 1
        self._fitted = True
        return self

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        return X @ self._estimator.coef_ + self._estimator.intercept_


class MetalLogisticRegression(BaseGPUEstimator):
    def fit(self, X, y, **kwargs):
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
                X, y, C, tol, max_iter, fit_intercept, penalty, pos_label=classes[1]
            )
            self._estimator.coef_ = coef_.reshape(1, -1)
            self._estimator.intercept_ = np.array([intercept_])
        else:
            coefs = []
            intercepts = []
            for c in classes:
                y_bin = np.where(y == c, np.float32(1.0), np.float32(0.0))
                coef_c, intercept_c = self._fit_binary(
                    X, y_bin, C, tol, max_iter, fit_intercept, penalty, pos_label=c
                )
                coefs.append(coef_c)
                intercepts.append(intercept_c)
            self._estimator.coef_ = np.array(coefs)
            self._estimator.intercept_ = np.array(intercepts)

        self._estimator.classes_ = classes
        self._fitted = True
        return self

    def _fit_binary(self, X, y, C, tol, max_iter, fit_intercept, penalty, pos_label):
        n, p = X.shape
        alpha = 1.0 / C if penalty == "l2" else 0.0

        if fit_intercept:
            ones = np.ones(n, dtype=np.float32)
            Xe = np.column_stack([X, ones])
            pe = p + 1
        else:
            Xe = X
            pe = p

        w = np.zeros(pe, dtype=np.float32)

        # Pre-allocate temp buffers (reused across iterations)
        linear = np.empty(n, dtype=np.float32)
        weight = np.empty(n, dtype=np.float32)
        X_scaled = np.empty((n, pe), dtype=np.float32)
        Hessian = np.empty((pe, pe), dtype=np.float32)
        gradient = np.empty(pe, dtype=np.float32)

        for it in range(max_iter):
            logreg_irls_iter(Xe, y, w, 0.0, linear, weight, X_scaled, Hessian, gradient)

            # Add L2 regularization
            if alpha > 0:
                Hessian[np.diag_indices_from(Hessian)] += alpha
                gradient += alpha * w

            grad_norm = np.linalg.norm(gradient)
            if grad_norm < tol * max(1.0, np.linalg.norm(w)):
                break

            delta = np.linalg.solve(Hessian, gradient)
            w -= delta

        if fit_intercept:
            w_final = w[:-1].copy()
            b_final = float(w[-1])
        else:
            w_final = w.copy()
            b_final = 0.0

        return w_final, b_final

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        scores = X @ self._estimator.coef_.T + self._estimator.intercept_
        return self._estimator.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict_proba(X)
        scores = X @ self._estimator.coef_.T + self._estimator.intercept_
        if scores.shape[1] == 1:
            prob = 1.0 / (1.0 + np.exp(-scores.ravel()))
            return np.column_stack([1 - prob, prob])
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)


