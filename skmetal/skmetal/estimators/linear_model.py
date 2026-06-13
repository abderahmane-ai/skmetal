import numpy as np
from ._base import BaseGPUEstimator
from .._bridge import gemm, ridge_fit, logreg_irls_fused, logreg_irls_fused_solve, multinomial_irls_iter, fista_fit


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

        X = X.copy()

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
            self._fit_multinomial(X, y, classes, n, p, n_classes, tol, max_iter, fit_intercept, penalty)

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

        # y already float32 from _validate_data (GPU reads as float32)
        w = np.zeros(pe, dtype=np.float32)

        # Pre-allocate temp buffers (reused across iterations)
        linear = np.empty(n, dtype=np.float32)
        weight = np.empty(n, dtype=np.float32)
        X_scaled = np.empty((n, pe), dtype=np.float32)
        Hessian = np.empty((pe, pe), dtype=np.float32)
        gradient = np.empty(pe, dtype=np.float32)
        delta = np.empty(pe, dtype=np.float32)

        for it in range(max_iter):
            if pe >= 500:
                # Full GPU solve: includes L2 regularization, Cholesky factorization and triangular solve
                logreg_irls_fused_solve(
                    Xe, y, w, 0.0, linear, weight, X_scaled, Hessian, gradient, delta, float(alpha)
                )
            else:
                # Fused GPU calculations, followed by CPU solver (faster for small matrices)
                logreg_irls_fused(Xe, y, w, 0.0, linear, weight, X_scaled, Hessian, gradient)
                if alpha > 0:
                    Hessian[np.diag_indices_from(Hessian)] += alpha
                    gradient += alpha * w
                delta = np.linalg.solve(Hessian, gradient)

            grad_norm = np.linalg.norm(gradient)
            if grad_norm < tol * max(1.0, np.linalg.norm(w)):
                break

            w -= delta

        if fit_intercept:
            w_final = w[:-1].copy()
            b_final = float(w[-1])
        else:
            w_final = w.copy()
            b_final = 0.0

        return w_final, b_final

    def _fit_multinomial(self, X, y, classes, n, p, C, tol, max_iter, fit_intercept, penalty):
        alpha = 1.0 / C if penalty == "l2" else 0.0

        y_enc = np.zeros(n, dtype=np.float32)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for i in range(n):
            y_enc[i] = class_to_idx[y[i]]

        if fit_intercept:
            ones = np.ones((n, 1), dtype=np.float32)
            Xe = np.column_stack([X, ones])
            pe = p + 1
        else:
            Xe = X
            pe = p

        W = np.zeros((pe, C), dtype=np.float32)
        scores = np.empty((n, C), dtype=np.float32)
        prob = np.empty_like(scores)
        max_vals = np.empty(n, dtype=np.float32)
        sum_exp = np.empty(n, dtype=np.float32)
        residual = np.empty_like(scores)
        gradient = np.empty((pe, C), dtype=np.float32)
        hessians = np.empty((C, pe, pe), dtype=np.float32)

        for it in range(max_iter):
            multinomial_irls_iter(Xe, W, y_enc, scores, prob, max_vals, sum_exp, residual, gradient, hessians)

            if alpha > 0:
                alpha_scaled = alpha / n
                for c in range(C):
                    hessians[c, range(pe), range(pe)] += alpha_scaled
                    gradient[:, c] += alpha_scaled * W[:, c]

            grad_norm = np.linalg.norm(gradient)
            if grad_norm < tol * max(1.0, np.linalg.norm(W)):
                break

            for c in range(C):
                H = hessians[c]
                G_c = gradient[:, c].copy()
                delta = np.linalg.solve(H, G_c)
                W[:, c] -= delta

        if fit_intercept:
            self._estimator.coef_ = W[:-1].T.copy()
            self._estimator.intercept_ = W[-1].copy()
        else:
            self._estimator.coef_ = W.T.copy()
            self._estimator.intercept_ = np.zeros(C, dtype=np.float32)

        self._estimator.classes_ = classes

    def predict(self, X):
        X = self._validate_data(X)[0]
        if not self._should_use_gpu(X) or not self._fitted:
            return self._fallback_predict(X)
        scores = X @ self._estimator.coef_.T + self._estimator.intercept_
        if scores.shape[1] == 1:
            return self._estimator.classes_[(scores.ravel() > 0).astype(int)]
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


