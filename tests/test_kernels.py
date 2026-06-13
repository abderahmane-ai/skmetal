"""Unit tests for individual Metal kernel functions via the Python bridge.

Each test exercises one _bridge wrapper with a tiny, hand-computable input
and compares the GPU result against a numpy reference with tight tolerance.
This catches regressions in both the Metal kernel and the Python glue.
"""

import numpy as np
import pytest
from skmetal import _bridge

pytestmark = [
    pytest.mark.skipif(
        not hasattr(_bridge, "_lib"),
        reason="SkMetalBridge dylib not available",
    ),
]


# ===========================================================================
# Reduction kernels
# ===========================================================================

class TestReductionKernels:
    def test_reduce_sum_positive(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = _bridge.reduce_sum(X)
        assert result == pytest.approx(15.0, abs=1e-5)

    def test_reduce_sum_negative(self):
        X = np.array([-1.0, -2.0, 3.0], dtype=np.float32)
        result = _bridge.reduce_sum(X)
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_reduce_sum_single(self):
        X = np.array([42.0], dtype=np.float32)
        result = _bridge.reduce_sum(X)
        assert result == pytest.approx(42.0, abs=1e-5)

    def test_reduce_sum_uniform(self):
        X = np.full(100, 3.0, dtype=np.float32)
        result = _bridge.reduce_sum(X)
        assert result == pytest.approx(300.0, abs=1e-4)

    def test_reduce_mean_var_known(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        mean, var = _bridge.reduce_mean_var(X)
        assert mean == pytest.approx(3.0, abs=1e-5)
        np.testing.assert_allclose(var, 2.0, rtol=1e-4)

    def test_reduce_mean_var_uniform(self):
        X = np.full(50, 7.0, dtype=np.float32)
        mean, var = _bridge.reduce_mean_var(X)
        assert mean == pytest.approx(7.0, abs=1e-5)
        assert var == pytest.approx(0.0, abs=1e-5)

    def test_reduce_mean_var_single(self):
        X = np.array([3.14], dtype=np.float32)
        mean, var = _bridge.reduce_mean_var(X)
        assert mean == pytest.approx(3.14, abs=1e-5)
        assert var == pytest.approx(0.0, abs=1e-5)

    def test_reduce_mean_var_two_elements(self):
        X = np.array([0.0, 10.0], dtype=np.float32)
        mean, var = _bridge.reduce_mean_var(X)
        assert mean == pytest.approx(5.0, abs=1e-5)
        assert var == pytest.approx(25.0, abs=1e-4)


# ===========================================================================
# Element-wise kernels (sigmoid, subtract, axpy, norm_sq, negate)
# ===========================================================================

class TestElementWiseKernels:
    def test_sigmoid_zero(self):
        x = np.array([0.0], dtype=np.float32)
        out = np.empty(1, dtype=np.float32)
        _bridge.sigmoid(x, out)
        assert out[0] == pytest.approx(0.5, abs=1e-5)

    def test_sigmoid_symmetric(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        out = np.empty(3, dtype=np.float32)
        _bridge.sigmoid(x, out)
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_sigmoid_extreme(self):
        x = np.array([-100.0, 100.0], dtype=np.float32)
        out = np.empty(2, dtype=np.float32)
        _bridge.sigmoid(x, out)
        assert out[0] == pytest.approx(0.0, abs=1e-6)
        assert out[1] == pytest.approx(1.0, abs=1e-6)

    def test_sigmoid_known_values(self):
        x = np.array([0.5, 2.0, -2.0], dtype=np.float32)
        out = np.empty(3, dtype=np.float32)
        _bridge.sigmoid(x, out)
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_subtract_basic(self):
        a = np.array([5.0, 3.0, 1.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = np.empty(3, dtype=np.float32)
        _bridge.subtract(a, b, out)
        np.testing.assert_allclose(out, [4.0, 1.0, -2.0], rtol=1e-5)

    def test_subtract_negative(self):
        a = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        b = np.array([1.0, -1.0, 0.0], dtype=np.float32)
        out = np.empty(3, dtype=np.float32)
        _bridge.subtract(a, b, out)
        np.testing.assert_allclose(out, [-2.0, 1.0, 1.0], rtol=1e-5)

    def test_axpy_alpha_zero(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        _bridge.axpy(a, b, 0.0)
        np.testing.assert_allclose(a, [1.0, 2.0, 3.0], rtol=1e-5)

    def test_axpy_positive(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        _bridge.axpy(a, b, 2.0)
        np.testing.assert_allclose(a, [21.0, 42.0, 63.0], rtol=1e-5)

    def test_axpy_negative_alpha(self):
        a = np.array([10.0, 20.0], dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        _bridge.axpy(a, b, -1.0)
        np.testing.assert_allclose(a, [9.0, 18.0], rtol=1e-5)

    def test_norm_sq_basic(self):
        x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        out = np.empty(3, dtype=np.float32)
        _bridge.norm_sq(x, out)
        np.testing.assert_allclose(out, [4.0, 9.0, 16.0], rtol=1e-5)

    def test_norm_sq_negative(self):
        x = np.array([-2.0, -3.0, 0.0], dtype=np.float32)
        out = np.empty(3, dtype=np.float32)
        _bridge.norm_sq(x, out)
        np.testing.assert_allclose(out, [4.0, 9.0, 0.0], rtol=1e-5)

    def test_negate_simple(self):
        a = np.array([1.0, -2.0, 0.0], dtype=np.float32)
        out = np.empty(3, dtype=np.float32)
        _bridge.negate(a, out)
        np.testing.assert_allclose(out, [-1.0, 2.0, 0.0], rtol=1e-5)

    def test_row_max_basic(self):
        matrix = np.array([[1.0, 5.0, 3.0],
                           [4.0, 2.0, 9.0],
                           [7.0, 8.0, 0.0]], dtype=np.float32)
        max_vals = np.empty(3, dtype=np.float32)
        _bridge.row_max(matrix, max_vals)
        np.testing.assert_allclose(max_vals, [5.0, 9.0, 8.0], rtol=1e-5)

    def test_row_max_single_col(self):
        matrix = np.array([[3.0], [1.0], [4.0]], dtype=np.float32)
        max_vals = np.empty(3, dtype=np.float32)
        _bridge.row_max(matrix, max_vals)
        np.testing.assert_allclose(max_vals, [3.0, 1.0, 4.0], rtol=1e-5)

    def test_row_sum_basic(self):
        matrix = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]], dtype=np.float32)
        sums = np.empty(2, dtype=np.float32)
        _bridge.row_sum(matrix, sums)
        np.testing.assert_allclose(sums, [6.0, 15.0], rtol=1e-5)

    def test_row_sum_negative(self):
        matrix = np.array([[1.0, -2.0, 3.0],
                           [-4.0, 5.0, -6.0]], dtype=np.float32)
        sums = np.empty(2, dtype=np.float32)
        _bridge.row_sum(matrix, sums)
        np.testing.assert_allclose(sums, [2.0, -5.0], rtol=1e-5)

    def test_softmax_exp(self):
        matrix = np.array([[1.0, 2.0, 3.0],
                           [0.0, 0.0, 0.0]], dtype=np.float32)
        max_vals = np.array([3.0, 0.0], dtype=np.float32)
        out = np.empty((2, 3), dtype=np.float32)
        _bridge.softmax_exp(matrix, max_vals, out)
        expected = np.exp(matrix - max_vals.reshape(-1, 1))
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_softmax_normalize_residual(self):
        prob = np.array([[0.2, 0.3, 0.5],
                         [0.7, 0.2, 0.1]], dtype=np.float32)
        row_sums = np.array([1.0, 1.0], dtype=np.float32)
        y = np.array([2.0, 0.0], dtype=np.float32)
        residual = np.empty((2, 3), dtype=np.float32)
        _bridge.softmax_normalize_residual(prob, row_sums, y, residual)
        prob_norm = prob / row_sums.reshape(-1, 1)
        one_hot = np.zeros((2, 3), dtype=np.float32)
        one_hot[np.arange(2), y.astype(int)] = 1.0
        expected = prob_norm - one_hot
        np.testing.assert_allclose(residual, expected, rtol=1e-5)

    def test_transpose_f32(self):
        x = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)
        out = np.empty((3, 2), dtype=np.float32)
        _bridge.transpose_f32(x, out)
        np.testing.assert_allclose(out, x.T, rtol=1e-5)

    def test_transpose_f32_square(self):
        x = np.array([[1.0, 2.0],
                      [3.0, 4.0]], dtype=np.float32)
        out = np.empty((2, 2), dtype=np.float32)
        _bridge.transpose_f32(x, out)
        np.testing.assert_allclose(out, x.T, rtol=1e-5)


# ===========================================================================
# Distance kernels
# ===========================================================================

class TestDistanceKernels:
    def test_pairwise_distance_identity(self):
        X = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0]], dtype=np.float32)
        D = _bridge.pairwise_distance(X)
        assert D.shape == (3, 3)
        np.testing.assert_allclose(np.diag(D), [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(D[0, 1], 2.0, rtol=1e-5)  # (1-0)² + (0-1)²
        np.testing.assert_allclose(D[0, 2], 1.0, rtol=1e-5)  # (1-1)² + (0-1)²

    def test_pairwise_distance_symmetric(self):
        X = np.array([[0.0, 0.0],
                      [3.0, 4.0],
                      [1.0, 2.0]], dtype=np.float32)
        D = _bridge.pairwise_distance(X)
        assert np.allclose(D, D.T, rtol=1e-5), "Distance matrix must be symmetric"

    def test_row_norm_sq_basic(self):
        X = np.array([[3.0, 4.0],
                      [1.0, 2.0],
                      [0.0, 5.0]], dtype=np.float32)
        norms = np.empty(3, dtype=np.float32)
        _bridge.row_norm_sq(X, norms)
        np.testing.assert_allclose(norms, [25.0, 5.0, 25.0], rtol=1e-5)

    def test_distance_correct_expansion(self):
        D = np.array([[0.0, 2.0],
                      [2.0, 0.0]], dtype=np.float32)
        X_norm = np.array([1.0, 4.0], dtype=np.float32)
        C_norm = np.array([4.0, 1.0], dtype=np.float32)
        _bridge.distance_correct(D, X_norm, C_norm)
        expected = X_norm[:, None] + C_norm[None, :] - 2.0 * np.array([[0.0, 2.0], [2.0, 0.0]])
        np.testing.assert_allclose(D, expected, rtol=1e-5)


# ===========================================================================
# IRLS kernels
# ===========================================================================

class TestIrlsKernels:
    def test_irls_weight_perfect(self):
        p = np.array([0.0, 1.0, 0.5], dtype=np.float32)
        w = np.empty(3, dtype=np.float32)
        _bridge.irls_weight(p, w)
        clamped = np.clip(p, 1e-7, 1 - 1e-7)
        expected = np.sqrt(clamped * (1 - clamped))
        np.testing.assert_allclose(w, expected, rtol=1e-5)

    def test_irls_weight_uniform(self):
        p = np.full(5, 0.5, dtype=np.float32)
        w = np.empty(5, dtype=np.float32)
        _bridge.irls_weight(p, w)
        np.testing.assert_allclose(w, [0.5] * 5, rtol=1e-5)

    def test_scale_rows_basic(self):
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=np.float32)
        weights = np.array([2.0, 0.5, 1.0], dtype=np.float32)
        out = np.empty_like(X)
        _bridge.scale_rows(X, weights, out)
        expected = X * weights[:, None]
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_scale_rows_zero_weight(self):
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0]], dtype=np.float32)
        weights = np.array([0.0, 0.0], dtype=np.float32)
        out = np.empty_like(X)
        _bridge.scale_rows(X, weights, out)
        np.testing.assert_allclose(out, [[0.0, 0.0], [0.0, 0.0]], rtol=1e-5)


# ===========================================================================
# Centering / scaling kernels
# ===========================================================================

class TestCenterKernels:
    def test_center_columns(self):
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=np.float32)
        mean = np.array([3.0, 4.0], dtype=np.float32)
        _bridge.center_columns(X, mean)
        np.testing.assert_allclose(X, [[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]], rtol=1e-5)
        assert np.allclose(X.sum(axis=0), [0.0, 0.0], atol=1e-5)

    def test_scaler_fit_known(self):
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=np.float32)
        mean_out = np.empty(2, dtype=np.float32)
        var_out = np.empty(2, dtype=np.float32)
        _bridge.scaler_fit(X, mean_out, var_out)
        np.testing.assert_allclose(mean_out, [3.0, 4.0], rtol=1e-5)
        # Population variance (ddof=0): [(1-3)² + (3-3)² + (5-3)²] / 3 = 8/3
        np.testing.assert_allclose(var_out, [8.0 / 3.0, 8.0 / 3.0], rtol=1e-4)

    def test_scaler_fit_uniform(self):
        X = np.full((10, 3), 5.0, dtype=np.float32)
        mean_out = np.empty(3, dtype=np.float32)
        var_out = np.empty(3, dtype=np.float32)
        _bridge.scaler_fit(X, mean_out, var_out)
        np.testing.assert_allclose(mean_out, [5.0, 5.0, 5.0], rtol=1e-5)
        np.testing.assert_allclose(var_out, [0.0, 0.0, 0.0], atol=1e-5)

    def test_column_minmax_basic(self):
        X = np.array([[3.0, 8.0],
                      [1.0, 6.0],
                      [4.0, 9.0],
                      [0.0, 7.0]], dtype=np.float32)
        min_out = np.empty(2, dtype=np.float32)
        max_out = np.empty(2, dtype=np.float32)
        _bridge.column_minmax(X, min_out, max_out)
        np.testing.assert_allclose(min_out, [0.0, 6.0], rtol=1e-5)
        np.testing.assert_allclose(max_out, [4.0, 9.0], rtol=1e-5)

    def test_column_minmax_single(self):
        X = np.array([[5.0]], dtype=np.float32)
        min_out = np.empty(1, dtype=np.float32)
        max_out = np.empty(1, dtype=np.float32)
        _bridge.column_minmax(X, min_out, max_out)
        assert min_out[0] == pytest.approx(5.0, abs=1e-5)
        assert max_out[0] == pytest.approx(5.0, abs=1e-5)

    def test_column_transform(self):
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=np.float32)
        center = np.array([3.0, 4.0], dtype=np.float32)
        scale = np.array([2.0, 0.5], dtype=np.float32)
        out = np.empty_like(X)
        _bridge.column_transform(X, out, center, scale)
        expected = (X - center) * scale
        np.testing.assert_allclose(out, expected, rtol=1e-5)


# ===========================================================================
# Argmin / KMeans kernels
# ===========================================================================

class TestKMeansKernels:
    def test_argmin_rows_basic(self):
        matrix = np.array([[5.0, 1.0, 3.0],
                           [2.0, 4.0, 0.0],
                           [9.0, 7.0, 8.0]], dtype=np.float32)
        indices = np.empty(3, dtype=np.int32)
        _bridge.argmin_rows(matrix, indices)
        np.testing.assert_array_equal(indices, [1, 2, 1])

    def test_argmin_rows_tie_first(self):
        matrix = np.array([[1.0, 1.0, 3.0],
                           [4.0, 2.0, 2.0]], dtype=np.float32)
        indices = np.empty(2, dtype=np.int32)
        _bridge.argmin_rows(matrix, indices)
        # On tie, first column wins
        assert indices[0] == 0

    def test_kmeans_assign(self):
        X = np.array([[0.0, 0.0],
                      [1.0, 1.0],
                      [10.0, 10.0],
                      [9.0, 9.0]], dtype=np.float32)
        centroids = np.array([[0.0, 0.0],
                              [10.0, 10.0]], dtype=np.float32)
        assignments = np.empty(4, dtype=np.int32)
        _bridge.kmeans_assign(X, centroids, assignments, 4, 2, 2)
        np.testing.assert_array_equal(assignments, [0, 0, 1, 1])

    def test_sv_init(self):
        parent = np.empty(5, dtype=np.int32)
        _bridge.sv_init(parent)
        np.testing.assert_array_equal(parent, [0, 1, 2, 3, 4])

    def test_sv_shortcut(self):
        parent = np.array([0, 2, 2, 1, 3], dtype=np.int32)
        _bridge.sv_shortcut(parent)
        # Parallel: parent[i] = parent[parent[i]] (for i != parent[i])
        # i=0: parent[0]=parent[parent[0]]=parent[0]=0 (skip)
        # i=1: parent[1]=parent[parent[1]]=parent[2]=2
        # i=2: parent[2]=parent[parent[2]]=parent[2]=2 (skip)
        # i=3: parent[3]=parent[parent[3]]=parent[1]=2
        # i=4: parent[4]=parent[parent[4]]=parent[3]=1
        np.testing.assert_array_equal(parent, [0, 2, 2, 2, 1])


# ===========================================================================
# Lasso / FISTA kernels
# ===========================================================================

class TestExtraKernels:
    def test_soft_threshold_below(self):
        w = np.empty(3, dtype=np.float32)
        w_temp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        _bridge.soft_threshold(w, w_temp, 5.0)
        np.testing.assert_allclose(w, [0.0, 0.0, 0.0], atol=1e-6)

    def test_soft_threshold_above(self):
        w = np.empty(4, dtype=np.float32)
        w_temp = np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float32)
        _bridge.soft_threshold(w, w_temp, 0.5)
        np.testing.assert_allclose(w, [0.5, 0.0, 0.0, -0.5], rtol=1e-5)

    def test_soft_threshold_zero(self):
        w = np.empty(2, dtype=np.float32)
        w_temp = np.array([0.0, 0.0], dtype=np.float32)
        _bridge.soft_threshold(w, w_temp, 0.1)
        np.testing.assert_allclose(w, [0.0, 0.0], atol=1e-6)


# ===========================================================================
# GEMM kernel
# ===========================================================================

class TestGEMMKernels:
    def test_gemm_identity(self):
        A = np.eye(3, dtype=np.float32)
        B = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], dtype=np.float32)
        C = _bridge.gemm(A, B)
        np.testing.assert_allclose(C, B, rtol=1e-5)

    def test_gemm_matrix_vector(self):
        A = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=np.float32)
        B = np.array([[1.0], [0.0]], dtype=np.float32)
        C = _bridge.gemm(A, B)
        np.testing.assert_allclose(C, [[1.0], [3.0], [5.0]], rtol=1e-5)

    def test_gemm_transpose(self):
        # A @ B with A transposed: A is (2,3), A^T is (3,2), B is (2,2)
        # A^T @ B = [[1,4],[2,5],[3,6]] @ I = [[1,4],[2,5],[3,6]]
        A = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)
        B = np.eye(2, dtype=np.float32)
        C = _bridge.gemm(A, B, trans_A=True)
        expected = A.T @ B
        np.testing.assert_allclose(C, expected, rtol=1e-5)

    def test_gemm_alpha(self):
        A = np.ones((2, 2), dtype=np.float32)
        B = np.ones((2, 2), dtype=np.float32)
        C = _bridge.gemm(A, B, alpha=2.0)
        # alpha * (A @ B) = 2 * [[2,2],[2,2]] = [[4,4],[4,4]]
        np.testing.assert_allclose(C, [[4.0, 4.0], [4.0, 4.0]], rtol=1e-5)

    def test_gemm_single_element(self):
        A = np.array([[3.0]], dtype=np.float32)
        B = np.array([[4.0]], dtype=np.float32)
        C = _bridge.gemm(A, B)
        np.testing.assert_allclose(C, [[12.0]], rtol=1e-5)
