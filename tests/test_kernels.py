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
        np.testing.assert_allclose(D[0, 1], 2.0, rtol=1e-5)
        np.testing.assert_allclose(D[0, 2], 1.0, rtol=1e-5)

    def test_pairwise_distance_symmetric(self):
        X = np.array([[0.0, 0.0],
                      [3.0, 4.0],
                      [1.0, 2.0]], dtype=np.float32)
        D = _bridge.pairwise_distance(X)
        assert np.allclose(D, D.T, rtol=1e-5), "Distance matrix must be symmetric"


# ===========================================================================
# Centering / scaling kernels
# ===========================================================================

class TestCenterKernels:
    def test_scaler_fit_known(self):
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=np.float32)
        mean_out = np.empty(2, dtype=np.float32)
        var_out = np.empty(2, dtype=np.float32)
        _bridge.scaler_fit(X, mean_out, var_out)
        np.testing.assert_allclose(mean_out, [3.0, 4.0], rtol=1e-5)
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
# KMeans kernels
# ===========================================================================

class TestKMeansKernels:
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
        np.testing.assert_array_equal(parent, [0, 2, 2, 2, 1])


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
        np.testing.assert_allclose(C, [[4.0, 4.0], [4.0, 4.0]], rtol=1e-5)

    def test_gemm_single_element(self):
        A = np.array([[3.0]], dtype=np.float32)
        B = np.array([[4.0]], dtype=np.float32)
        C = _bridge.gemm(A, B)
        np.testing.assert_allclose(C, [[12.0]], rtol=1e-5)
