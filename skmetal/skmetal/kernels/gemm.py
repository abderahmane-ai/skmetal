"""GEMM kernel dispatch."""

from .._bridge import gemm as _gemm


def gemm(A, B, alpha=1.0, beta=0.0, trans_A=False, trans_B=False):
    """Matrix multiplication: C = alpha * A @ B + beta * C"""
    return _gemm(A, B, alpha=alpha, beta=beta, trans_A=trans_A, trans_B=trans_B)