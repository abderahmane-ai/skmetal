"""Reduction kernel dispatch."""

from .._bridge import reduce_sum as _reduce_sum, reduce_mean_var as _reduce_mean_var


def reduce_sum(X):
    """Sum reduction."""
    return _reduce_sum(X)


def reduce_mean_var(X):
    """Mean and variance reduction (Welford's algorithm)."""
    return _reduce_mean_var(X)