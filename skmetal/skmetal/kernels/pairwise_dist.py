"""Pairwise distance kernel dispatch."""

from .._bridge import pairwise_distance as _pairwise_distance


def pairwise_distance(X):
    """Squared Euclidean pairwise distance."""
    return _pairwise_distance(X)