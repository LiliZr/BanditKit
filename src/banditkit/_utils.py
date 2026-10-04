"""Small numerical helpers used across the package."""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "make_rng",
    "argmax_random",
    "sherman_morrison_",
    "pareto_front",
    "hadamard",
]


def make_rng(seed: Any = None) -> np.random.Generator:
    """Return a ``numpy.random.Generator`` from a seed, a SeedSequence or a Generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def argmax_random(values: np.ndarray, rng: np.random.Generator) -> int:
    """Index of the maximum of ``values``, ties broken uniformly at random."""
    best = values.max()
    if np.isnan(best):
        raise FloatingPointError("NaN encountered while selecting an arm.")
    candidates = np.flatnonzero(values == best)
    if candidates.size == 1:
        return int(candidates[0])
    return int(candidates[rng.integers(candidates.size)])


def sherman_morrison_(inv: np.ndarray, x: np.ndarray) -> None:
    """In place update ``inv <- (inv^{-1} + x x^T)^{-1}``."""
    vx = inv @ x
    denom = 1.0 + float(x @ vx)
    inv -= np.multiply.outer(vx, vx) / denom


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows of ``points`` (maximization).

    Row ``j`` dominates row ``i`` if it is >= on every coordinate and > on at
    least one. Infinite values are allowed (unexplored arms).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points[:, None]
    n = points.shape[0]
    ge = np.all(points[None, :, :] >= points[:, None, :], axis=2)  # ge[i, j]: j >= i
    gt = np.any(points[None, :, :] > points[:, None, :], axis=2)
    dominated = np.any(ge & gt, axis=1)
    return np.flatnonzero(~dominated) if n else np.array([], dtype=int)


def hadamard(n: int) -> np.ndarray:
    """Sylvester Hadamard matrix of order ``n`` (a power of two)."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard order must be a power of two, got {n}.")
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h
