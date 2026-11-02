"""Building blocks of linear policies: ridge model, action buckets, random projections."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .._utils import argmax_random, hadamard, sherman_morrison_

__all__ = ["RidgeModel", "ActionBuckets", "make_projection", "weighted_norms"]


def weighted_norms(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """``||x||_A = sqrt(x^T A x)`` for every row x of X."""
    q = np.einsum("ij,ij->i", X @ A, X)
    return np.sqrt(np.maximum(q, 0.0, out=q), out=q)


class RidgeModel:
    """Online ridge regression, the estimator of LinUCB / OFUL.

    Attributes
    ----------
    V_inv : (dim, dim) inverse of the design matrix ``V = lambda I + sum x x^T``.
    V : (dim, dim) design matrix itself, kept only if ``keep_V`` (needed to merge or split clusters).
    b : (dim,) ``sum reward * x``.
    theta : (dim,) ridge estimate ``V^-1 b``.
    n_obs : number of observations.
    buckets : optional :class:`ActionBuckets` attached to this model.
    buckets_on : the action features the buckets were built on (rebuilt when the action set changes).
    """

    __slots__ = ("V_inv", "V", "b", "theta", "n_obs", "buckets", "buckets_on", "steps_since_rebucket")

    def __init__(self, dim: int, lambda_: float, dtype=np.float64, keep_V: bool = False) -> None:
        self.V_inv = np.eye(dim, dtype=dtype) / lambda_
        self.V = np.eye(dim, dtype=dtype) * lambda_ if keep_V else None
        self.b = np.zeros(dim, dtype=dtype)
        self.theta = np.zeros(dim, dtype=dtype)
        self.n_obs = 0
        self.buckets: Optional[ActionBuckets] = None
        self.buckets_on: Optional[np.ndarray] = None
        self.steps_since_rebucket = 0

    def update(self, x: np.ndarray, reward: float) -> None:
        """Add one observation (features x, scalar reward). O(dim^2) with Sherman-Morrison."""
        sherman_morrison_(self.V_inv, x)
        if self.V is not None:
            self.V += np.multiply.outer(x, x)
        self.b += reward * x
        np.dot(self.V_inv, self.b, out=self.theta)
        self.n_obs += 1
        self.steps_since_rebucket += 1

    def refresh_from_V(self) -> None:
        """Recompute ``V_inv`` and ``theta`` after ``V`` and ``b`` changed in bulk (cluster merge or split)."""
        self.V_inv = np.linalg.inv(self.V).astype(self.b.dtype, copy=False)
        np.dot(self.V_inv, self.b, out=self.theta)


class ActionBuckets:
    """Arm bucketing ("AB"): two-step selection over buckets of actions.

    Actions are sorted by a score (estimated mean, or UCB) and cut into
    ``n_buckets`` buckets of equal size; the champion of a bucket is its best
    scored action. Selection: (1) the UCB of the champions picks a bucket,
    (2) the action is the UCB argmax over that bucket plus all champions.
    Cost per step: O((G + K / G) dim^2) instead of O(K dim^2) for G buckets.

    Parameters
    ----------
    n_actions : int
        Number of actions K.
    n_buckets : int
        Number of buckets G (capped at K).
    refresh_champions : bool
        After each selection, make the best UCB action of the chosen bucket its champion.
    """

    __slots__ = ("n_buckets", "groups", "champions", "last_bucket", "refresh_champions", "last_inputs")

    def __init__(self, n_actions: int, n_buckets: int, refresh_champions: bool = False) -> None:
        self.n_buckets = max(1, min(n_buckets, n_actions))
        self.groups = [np.arange(n_actions)]           # actions of each bucket
        self.champions = np.zeros(1, dtype=np.int64)   # champion of each bucket
        self.last_bucket = -1
        self.refresh_champions = refresh_champions
        # Inputs of the last selection (X, theta, V_inv, width_scale, champions), read by bk.trace.
        self.last_inputs = None

    def rebuild(self, scores: np.ndarray, rng: np.random.Generator) -> None:
        """Sort actions by ``scores`` (descending) and cut them into equal buckets."""
        n_actions, n_buckets = scores.shape[0], self.n_buckets
        # Shuffle before a stable sort so that ties do not depend on action indices.
        perm = rng.permutation(n_actions)
        order = perm[np.argsort(-scores[perm], kind="stable")]
        bounds = [g * (n_actions // n_buckets) for g in range(n_buckets)] + [n_actions]
        self.groups = [np.sort(order[bounds[g]:bounds[g + 1]]) for g in range(n_buckets)]
        self.champions = np.array([order[bounds[g]] for g in range(n_buckets)], dtype=np.int64)

    def select(self, X: np.ndarray, theta: np.ndarray, V_inv: np.ndarray, width_scale: float,
               rng: np.random.Generator) -> int:
        """UCB argmax ``x . theta + width_scale * ||x||_{V^-1}`` restricted as described above."""
        def ucb(idx):
            Xi = X[idx]
            return Xi @ theta + width_scale * weighted_norms(Xi, V_inv)

        self.last_inputs = (X, theta, V_inv, width_scale, self.champions.copy())
        bucket = argmax_random(ucb(self.champions), rng) if self.champions.size > 1 else 0
        candidates = np.union1d(self.groups[bucket], self.champions)
        scores = ucb(candidates)
        action = int(candidates[argmax_random(scores, rng)])
        if self.refresh_champions:
            in_bucket = np.isin(candidates, self.groups[bucket], assume_unique=True)
            self.champions[bucket] = int(candidates[np.argmax(np.where(in_bucket, scores, -np.inf))])
        self.last_bucket = bucket
        return action


def make_projection(n_features: int, n_components: int, kind: str, rng: np.random.Generator,
                    n_points: int, eps: float = 0.1, fjlt_p: int = 2) -> np.ndarray:
    """Random projection matrix of shape (n_components, n_features).

    Parameters
    ----------
    kind : {"gaussian", "fjlt"}
        ``"gaussian"``: i.i.d. N(0, 1/n_components) entries (as scikit-learn's
        GaussianRandomProjection). ``"fjlt"``: Fast Johnson-Lindenstrauss transform
        (Ailon and Chazelle, 2009), ``k^-1/2 P H D`` with D random signs, H the
        normalized Hadamard matrix (features padded to a power of two) and P sparse
        with N(0, 1/q) entries kept with probability ``q = min(1, eps^(p-2) ln(n)^p / d)``.
    n_points : int
        Number of points n to embed (the actions); used by the FJLT sparsity.
    eps : float
        Distortion parameter of the FJLT.
    fjlt_p : int
        Norm (1 or 2) preserved by the FJLT.
    """
    if kind == "gaussian":
        return rng.normal(0.0, 1.0 / math.sqrt(n_components), size=(n_components, n_features))
    if kind == "fjlt":
        d_pad = 1 << max(0, math.ceil(math.log2(n_features)))
        q = min(1.0, eps ** (fjlt_p - 2) * math.log(max(n_points, 2)) ** fjlt_p / d_pad)
        keep = rng.random((n_components, d_pad)) <= q
        P = np.zeros((n_components, d_pad))
        P[keep] = rng.normal(0.0, math.sqrt(1.0 / q), size=int(keep.sum()))
        H = hadamard(d_pad)[:, :n_features] / math.sqrt(d_pad)
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_features)
        return (P @ (H * signs)) / math.sqrt(n_components)
    raise ValueError(f"Unknown projection {kind!r}; use 'gaussian' or 'fjlt'.")
