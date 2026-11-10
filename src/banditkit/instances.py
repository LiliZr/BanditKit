"""Several synthetic instances of one kind of problem, to average results over problems.

One instance gives the regret on that problem only (frequentist view). To compare
algorithms on a kind of problem, draw several instances from one distribution and
average over them (Bayesian regret). For the average to mean something, every
instance must come from the same distribution: same number of actions, dimension,
noise level and parameter norm, so that regrets share one scale. ``min_gap`` can
exclude degenerate instances whose best action is almost tied with the second.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ._validation import check_choice, check_float, check_int
from .envs import BernoulliMAB, GaussianMAB, LinearEnv, MultiUserLinearEnv
from .envs.base import Environment

__all__ = ["sample_instances", "KINDS"]

#: Kinds of instances: classical MAB with Bernoulli or Gaussian rewards, linear bandit, clustered users.
KINDS = ("bernoulli", "gaussian", "linear", "multi_user")

_MAX_TRIES = 1000


def _unit_rows(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """n points drawn uniformly on the unit sphere of R^d."""
    X = rng.standard_normal((n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _gap(means: np.ndarray) -> float:
    """Gap between the best and the second best mean."""
    top = np.sort(means)[-2:]
    return float(top[1] - top[0])


def sample_instances(kind: str, n_instances: int, n_actions: int, n_features: Optional[int] = None,
                     noise_std: float = 0.1, theta_norm: float = 1.0, mean_range=(0.05, 0.95),
                     min_gap: Optional[float] = None, n_users: int = 100, n_clusters: int = 5,
                     n_candidates: Optional[int] = None, seed: Optional[int] = 0) -> List[Environment]:
    """Draw ``n_instances`` problems from one distribution.

    Parameters
    ----------
    kind : {"bernoulli", "gaussian", "linear", "multi_user"}
        ``"bernoulli"``: means uniform in ``mean_range``. ``"gaussian"``: same means,
        Gaussian noise of std ``noise_std``. ``"linear"``: a new action set (uniform on
        the unit sphere) and a new theta (uniform direction, norm ``theta_norm``) per
        instance, Gaussian noise of std ``noise_std``. ``"multi_user"``: the same
        action set, plus ``n_clusters`` thetas shared by the ``n_users`` users
        (user u in cluster u mod n_clusters), as in the synthetic experiments of
        CLUB and SCLUB.
    n_instances : int
        Number of problems.
    n_actions : int
        Number of actions K of every problem.
    n_features : int, optional
        Dimension d (linear problems only).
    noise_std : float
        Noise standard deviation (Gaussian and linear problems).
    theta_norm : float
        Norm of theta (linear problems).
    mean_range : (low, high)
        Interval of the arm means (Bernoulli and Gaussian problems).
    min_gap : float, optional
        Redraw an instance while the gap between its best and second best mean
        reward is below this value (not for ``"multi_user"``).
    n_users, n_clusters : int
        Users and clusters of users (``"multi_user"`` only).
    n_candidates : int, optional
        Actions offered at each step, drawn at random (``"multi_user"`` only; None: all).
    seed : int, optional
        Seed of the draws (the same seed gives the same instances).

    Returns
    -------
    list of Environment, to pass to :class:`~banditkit.Experiment` as ``env``.
    """
    kind = check_choice("kind", kind, KINDS)
    n_instances = check_int("n_instances", n_instances, low=1)
    n_actions = check_int("n_actions", n_actions, low=2)
    noise_std = check_float("noise_std", noise_std, low=0.0)
    theta_norm = check_float("theta_norm", theta_norm, low=0.0, low_inclusive=False)
    min_gap = check_float("min_gap", min_gap, low=0.0, allow_none=True)
    low, high = (check_float("mean_range[0]", mean_range[0]), check_float("mean_range[1]", mean_range[1]))
    if not low < high:
        raise ValueError(f"`mean_range` must satisfy low < high, got {tuple(mean_range)}.")
    if kind == "bernoulli" and not 0.0 <= low < high <= 1.0:
        raise ValueError("Bernoulli means must lie in [0, 1]: set `mean_range` inside [0, 1].")
    if kind in ("linear", "multi_user"):
        n_features = check_int("n_features", n_features, low=1)
    elif n_features is not None:
        raise ValueError(f"`n_features` only applies to kind='linear' or 'multi_user', not {kind!r}.")
    if kind == "multi_user":
        n_users = check_int("n_users", n_users, low=1)
        n_clusters = check_int("n_clusters", n_clusters, low=1, high=n_users)
        if min_gap is not None:
            raise ValueError("`min_gap` does not apply to kind='multi_user'.")

    rng = np.random.default_rng(seed)
    instances: List[Environment] = []
    for i in range(n_instances):
        if kind == "multi_user":
            X = _unit_rows(rng, n_actions, n_features)
            centers = theta_norm * _unit_rows(rng, n_clusters, n_features)
            thetas = centers[np.arange(n_users) % n_clusters]
            instances.append(MultiUserLinearEnv(X, thetas, noise_std=noise_std, n_candidates=n_candidates))
            continue
        for _ in range(_MAX_TRIES):
            if kind == "linear":
                X = _unit_rows(rng, n_actions, n_features)
                theta = theta_norm * _unit_rows(rng, 1, n_features)[0]
                means = X @ theta
            else:
                means = rng.uniform(low, high, size=n_actions)
            if min_gap is None or _gap(means) >= min_gap:
                break
        else:
            raise ValueError(f"No instance with a gap >= {min_gap} after {_MAX_TRIES} draws; lower `min_gap`.")
        if kind == "bernoulli":
            instances.append(BernoulliMAB(means))
        elif kind == "gaussian":
            instances.append(GaussianMAB(means, noise_std=noise_std))
        else:
            instances.append(LinearEnv(X, theta, noise_std=noise_std))
    return instances
