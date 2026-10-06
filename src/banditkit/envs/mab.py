"""Classical and multi-objective multi-armed bandit environments."""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from .._validation import check_array, check_choice, check_float
from ..spec import EnvSpec, Observation
from .base import Environment

__all__ = ["GaussianMAB", "BernoulliMAB", "EmpiricalMAB", "GroupRecommendationMAB"]


def _as_means(means: Any) -> np.ndarray:
    """Means as an array of shape (n_actions, reward_dim)."""
    mu = check_array("means", means, ndim=(1, 2))
    return mu[:, None] if mu.ndim == 1 else mu


_erf = np.vectorize(math.erf, otypes=[float])


def _clipped_gaussian_mean(mu: np.ndarray, sd: np.ndarray, low: float, high: float) -> np.ndarray:
    """E[clip(X, low, high)] for X ~ N(mu, sd^2), elementwise (sd may be 0)."""
    out = np.clip(mu, low, high).astype(np.float64)
    pos = sd > 0
    if np.any(pos):
        m, s = mu[pos], sd[pos]
        a, b = (low - m) / s, (high - m) / s
        cdf_a, cdf_b = 0.5 * (1 + _erf(a / math.sqrt(2))), 0.5 * (1 + _erf(b / math.sqrt(2)))
        pdf_a, pdf_b = np.exp(-a * a / 2) / math.sqrt(2 * math.pi), np.exp(-b * b / 2) / math.sqrt(2 * math.pi)
        out[pos] = low * cdf_a + high * (1 - cdf_b) + m * (cdf_b - cdf_a) + s * (pdf_a - pdf_b)
    return out


class _StaticMAB(Environment):
    """MAB whose observation carries no information and whose means never change."""

    _means: np.ndarray

    def _observation(self, t: int) -> Observation:
        return Observation(t=t)

    def mean_rewards(self, obs: Observation) -> np.ndarray:
        return self._means

    def mean_reward(self, action: int, obs: Observation) -> np.ndarray:
        return self._means[action]


class GaussianMAB(_StaticMAB):
    """Actions with Gaussian reward vectors.

    Parameters
    ----------
    means : array of shape (n_actions,) or (n_actions, reward_dim)
        Mean rewards (one column per objective).
    noise_std : float or array of shape (n_actions,) or (n_actions, reward_dim)
        Noise standard deviation. Ignored when ``noise_cov`` is given.
    noise_cov : array of shape (reward_dim, reward_dim) or (n_actions, reward_dim, reward_dim), optional
        Noise covariance between objectives (symmetric positive semi-definite).
    noisy_dims : sequence of bool of length reward_dim, optional
        Objectives that receive noise; the others return exactly their mean.
    clip : (low, high), optional
        Rewards are clipped to this interval. Mean rewards account for the clipping.
    """

    def __init__(self, means: Any, noise_std: Any = 1.0, noise_cov: Any = None,
                 noisy_dims: Optional[Sequence[bool]] = None, clip: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        raw_means = _as_means(means)
        n_actions, reward_dim = raw_means.shape
        self._raw_means = raw_means

        # Noise: z ~ N(0, I) is turned into noise = factor @ z (covariance) or std * z.
        self._noise_factor = None
        if noise_cov is not None:
            cov = check_array("noise_cov", noise_cov, ndim=(2, 3))
            if cov.ndim == 2:
                cov = np.broadcast_to(cov, (n_actions, reward_dim, reward_dim))
            if cov.shape != (n_actions, reward_dim, reward_dim):
                raise ValueError(f"`noise_cov` must have shape (D, D) or (K, D, D) with K={n_actions}, "
                                 f"D={reward_dim}; got {cov.shape}.")
            if not np.allclose(cov, np.swapaxes(cov, 1, 2)):
                raise ValueError("`noise_cov` must be symmetric.")
            eigval, eigvec = np.linalg.eigh(cov)
            if np.any(eigval < -1e-10 * max(1.0, float(np.abs(eigval).max()))):
                raise ValueError("`noise_cov` must be positive semi-definite.")
            self._noise_factor = eigvec * np.sqrt(np.clip(eigval, 0.0, None))[:, None, :]
            noise_std = np.sqrt(np.clip(np.diagonal(cov, axis1=1, axis2=2), 0.0, None))
        std = check_array("noise_std", noise_std, ndim=(0, 1, 2), nonneg=True)
        if std.ndim == 1:
            std = std[:, None]
        try:
            self._noise_std = np.broadcast_to(std, (n_actions, reward_dim)).astype(np.float64)
        except ValueError:
            raise ValueError(f"`noise_std` must be a scalar or have shape (K,) or (K, D) with K={n_actions}, "
                             f"D={reward_dim}; got {np.shape(noise_std)}.") from None

        self._noisy_dims = None
        if noisy_dims is not None:
            self._noisy_dims = check_array("noisy_dims", noisy_dims, shape=(reward_dim,), dtype=bool, finite=False)

        self._clip = None
        if clip is not None:
            if len(clip) != 2:
                raise ValueError("`clip` must be a pair (low, high).")
            low, high = check_float("clip[0]", clip[0]), check_float("clip[1]", clip[1])
            if not low < high:
                raise ValueError(f"`clip` must satisfy low < high, got {tuple(clip)}.")
            self._clip = (low, high)

        effective_std = self._noise_std if self._noisy_dims is None else self._noise_std * self._noisy_dims
        self._means = raw_means if self._clip is None else _clipped_gaussian_mean(raw_means, effective_std, *self._clip)
        # Largest noise level over the arms, per objective (clipping only reduces it).
        self._spec = EnvSpec(n_actions=n_actions, reward_dim=reward_dim, noise_std=effective_std.max(axis=0))

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    def _sample_reward(self, action: int, obs: Observation) -> np.ndarray:
        z = self.np_random.standard_normal(self._raw_means.shape[1])
        noise = self._noise_factor[action] @ z if self._noise_factor is not None else self._noise_std[action] * z
        if self._noisy_dims is not None:
            noise = noise * self._noisy_dims
        reward = self._raw_means[action] + noise
        if self._clip is not None:
            np.clip(reward, *self._clip, out=reward)
        return reward


class BernoulliMAB(_StaticMAB):
    """Actions whose objectives are independent Bernoulli variables.

    Parameters
    ----------
    means : array of shape (n_actions,) or (n_actions, reward_dim)
        Success probabilities, in [0, 1].
    """

    def __init__(self, means: Any) -> None:
        super().__init__()
        self._means = _as_means(means)
        check_array("means", self._means, low=0.0, high=1.0)
        # Rewards in {0, 1}: 1/2-sub-Gaussian (Hoeffding).
        self._spec = EnvSpec(n_actions=self._means.shape[0], reward_dim=self._means.shape[1], reward_range=1.0,
                             noise_std=0.5)

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    def _sample_reward(self, action: int, obs: Observation) -> np.ndarray:
        return (self.np_random.random(self._means.shape[1]) < self._means[action]).astype(np.float64)


class EmpiricalMAB(_StaticMAB):
    """Actions that return reward vectors resampled from observed data.

    Each action (a genre, a category...) holds the rewards observed for it; a
    step returns one of them uniformly at random. The mean reward of an action
    is the mean of its samples.

    Parameters
    ----------
    samples : sequence of n_actions arrays
        Array i has shape (n_i,) or (n_i, reward_dim), with n_i >= 1.
    action_names : sequence of str, optional
        Name of each action, shown by the figures (a zone, a genre, a game...).
    reward_range : float, optional
        Known length b of the reward interval, e.g. 1 for rewards scaled to [0, 1].
        Default: max - min of the samples.
    """

    def __init__(self, samples: Sequence[Any], action_names: Optional[Sequence[str]] = None,
                 reward_range: Optional[float] = None) -> None:
        super().__init__()
        if isinstance(samples, np.ndarray) and samples.dtype != object:
            samples = list(samples)
        if not isinstance(samples, (list, tuple)) or not samples:
            raise TypeError("`samples` must be a non-empty list with one array of rewards per action.")
        arrays = []
        for i, s in enumerate(samples):
            a = check_array(f"samples[{i}]", s, ndim=(1, 2))
            a = a[:, None] if a.ndim == 1 else a
            if arrays and a.shape[1] != arrays[0].shape[1]:
                raise ValueError(f"All actions must share the reward dimension: samples[0] has "
                                 f"{arrays[0].shape[1]}, samples[{i}] has {a.shape[1]}.")
            arrays.append(a)
        self._samples = arrays
        if action_names is not None and len(action_names) != len(arrays):
            raise ValueError(f"`action_names` has {len(action_names)} names for {len(arrays)} actions.")
        self.action_names = None if action_names is None else [str(n) for n in action_names]
        self._means = np.stack([a.mean(axis=0) for a in arrays])
        low, high = min(float(a.min()) for a in arrays), max(float(a.max()) for a in arrays)
        # Bounded rewards: (high - low) / 2 sub-Gaussian per objective (Hoeffding).
        spans = np.max([a.max(axis=0) for a in arrays], axis=0) - np.min([a.min(axis=0) for a in arrays], axis=0)
        if reward_range is not None:
            reward_range = check_float("reward_range", reward_range, low=0.0, low_inclusive=False)
            if high - low > reward_range * (1 + 1e-12):
                raise ValueError(f"The samples span {high - low:.4g}, more than `reward_range`={reward_range}.")
        else:
            reward_range = high - low if high > low else None
        self._spec = EnvSpec(n_actions=len(arrays), reward_dim=arrays[0].shape[1], reward_range=reward_range,
                             noise_std=spans / 2.0)

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    def _sample_reward(self, action: int, obs: Observation) -> np.ndarray:
        s = self._samples[action]
        return s[self.np_random.integers(s.shape[0])].copy()


class GroupRecommendationMAB(_StaticMAB):
    """Recommend a category (action) to a group of users, as in the MoMAB experiments.

    Playing category c draws a group member u uniformly at random:

    * ``"relevance"`` objective: ``relevance[u, c]``;
    * ``"agreement"`` objective: a second member v != u is drawn and the reward is
      ``1 - |relevance[u, c] - relevance[v, c]|`` (one minus the pairwise disagreement).

    Mean rewards are the exact expectations: the mean relevance over members and
    the mean agreement over pairs of distinct members.

    Parameters
    ----------
    relevance : array of shape (n_members, n_actions)
        Relevance of each category for each member, in [0, 1].
    objectives : sequence of {"relevance", "agreement"}
        Objectives returned, in this order.
    """

    def __init__(self, relevance: Any, objectives: Sequence[str] = ("relevance", "agreement")) -> None:
        super().__init__()
        R = check_array("relevance", relevance, ndim=2, low=0.0, high=1.0)
        objectives = tuple(objectives)
        if not objectives:
            raise ValueError("`objectives` must not be empty.")
        for o in objectives:
            check_choice("objectives", o, ("relevance", "agreement"))
        if "agreement" in objectives and R.shape[0] < 2:
            raise ValueError("The agreement objective needs at least 2 group members.")
        self._relevance, self._objectives = R, objectives
        means = []
        for o in objectives:
            means.append(R.mean(axis=0) if o == "relevance" else 1.0 - self._mean_pairwise_gap(R))
        self._means = np.stack(means, axis=1)
        self._spec = EnvSpec(n_actions=R.shape[1], reward_dim=len(objectives), reward_range=1.0, noise_std=0.5)

    @staticmethod
    def _mean_pairwise_gap(R: np.ndarray) -> np.ndarray:
        """Mean of |R[u, c] - R[v, c]| over pairs u != v, per column, in O(n log n)."""
        n = R.shape[0]
        s = np.sort(R, axis=0)
        rank_weight = (2 * np.arange(n) - n + 1)[:, None]  # sum_{u<v} (s_v - s_u) = sum_k s_k (2k - n + 1)
        return 2.0 * (s * rank_weight).sum(axis=0) / (n * (n - 1))

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    def _sample_reward(self, action: int, obs: Observation) -> np.ndarray:
        R, n = self._relevance, self._relevance.shape[0]
        u = int(self.np_random.integers(n))
        reward = np.empty(len(self._objectives))
        for j, o in enumerate(self._objectives):
            if o == "relevance":
                reward[j] = R[u, action]
            else:
                v = int(self.np_random.integers(n - 1))
                v += v >= u  # uniform over members other than u
                reward[j] = 1.0 - abs(R[u, action] - R[v, action])
        return reward
