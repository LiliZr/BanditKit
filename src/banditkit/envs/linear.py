"""Linear bandit environments with a single user."""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .._validation import check_array, check_int
from ..spec import EnvSpec, Observation
from .base import Environment

__all__ = ["LinearEnv", "DynamicLinearEnv"]


def as_theta(theta: Any, n_features: int, name: str = "theta") -> np.ndarray:
    """Parameter as an array of shape (reward_dim, n_features)."""
    th = check_array(name, theta, ndim=(1, 2))
    th = th[None, :] if th.ndim == 1 else th
    if th.shape[1] != n_features:
        raise ValueError(f"`{name}` must have {n_features} columns to match the action features, "
                         f"got shape {np.shape(theta)}.")
    return th


def as_noise_std(noise_std: Any, reward_dim: int) -> np.ndarray:
    """Noise standard deviation as an array of shape (reward_dim,)."""
    std = check_array("noise_std", noise_std, ndim=(0, 1), nonneg=True)
    try:
        return np.broadcast_to(std, (reward_dim,)).astype(np.float64)
    except ValueError:
        raise ValueError(f"`noise_std` must be a scalar or have length reward_dim={reward_dim}, "
                         f"got shape {std.shape}.") from None


class LinearEnv(Environment):
    """Fixed set of actions with features in R^d and a linear mean reward.

    The mean reward of action x on objective j is ``theta[j] @ x``; the observed
    reward adds Gaussian noise.

    Parameters
    ----------
    action_features : array of shape (n_actions, n_features)
        Features of the actions (arms).
    theta : array of shape (n_features,) or (reward_dim, n_features)
        Unknown parameter, one row per objective.
    noise_std : float or array of shape (reward_dim,)
        Standard deviation of the Gaussian noise.
    normalize : bool
        Map mean rewards to [0, 1] per objective with the min and max over actions,
        as in the experiments of the Stochastic-linear-bandits repository.
    """

    def __init__(self, action_features: Any, theta: Any, noise_std: Any = 0.1, normalize: bool = False) -> None:
        super().__init__()
        self.action_features = np.ascontiguousarray(check_array("action_features", action_features, ndim=2))
        n_actions, n_features = self.action_features.shape
        self.theta = as_theta(theta, n_features)
        self._noise_std = as_noise_std(noise_std, self.theta.shape[0])
        means = self.action_features @ self.theta.T  # (n_actions, reward_dim)
        if normalize:
            low, high = means.min(axis=0), means.max(axis=0)
            means = (means - low) / np.where(high > low, high - low, 1.0)
        self._means = means
        # After min-max normalization the means are no longer linear in theta: its norm is not a valid S.
        self._spec = EnvSpec(n_actions=n_actions, n_features=n_features, reward_dim=self.theta.shape[0],
                             feature_norm=float(np.linalg.norm(self.action_features, axis=1).max()),
                             noise_std=self._noise_std,
                             theta_norm=None if normalize else np.linalg.norm(self.theta, axis=1))

    @classmethod
    def from_means(cls, action_features: Any, means: Any, noise_std: Any = 0.1) -> "LinearEnv":
        """Actions with features and given mean rewards, linear or not (e.g. observed ratings).

        ``theta`` is then None: linear policies face a misspecified model.
        """
        env = cls.__new__(cls)
        Environment.__init__(env)
        env.action_features = np.ascontiguousarray(check_array("action_features", action_features, ndim=2))
        n_actions, n_features = env.action_features.shape
        m = check_array("means", means, ndim=(1, 2))
        m = m[:, None] if m.ndim == 1 else m
        if m.shape[0] != n_actions:
            raise ValueError(f"`means` must have one row per action ({n_actions}), got shape {np.shape(means)}.")
        env.theta = None
        env._noise_std = as_noise_std(noise_std, m.shape[1])
        env._means = np.ascontiguousarray(m)
        env._spec = EnvSpec(n_actions=n_actions, n_features=n_features, reward_dim=m.shape[1],
                            feature_norm=float(np.linalg.norm(env.action_features, axis=1).max()),
                            noise_std=env._noise_std, theta_norm=None)
        return env

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    def _observation(self, t: int) -> Observation:
        return Observation(t=t, action_features=self.action_features)

    def _sample_reward(self, action: int, obs: Observation) -> np.ndarray:
        return self._means[action] + self._noise_std * self.np_random.standard_normal(self._noise_std.size)

    def mean_rewards(self, obs: Observation) -> np.ndarray:
        return self._means

    def mean_reward(self, action: int, obs: Observation) -> np.ndarray:
        return self._means[action]


class DynamicLinearEnv(Environment):
    """Linear bandit whose available actions change at every step.

    Parameters
    ----------
    action_sampler : callable (t, rng) -> array of shape (n_actions, n_features)
        Draws the features of the actions available at step t.
    theta : array of shape (n_features,) or (reward_dim, n_features)
        Unknown parameter.
    n_actions, n_features : int
        Shape of every array returned by ``action_sampler``.
    noise_std : float or array of shape (reward_dim,)
        Standard deviation of the Gaussian noise.
    feature_norm : float, optional
        Upper bound on the norm of action features, if known.
    """

    static_means = False

    def __init__(self, action_sampler: Callable[[int, np.random.Generator], np.ndarray], theta: Any,
                 n_actions: int, n_features: int, noise_std: Any = 0.1, feature_norm: Optional[float] = None) -> None:
        super().__init__()
        if not callable(action_sampler):
            raise TypeError("`action_sampler` must be a callable (t, rng) -> array of shape (n_actions, n_features).")
        self._sampler = action_sampler
        n_actions = check_int("n_actions", n_actions, low=1)
        n_features = check_int("n_features", n_features, low=1)
        self.theta = as_theta(theta, n_features)
        self._noise_std = as_noise_std(noise_std, self.theta.shape[0])
        self._spec = EnvSpec(n_actions=n_actions, n_features=n_features, reward_dim=self.theta.shape[0],
                             feature_norm=feature_norm, noise_std=self._noise_std,
                             theta_norm=np.linalg.norm(self.theta, axis=1))

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    def _observation(self, t: int) -> Observation:
        X = check_array("action_sampler output", self._sampler(t, self.np_random),
                        shape=(self._spec.n_actions, self._spec.n_features))
        return Observation(t=t, action_features=np.ascontiguousarray(X))

    def _sample_reward(self, action: int, obs: Observation) -> np.ndarray:
        mean = obs.action_features[action] @ self.theta.T
        return mean + self._noise_std * self.np_random.standard_normal(self._noise_std.size)

    def mean_rewards(self, obs: Observation) -> np.ndarray:
        return obs.action_features @ self.theta.T

    def mean_reward(self, action: int, obs: Observation) -> np.ndarray:
        return obs.action_features[action] @ self.theta.T
