"""Multi-user linear bandit environment (recommendation setting)."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .._validation import check_array
from ..spec import EnvSpec, Observation
from .base import Environment
from .linear import as_noise_std

__all__ = ["MultiUserLinearEnv"]


class MultiUserLinearEnv(Environment):
    """One user arrives per step; each user has their own linear parameter.

    Parameters
    ----------
    action_features : array of shape (n_actions, n_features)
        Item features, shared by all users.
    user_thetas : array of shape (n_users, n_features) or (n_users, reward_dim, n_features)
        Parameter of each user, one row per objective.
    noise_std : float or array of shape (reward_dim,)
        Standard deviation of the Gaussian noise.
    user_stream : sequence of int, optional
        User of each step (e.g. the chronological order of a dataset). Its length
        is the largest horizon the environment can serve.
    user_probs : array of shape (n_users,), optional
        Arrival probabilities when no stream is given (default: uniform).

    The optimum of a user is computed the first time the user arrives and then
    cached, so memory stays linear in the number of users.
    """

    def __init__(self, action_features: Any, user_thetas: Any, noise_std: Any = 0.1,
                 user_stream: Any = None, user_probs: Any = None) -> None:
        super().__init__()
        self.action_features = np.ascontiguousarray(check_array("action_features", action_features, ndim=2))
        n_actions, n_features = self.action_features.shape
        th = check_array("user_thetas", user_thetas, ndim=(2, 3))
        if th.ndim == 2:
            th = th[:, None, :]
        if th.shape[2] != n_features:
            raise ValueError(f"`user_thetas` last dimension must be n_features={n_features}, "
                             f"got shape {np.shape(user_thetas)}.")
        self.user_thetas = np.ascontiguousarray(th)  # (n_users, reward_dim, n_features)
        n_users, reward_dim, _ = th.shape
        self._noise_std = as_noise_std(noise_std, reward_dim)

        if user_stream is not None and user_probs is not None:
            raise ValueError("Give either `user_stream` or `user_probs`, not both.")
        self.user_stream = None
        if user_stream is not None:
            stream = check_array("user_stream", user_stream, ndim=1, dtype=np.int64)
            if stream.min() < 0 or stream.max() >= n_users:
                raise ValueError(f"`user_stream` ids must lie in [0, {n_users}), "
                                 f"found range [{stream.min()}, {stream.max()}].")
            self.user_stream = stream
        self.user_probs = None
        if user_probs is not None:
            p = check_array("user_probs", user_probs, shape=(n_users,), nonneg=True)
            if not np.isclose(p.sum(), 1.0):
                raise ValueError(f"`user_probs` must sum to 1, got {p.sum():.6g}.")
            self.user_probs = p / p.sum()

        self._spec = EnvSpec(n_actions=n_actions, n_features=n_features, reward_dim=reward_dim, n_users=n_users,
                             feature_norm=float(np.linalg.norm(self.action_features, axis=1).max()),
                             noise_std=self._noise_std, theta_norm=np.linalg.norm(th, axis=2).max(axis=0))

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    @property
    def max_episode_steps(self) -> Optional[int]:
        return None if self.user_stream is None else int(self.user_stream.size)

    def _observation(self, t: int) -> Observation:
        if self.user_stream is not None:
            user = int(self.user_stream[t])
        elif self.user_probs is not None:
            user = int(self.np_random.choice(self.user_probs.size, p=self.user_probs))
        else:
            user = int(self.np_random.integers(self.user_thetas.shape[0]))
        return Observation(t=t, user_id=user, action_features=self.action_features)

    def _sample_reward(self, action: int, obs: Observation) -> np.ndarray:
        mean = self.user_thetas[obs.user_id] @ self.action_features[action]
        return mean + self._noise_std * self.np_random.standard_normal(self._noise_std.size)

    def mean_rewards(self, obs: Observation) -> np.ndarray:
        return self.action_features @ self.user_thetas[obs.user_id].T

    def mean_reward(self, action: int, obs: Observation) -> np.ndarray:
        return self.user_thetas[obs.user_id] @ self.action_features[action]

    def best_mean_reward(self, weights: np.ndarray, obs: Observation) -> float:
        key = (obs.user_id, weights.tobytes())
        if key not in self._best_cache:
            # Scalarize theta first: K*d operations instead of K*d*D.
            self._best_cache[key] = float(np.max(self.action_features @ (weights @ self.user_thetas[obs.user_id])))
        return self._best_cache[key]
