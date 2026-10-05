"""Base class of environments, with a Gymnasium-style interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .._utils import make_rng
from ..spec import EnvSpec, Observation

__all__ = ["Environment"]


class Environment(ABC):
    """A stochastic bandit problem.

    The interface follows Gymnasium, with a vector reward as in MO-Gymnasium::

        obs, info = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(action)

    ``reward`` has shape (reward_dim,). Bandit steps are independent, so
    ``terminated`` is always False; ``truncated`` becomes True when a finite
    source of steps (a user stream) is exhausted.

    Unlike a Gymnasium environment, a bandit environment also exposes its mean
    rewards (:meth:`mean_rewards`), which experiments use to compute the regret.

    Subclasses implement :attr:`spec`, :meth:`_observation`, :meth:`_sample_reward`
    and :meth:`mean_rewards`. Set ``static_means = False`` when the mean rewards
    of a user can change between steps (it disables the cache of the optimum).
    """

    static_means: bool = True

    def __init__(self) -> None:
        self.np_random = make_rng(None)
        self._t = 0
        self._obs: Optional[Observation] = None
        self._best_cache: Dict[Any, float] = {}

    # ------------------------------------------------------------ to implement
    @property
    @abstractmethod
    def spec(self) -> EnvSpec:
        """Static description of the problem."""

    @abstractmethod
    def _observation(self, t: int) -> Observation:
        """Observation at step ``t``."""

    @abstractmethod
    def _sample_reward(self, action: int, obs: Observation) -> np.ndarray:
        """Noisy reward vector (reward_dim,) of ``action`` at ``obs``."""

    @abstractmethod
    def mean_rewards(self, obs: Observation) -> np.ndarray:
        """Mean reward vectors of all actions at ``obs``, shape (n_actions, reward_dim)."""

    def _on_reset(self) -> None:
        """Hook for subclasses with per-episode state."""

    # ---------------------------------------------------------- Gymnasium API
    @property
    def max_episode_steps(self) -> Optional[int]:
        """Number of steps the environment can serve (None: unbounded)."""
        return None

    def reset(self, seed: Any = None) -> Tuple[Observation, Dict[str, Any]]:
        """Start a new episode. Returns ``(observation, info)``."""
        self.np_random = make_rng(seed)
        self._t = 0
        self._on_reset()
        self._obs = self._observation(0)
        return self._obs, {}

    def step(self, action: Any) -> Tuple[Observation, np.ndarray, bool, bool, Dict[str, Any]]:
        """Play ``action``. Returns ``(next_observation, reward, terminated, truncated, info)``."""
        if self._obs is None:
            raise RuntimeError("Call reset() before step().")
        action = self.check_action(action)
        reward = self._sample_reward(action, self._obs)
        self._t += 1
        limit = self.max_episode_steps
        truncated = limit is not None and self._t >= limit
        if not truncated:
            self._obs = self._observation(self._t)
        return self._obs, reward, False, truncated, {}

    # ------------------------------------------------------ bandit quantities
    def mean_reward(self, action: int, obs: Observation) -> np.ndarray:
        """Mean reward vector of one action, shape (reward_dim,)."""
        return self.mean_rewards(obs)[action]

    def best_mean_reward(self, weights: np.ndarray, obs: Observation) -> float:
        """Best scalarized mean reward ``max_a weights . mu_a`` at ``obs``."""
        if not self.static_means:
            return float(np.max(self.mean_rewards(obs) @ weights))
        key = (obs.user_id, weights.tobytes())
        if key not in self._best_cache:
            self._best_cache[key] = float(np.max(self.mean_rewards(obs) @ weights))
        return self._best_cache[key]

    def pareto_gaps(self, obs: Observation) -> np.ndarray:
        """Pareto suboptimality gap of every action, shape (n_actions,).

        The gap of action i is the smallest epsilon >= 0 such that ``mu_i + epsilon``
        (added to every objective) is dominated by no action. Pareto optimal
        actions have a zero gap (Drugan and Nowe, 2013).
        """
        mu = self.mean_rewards(obs)
        diff = mu[None, :, :] - mu[:, None, :]  # diff[i, j] = mu_j - mu_i
        return np.maximum(diff.min(axis=2), 0.0).max(axis=1)

    def extra_metrics(self, action: int, obs: Observation) -> Optional[Dict[str, float]]:
        """Optional dataset metrics for this step (NDCG, CTR...). None by default."""
        return None

    def check_action(self, action: Any) -> int:
        """Return ``action`` as an int after checking it is a valid action index."""
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError(f"Actions must be integer indices, got {action!r}.")
        action = int(action)
        if not 0 <= action < self.spec.n_actions:
            raise IndexError(f"Action {action} out of range [0, {self.spec.n_actions}).")
        return action

    def __repr__(self) -> str:
        s = self.spec
        parts = [f"K={s.n_actions}"]
        if s.n_features is not None:
            parts.append(f"d={s.n_features}")
        if s.reward_dim > 1:
            parts.append(f"D={s.reward_dim}")
        if s.n_users > 1:
            parts.append(f"users={s.n_users}")
        return f"{type(self).__name__}({', '.join(parts)})"
