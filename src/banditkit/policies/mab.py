"""Policies for classical and multi-objective (scalarized) multi-armed bandits.

All of them learn from the scalarized reward ``weights @ reward``. Variance-aware
policies use the empirical variance of the scalarized reward, i.e. the
estimate of ``w^T Sigma w`` for the covariance Sigma of the reward vector.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, ClassVar, Dict, Optional

import numpy as np

from .._utils import argmax_random
from .._validation import ParamSpec
from ..spec import EnvSpec, Observation
from .base import Policy

__all__ = ["Random", "EpsilonGreedy", "Softmax", "UCB1", "UCB1Tuned", "UCB1Normal", "UCBV", "UCBV2", "UCBMin",
           "UCBMinV1", "UCBMinV2"]

ALPHA = ParamSpec("float", default=1.0, low=0.0, doc="Multiplier of the exploration bonus (1: value of the paper).")
REWARD_RANGE = ParamSpec("float", default=None, low=0.0, low_inclusive=False, allow_none=True,
                         doc="Reward amplitude b (rewards in an interval of length b). "
                             "None: taken from the environment if known, else 1.")


class MABPolicy(Policy):
    """Shared statistics of MAB policies: pull counts, empirical means and variances."""

    families = frozenset({"mab"})
    params = {"reward_range": REWARD_RANGE}
    #: Whether the index or update uses the reward amplitude b (then its origin is reported).
    uses_reward_range: ClassVar[bool] = True

    def _reset(self) -> None:
        n_actions, reward_dim = self.spec.n_actions, self.spec.reward_dim
        self.reward_amplitude = self.reward_range or self.spec.reward_range or 1.0
        if self.uses_reward_range and self.reward_range is None:
            if self.spec.reward_range is not None:
                self._note("from_problem", f"reward_range: b={self.spec.reward_range:.3g}")
            else:
                self._note("assumed", "reward_range: b=1 (unknown for this environment)")
        self.counts = np.zeros(n_actions, dtype=np.int64)    # pulls of each action
        self.means = np.zeros(n_actions)                     # mean scalarized reward
        self.mean_vectors = np.zeros((n_actions, reward_dim))  # mean reward vector
        self._sq_dev = np.zeros(n_actions)                   # Welford sum of squared deviations
        self.t = 0                                           # total number of pulls
        self._n_unpulled = n_actions

    def _update(self, obs: Observation, action: int, reward: np.ndarray, scalar_reward: float) -> None:
        self._n_unpulled -= self.counts[action] == 0
        self.counts[action] += 1
        n = self.counts[action]
        delta = scalar_reward - self.means[action]
        self.means[action] += delta / n
        self._sq_dev[action] += delta * (scalar_reward - self.means[action])
        self.mean_vectors[action] += (reward - self.mean_vectors[action]) / n
        self.t += 1

    @property
    def variances(self) -> np.ndarray:
        """Empirical variance of the scalarized reward, normalized by n (0 for unpulled actions)."""
        return self._sq_dev / np.maximum(self.counts, 1)

    def _unpulled_action(self) -> Optional[int]:
        """A random action never pulled yet, or None. Index policies play each action once first."""
        if self._n_unpulled == 0:
            return None
        candidates = np.flatnonzero(self.counts == 0)
        return int(candidates[self.rng.integers(candidates.size)])

    def _log_t(self) -> float:
        return math.log(max(self.t, 1))

    def diagnostics(self) -> Dict[str, Any]:
        d = {"mean": self.means.copy()}
        if self.spec.reward_dim > 1:
            d["mean_vectors"] = self.mean_vectors.copy()
        return d


class Random(MABPolicy):
    """Uniformly random action. Accepted on every problem family (sanity-check baseline)."""

    families = frozenset({"mab", "linear", "multi_user"})
    base_name = "Random"
    uses_reward_range = False

    def _predict(self, obs: Observation) -> int:
        return int(self.rng.integers(self.spec.n_actions))


class EpsilonGreedy(MABPolicy):
    """Greedy on the empirical means, uniform exploration with probability epsilon.

    ``schedule="auer"`` uses ``epsilon_t = min(1, c K / (d^2 t))`` (Auer et al., 2002).
    """

    base_name = "EpsilonGreedy"
    params = {
        "epsilon": ParamSpec("float", default=0.1, low=0.0, high=1.0, doc="Exploration probability (constant schedule)."),
        "schedule": ParamSpec("choice", default="constant", choices=("constant", "auer"), doc="Exploration schedule."),
        "c": ParamSpec("float", default=1.0, low=0.0, low_inclusive=False, doc="Constant c of the 'auer' schedule."),
        "d": ParamSpec("float", default=None, low=0.0, low_inclusive=False, high=1.0, allow_none=True,
                       doc="Lower bound on the smallest gap, required by the 'auer' schedule."),
    }

    uses_reward_range = False
    default_search_params = ("epsilon",)

    def _check_params(self) -> None:
        if self.schedule == "auer" and self.d is None:
            raise ValueError("EpsilonGreedy(schedule='auer') needs `d`, a lower bound on the smallest gap.")

    def _reset(self) -> None:
        super()._reset()
        if self.schedule == "constant" and self.epsilon == self._all_params()["epsilon"].default:
            self._note("arbitrary", f"epsilon={self.epsilon!r} (no theoretical value for a constant schedule)")

    def _predict(self, obs: Observation) -> int:
        action = self._unpulled_action()
        if action is not None:
            return action
        if self.schedule == "constant":
            epsilon = self.epsilon
        else:
            epsilon = min(1.0, self.c * self.spec.n_actions / (self.d ** 2 * self.t))
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.spec.n_actions))
        return argmax_random(self.means, self.rng)


class Softmax(MABPolicy):
    """Boltzmann exploration: P(a) proportional to exp(mean_a / tau)."""

    base_name = "Softmax"
    params = {"tau": ParamSpec("float", default=0.1, low=0.0, low_inclusive=False, doc="Temperature (as in MABWiser).")}
    arbitrary_defaults = {"tau": "no theoretical value"}
    uses_reward_range = False
    default_search_params = ("tau",)

    def _predict(self, obs: Observation) -> int:
        action = self._unpulled_action()
        if action is not None:
            return action
        p = np.exp((self.means - self.means.max()) / self.tau)
        return int(self.rng.choice(self.spec.n_actions, p=p / p.sum()))


class IndexPolicy(MABPolicy):
    """Optimistic policies: play the argmax of ``mean + bonus``. Subclasses define ``_bonus``."""

    params = {"alpha": ALPHA}

    def _reset(self) -> None:
        super()._reset()
        self._last_bonus = np.full(self.spec.n_actions, np.inf)

    def _bonus(self) -> np.ndarray:
        raise NotImplementedError

    def _predict(self, obs: Observation) -> int:
        action = self._unpulled_action()
        if action is not None:
            return action
        self._last_bonus = self._bonus()
        return argmax_random(self.means + self._last_bonus, self.rng)

    def diagnostics(self) -> Dict[str, Any]:
        return {**super().diagnostics(), "bonus": self._last_bonus.copy()}


class UCB1(IndexPolicy):
    """UCB1 (Auer et al., 2002): bonus ``alpha * b * sqrt(2 ln t / n)``.

    With ``sigma`` (sigma-sub-Gaussian rewards) the bonus is ``alpha * sigma * sqrt(2 ln t / n)``.
    """

    base_name = "UCB1"
    params = {"sigma": ParamSpec("float", default=None, low=0.0, low_inclusive=False, allow_none=True,
                                 doc="Sub-Gaussian constant of the rewards; replaces b in the bonus.")}

    def _check_params(self) -> None:
        if self.sigma is not None and self.reward_range is not None:
            raise ValueError("UCB1 takes either `sigma` (sub-Gaussian rewards) or `reward_range`, not both.")

    def _bonus(self) -> np.ndarray:
        scale = self.sigma if self.sigma is not None else self.reward_amplitude
        return self.alpha * scale * np.sqrt(2.0 * self._log_t() / self.counts)


class UCB1Tuned(IndexPolicy):
    """UCB1-Tuned (Auer et al., 2002).

    Bonus ``alpha * b * sqrt((ln t / n) * min(1/4, V / b^2 + sqrt(2 ln t / n)))``
    with V the empirical variance. For b = 1 this is the paper's index.
    """

    base_name = "UCB1Tuned"

    def _bonus(self) -> np.ndarray:
        log_t, n, b = self._log_t(), self.counts, self.reward_amplitude
        variance_bound = self.variances / b ** 2 + np.sqrt(2.0 * log_t / n)
        return self.alpha * b * np.sqrt(log_t / n * np.minimum(0.25, variance_bound))


class UCB1Normal(IndexPolicy):
    """UCB1-Normal (Auer et al., 2002), for Gaussian rewards.

    Actions pulled fewer than ``ceil(8 ln t)`` times are played first. Otherwise
    the bonus is ``alpha * sqrt(16 * s^2 * ln(t - 1) / n)``, with s^2 the sample
    variance (normalized by n - 1).
    """

    base_name = "UCB1Normal"

    def _predict(self, obs: Observation) -> int:
        action = self._unpulled_action()
        if action is not None:
            return action
        under_sampled = np.flatnonzero(self.counts < math.ceil(8.0 * self._log_t()))
        if under_sampled.size:
            return int(under_sampled[self.rng.integers(under_sampled.size)])
        return super()._predict(obs)

    def _bonus(self) -> np.ndarray:
        n = self.counts
        sample_var = np.where(n > 1, self._sq_dev / np.maximum(n - 1, 1), np.inf)
        return self.alpha * np.sqrt(16.0 * sample_var * math.log(max(self.t - 1, 1)) / n)


class UCBV(IndexPolicy):
    """UCB-V (Audibert, Munos and Szepesvari, 2009).

    Bonus ``alpha * (sqrt(2 V E / n) + c * 3 b E / n)`` with ``E = zeta ln t`` and V
    the empirical variance (normalized by n). Theorem 4 of the paper takes c = 1, zeta > 1.
    """

    base_name = "UCBV"
    params = {
        "zeta": ParamSpec("float", default=1.2, low=0.0, low_inclusive=False, doc="Exploration constant zeta of E = zeta ln t."),
        "c": ParamSpec("float", default=1.0, low=0.0, low_inclusive=False, doc="Weight c of the range term 3 b E / n."),
    }

    def _check_params(self) -> None:
        if self.zeta < 1.0:
            warnings.warn("UCB-V with zeta < 1 can have polynomial regret (Audibert et al., 2009, Thm 5).", stacklevel=3)

    def _variance_bonus(self) -> np.ndarray:
        exploration = self.zeta * self._log_t()
        n = self.counts
        return np.sqrt(2.0 * self.variances * exploration / n) + self.c * 3.0 * self.reward_amplitude * exploration / n

    def _bonus(self) -> np.ndarray:
        return self.alpha * self._variance_bonus()


class UCBV2(IndexPolicy):
    """Variance-aware UCB of Fan et al. (2025), called UCB-V2 here.

    Bonus ``alpha * max(sigma_hat / sqrt(L), b / sqrt(n)) * L / sqrt(n)`` with
    ``L = rho ln T`` and sigma_hat the empirical standard deviation. For b = 1
    this is the paper's index; its guarantees assume rho > 1. Needs the horizon T.
    """

    base_name = "UCBV2"
    params = {"rho": ParamSpec("float", default=2.0, low=0.0, low_inclusive=False, doc="Exploration constant rho of L = rho ln T.")}

    def _check_params(self) -> None:
        if self.rho <= 1.0:
            warnings.warn("UCB-V2 guarantees assume rho > 1 (Fan et al., 2025).", stacklevel=3)

    def _check_spec(self, spec: EnvSpec) -> None:
        if spec.horizon is None:
            raise ValueError(f"{type(self).__name__} needs the horizon T: give EnvSpec(horizon=...) or use an Experiment.")

    def _variance_bonus(self) -> np.ndarray:
        level = self.rho * math.log(max(self.spec.horizon, 2))
        sqrt_n = np.sqrt(self.counts)
        return np.maximum(np.sqrt(self.variances) / math.sqrt(level), self.reward_amplitude / sqrt_n) * level / sqrt_n

    def _bonus(self) -> np.ndarray:
        return self.alpha * self._variance_bonus()


class UCBMin(IndexPolicy):
    """UCB-min: for each action, the smaller of the UCB1 bonus and a variance-aware bonus.

    ``variant="v1"`` pairs UCB1 with UCB-V (UCB-minV1); ``variant="v2"`` pairs it
    with UCB-V2 (UCB-minV2, needs the horizon). The choice is made per action and per step.
    """

    base_name = "UCBMin"
    params = {
        "variant": ParamSpec("choice", default="v1", choices=("v1", "v2"), doc="'v1': UCB-V bonus, 'v2': UCB-V2 bonus."),
        "zeta": UCBV.params["zeta"],
        "c": UCBV.params["c"],
        "rho": UCBV2.params["rho"],
    }

    def _check_spec(self, spec: EnvSpec) -> None:
        if self.variant == "v2":
            UCBV2._check_spec(self, spec)

    def _reset(self) -> None:
        super()._reset()
        self._variance_bonus_used = np.zeros(self.spec.n_actions, dtype=bool)

    def _bonus(self) -> np.ndarray:
        ucb1_bonus = self.reward_amplitude * np.sqrt(2.0 * self._log_t() / self.counts)
        variance_bonus = (UCBV if self.variant == "v1" else UCBV2)._variance_bonus(self)
        self._variance_bonus_used = variance_bonus < ucb1_bonus
        return self.alpha * np.minimum(ucb1_bonus, variance_bonus)

    @property
    def label(self) -> str:
        return self._params.get("name") or ("UCBMinV1" if self.variant == "v1" else "UCBMinV2")

    def diagnostics(self) -> Dict[str, Any]:
        return {**super().diagnostics(), "variance_bonus_used": self._variance_bonus_used.copy()}


class UCBMinV1(UCBMin):
    """UCB-minV1: per-action minimum of the UCB1 and UCB-V bonuses."""

    params = {"variant": ParamSpec("choice", default="v1", choices=("v1",), doc="Fixed to 'v1'.")}


class UCBMinV2(UCBMin):
    """UCB-minV2: per-action minimum of the UCB1 and UCB-V2 bonuses (needs the horizon)."""

    params = {"variant": ParamSpec("choice", default="v2", choices=("v2",), doc="Fixed to 'v2'.")}
