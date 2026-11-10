"""Thompson Sampling for classical and multi-objective (scalarized) multi-armed bandits."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .._utils import argmax_random
from .._validation import ParamSpec
from ..spec import Observation
from .mab import MABPolicy

__all__ = ["BernoulliThompsonSampling", "GaussianThompsonSampling"]


class BernoulliThompsonSampling(MABPolicy):
    """Beta-Bernoulli Thompson Sampling.

    Rewards must lie in [0, b]. A non-binary reward r counts as a success with
    probability r / b (binarization trick of Agrawal and Goyal, 2012).
    """

    base_name = "ThompsonSampling"
    params = {
        "prior_alpha": ParamSpec("float", default=1.0, low=0.0, low_inclusive=False, doc="Beta prior: pseudo-successes."),
        "prior_beta": ParamSpec("float", default=1.0, low=0.0, low_inclusive=False, doc="Beta prior: pseudo-failures."),
    }

    def _reset(self) -> None:
        super()._reset()
        self.successes = np.zeros(self.spec.n_actions)
        self.failures = np.zeros(self.spec.n_actions)

    def _predict(self, obs: Observation) -> int:
        samples = self.rng.beta(self.prior_alpha + self.successes, self.prior_beta + self.failures)
        return argmax_random(samples, self.rng)

    def _update(self, obs: Observation, action: int, reward: np.ndarray, scalar_reward: float) -> None:
        super()._update(obs, action, reward, scalar_reward)
        p = scalar_reward / self.reward_amplitude
        if not -1e-9 <= p <= 1 + 1e-9:
            raise ValueError(f"BernoulliThompsonSampling needs scalarized rewards in [0, {self.reward_amplitude}], "
                             f"got {scalar_reward}. Set `reward_range` or use GaussianThompsonSampling.")
        if self.rng.random() < p:
            self.successes[action] += 1
        else:
            self.failures[action] += 1

    def diagnostics(self) -> Dict[str, Any]:
        return {**super().diagnostics(), "posterior_alpha": self.prior_alpha + self.successes,
                "posterior_beta": self.prior_beta + self.failures}


class GaussianThompsonSampling(MABPolicy):
    """Thompson Sampling with Gaussian priors (Agrawal and Goyal, AISTATS 2013, Algorithm 2).

    Samples ``theta_a ~ N(S_a / (n_a + 1), sigma^2 / (n_a + 1))`` with S_a the sum
    of the rewards of action a. The paper takes sigma = 1 for rewards in [0, 1],
    so the default is sigma = b, the reward amplitude. Giving the noise level as
    ``sigma`` turns it into the Gaussian posterior for known Gaussian noise.
    """

    base_name = "ThompsonSampling"
    params = {"sigma": ParamSpec("float", default=None, low=0.0, low_inclusive=False, allow_none=True,
                                 doc="Scale sigma of the sampling (None: the reward amplitude b, as in the paper).")}

    def _reset(self) -> None:
        super()._reset()  # reports where b comes from
        if self.sigma is not None:  # b is not used then
            self._notes = [note for note in self._notes if not note[1].startswith("reward_range")]
        self._sigma = self.sigma if self.sigma is not None else self.reward_amplitude

    def _posterior(self):
        n_plus_one = self.counts + 1.0
        return self.means * self.counts / n_plus_one, self._sigma / np.sqrt(n_plus_one)

    def _predict(self, obs: Observation) -> int:
        mean, std = self._posterior()
        return argmax_random(mean + std * self.rng.standard_normal(self.spec.n_actions), self.rng)

    def diagnostics(self) -> Dict[str, Any]:
        mean, std = self._posterior()
        return {**super().diagnostics(), "posterior_mean": mean, "posterior_std": std}
