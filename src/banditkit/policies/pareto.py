"""Pareto UCB1 for multi-objective bandits (Drugan and Nowe, 2013)."""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from .._utils import pareto_front
from .._validation import ParamSpec
from ..spec import EnvSpec, Observation
from .mab import ALPHA, MABPolicy

__all__ = ["ParetoUCB1"]


class ParetoUCB1(MABPolicy):
    """Pareto UCB1.

    Each action gets the index vector ``mean_vector + b * sqrt(2 ln(n (D A)^(1/4)) / n_a)``
    (the same bonus on every objective), with n the total number of pulls and D
    the number of objectives. The action is drawn uniformly among those whose
    index vectors are Pareto non-dominated.

    A is the number of Pareto optimal actions, unknown in practice: by default it
    is replaced by the number of actions K (the paper's "empirical Pareto UCB1").
    The scalarization weights are not used to choose actions; experiments still
    report the scalarized regret, and the Pareto regret if tracked.
    """

    base_name = "ParetoUCB1"
    params = {
        "alpha": ALPHA,
        "pareto_front_size": ParamSpec("int", default=None, low=1, allow_none=True,
                                       doc="Number A of Pareto optimal actions, if known (None: K)."),
    }

    def _check_spec(self, spec: EnvSpec) -> None:
        if self.pareto_front_size is not None and self.pareto_front_size > spec.n_actions:
            raise ValueError(f"`pareto_front_size` ({self.pareto_front_size}) cannot exceed n_actions ({spec.n_actions}).")

    def _reset(self) -> None:
        super()._reset()
        self._front = np.arange(self.spec.n_actions)
        self._bonus = np.full(self.spec.n_actions, np.inf)

    def _predict(self, obs: Observation) -> int:
        action = self._unpulled_action()
        if action is not None:
            return action
        n_pareto = self.pareto_front_size or self.spec.n_actions
        log_term = math.log(self.t * (self.spec.reward_dim * n_pareto) ** 0.25)
        self._bonus = self.alpha * self.reward_amplitude * np.sqrt(2.0 * log_term / self.counts)
        self._front = pareto_front(self.mean_vectors + self._bonus[:, None])
        return int(self._front[self.rng.integers(self._front.size)])

    def diagnostics(self) -> Dict[str, Any]:
        in_front = np.zeros(self.spec.n_actions, dtype=bool)
        in_front[self._front] = True
        return {"mean_vectors": self.mean_vectors.copy(), "bonus": self._bonus.copy(), "in_front": in_front}
