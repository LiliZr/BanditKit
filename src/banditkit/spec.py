"""Problem description (`EnvSpec`) and what a policy sees at each step (`Observation`)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple

import numpy as np

from ._validation import check_float, check_int

__all__ = ["EnvSpec", "Observation", "FAMILIES"]

#: Problem families: classical/multi-objective MAB, linear, multi-user linear.
FAMILIES = ("mab", "linear", "multi_user")


@dataclass(frozen=True)
class EnvSpec:
    """Static description of a bandit problem, given to policies in ``reset``.

    Parameters
    ----------
    n_actions : int
        Number of actions (arms) K available at each step.
    n_features : int or None
        Dimension d of the action features. None for a classical (non-contextual) MAB.
    reward_dim : int
        Dimension D of the reward vector (1 for a scalar reward), as in MO-Gymnasium.
    n_users : int
        Number of users (> 1 only for multi-user linear problems).
    horizon : int or None
        Number of steps T, when known. UCB-V2, UCB-minV2 and EUCBV need it.
    reward_range : float or None
        Known reward amplitude b (rewards lie in an interval of length b).
    feature_norm : float or None
        Upper bound L on the Euclidean norm of action features.
    noise_std : tuple of float or None
        Per objective, an upper bound on the noise standard deviation (sub-Gaussian
        constant R of the theoretical confidence bounds), when the environment knows it.
    theta_norm : tuple of float or None
        Per objective, an upper bound S on the norm of the unknown parameter
        (largest over users), for linear problems whose parameter is known.
    """

    n_actions: int
    n_features: Optional[int] = None
    reward_dim: int = 1
    n_users: int = 1
    horizon: Optional[int] = None
    reward_range: Optional[float] = None
    feature_norm: Optional[float] = None
    noise_std: Optional[Tuple[float, ...]] = None
    theta_norm: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        check_int("n_actions", self.n_actions, low=1)
        check_int("n_features", self.n_features, low=1, allow_none=True)
        check_int("reward_dim", self.reward_dim, low=1)
        check_int("n_users", self.n_users, low=1)
        check_int("horizon", self.horizon, low=1, allow_none=True)
        check_float("reward_range", self.reward_range, low=0.0, low_inclusive=False, allow_none=True)
        check_float("feature_norm", self.feature_norm, low=0.0, allow_none=True)
        if self.n_users > 1 and self.n_features is None:
            raise ValueError("Several users require action features (multi-user linear bandit).")
        for name in ("noise_std", "theta_norm"):
            value = getattr(self, name)
            if value is not None:
                arr = np.broadcast_to(np.asarray(value, dtype=np.float64), (self.reward_dim,))
                if np.any(~np.isfinite(arr)) or np.any(arr < 0):
                    raise ValueError(f"`{name}` must be finite and >= 0, got {value!r}.")
                object.__setattr__(self, name, tuple(float(v) for v in arr))

    def scalar_noise_std(self, weights: Any) -> Optional[float]:
        """Noise bound of the scalarized reward ``w . r``: sum of |w_i| R_i (valid for any noise correlation)."""
        return None if self.noise_std is None else float(np.abs(weights) @ np.asarray(self.noise_std))

    def scalar_theta_norm(self, weights: Any) -> Optional[float]:
        """Bound on the norm of the scalarized parameter ``w . theta``: sum of |w_i| S_i."""
        return None if self.theta_norm is None else float(np.abs(weights) @ np.asarray(self.theta_norm))

    @property
    def family(self) -> str:
        """``"mab"``, ``"linear"`` or ``"multi_user"``."""
        if self.n_features is None:
            return "mab"
        return "multi_user" if self.n_users > 1 else "linear"

    def with_horizon(self, horizon: int) -> "EnvSpec":
        """Copy of the spec with a known horizon."""
        return replace(self, horizon=horizon)


@dataclass
class Observation:
    """What the policy sees before choosing an action.

    Attributes
    ----------
    t : int
        Step index, starting at 0.
    user_id : int
        Current user (always 0 for single-user problems).
    action_features : ndarray of shape (n_actions, n_features) or None
        Features of the available actions; None for a classical MAB. Environments
        with a fixed action set return the same array at every step, so policies
        can cache quantities derived from it (projections, norms).
    """

    t: int
    user_id: int = 0
    action_features: Optional[np.ndarray] = None
