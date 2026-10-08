"""``make_env``: build the right environment from the shape of its arguments."""
from __future__ import annotations

from typing import Any

from .._validation import check_choice
from .base import Environment
from .linear import DynamicLinearEnv, LinearEnv
from .mab import BernoulliMAB, EmpiricalMAB, GaussianMAB
from .multi_user import MultiUserLinearEnv

__all__ = ["make_env"]


def make_env(*, means: Any = None, samples: Any = None, action_features: Any = None, theta: Any = None,
             user_thetas: Any = None, noise: str = "gaussian", noise_std: Any = None, **kwargs: Any) -> Environment:
    """Create an environment; the bandit family is detected from the arguments.

    ============================================  ===================================
    Arguments                                     Environment
    ============================================  ===================================
    ``means`` (n_actions,)                        classical MAB (Gaussian or Bernoulli)
    ``means`` (n_actions, reward_dim)             multi-objective MAB
    ``samples`` (one array per action)            MAB resampling observed rewards
    ``action_features`` + ``theta``               linear bandit
    callable ``action_features`` + ``theta``      linear bandit with changing actions
    ``action_features`` + ``user_thetas``         multi-user linear bandit
    ============================================  ===================================

    Parameters
    ----------
    noise : {"gaussian", "bernoulli"}
        Reward distribution of a MAB given by ``means``.
    noise_std : float or array, optional
        Gaussian noise standard deviation (default 1 for a MAB, 0.1 for linear problems).
    **kwargs
        Passed to the environment class (``noise_cov``, ``clip``, ``normalize``,
        ``user_stream``, ``user_probs``, ``n_actions``...).
    """
    given = {name for name, value in dict(means=means, samples=samples, action_features=action_features,
                                          theta=theta, user_thetas=user_thetas).items() if value is not None}
    noise = check_choice("noise", noise, ("gaussian", "bernoulli"))
    if not given:
        raise TypeError("make_env needs `means`, `samples`, or `action_features` with `theta` or `user_thetas`.")

    if given & {"means", "samples"}:
        if len(given) > 1:
            raise TypeError(f"`means` or `samples` describe a MAB on their own; got {sorted(given)}.")
        if "samples" in given:
            if noise_std is not None or kwargs:
                raise TypeError("`samples` environments take no noise parameters.")
            return EmpiricalMAB(samples)
        if noise == "bernoulli":
            if noise_std is not None or kwargs:
                raise TypeError("Bernoulli actions take no noise parameters.")
            return BernoulliMAB(means)
        return GaussianMAB(means, noise_std=1.0 if noise_std is None else noise_std, **kwargs)

    if noise != "gaussian":
        raise ValueError("Linear environments only support Gaussian noise.")
    if "action_features" not in given:
        raise TypeError("Linear environments need `action_features` (array of shape (n_actions, n_features)).")
    if ("theta" in given) == ("user_thetas" in given):
        raise TypeError("Give exactly one of `theta` (single user) or `user_thetas` (several users).")
    std = 0.1 if noise_std is None else noise_std
    if "user_thetas" in given:
        return MultiUserLinearEnv(action_features, user_thetas, noise_std=std, **kwargs)
    if callable(action_features):
        return DynamicLinearEnv(action_features, theta, noise_std=std, **kwargs)
    return LinearEnv(action_features, theta, noise_std=std, **kwargs)
