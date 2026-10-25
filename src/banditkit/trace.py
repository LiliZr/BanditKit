"""Step-by-step record of one run, used by the explanatory figures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ._validation import check_int, check_weights
from .envs.base import Environment
from .policies._linear_core import weighted_norms
from .policies.base import Policy

__all__ = ["Trace", "trace"]


@dataclass
class Trace:
    """One run of one policy, recorded step by step.

    Attributes
    ----------
    label : str
        Policy label.
    actions : ndarray (horizon,)
        Action played at each step.
    rewards : ndarray (horizon, reward_dim)
        Reward vector observed at each step.
    regret : ndarray (horizon,)
        Cumulative regret.
    weights : ndarray (reward_dim,)
        Scalarization weights.
    snapshot_steps : ndarray (n_snapshots,)
        Steps (1-based) at which the policy diagnostics were recorded.
    snapshots : dict of ndarray
        Diagnostics at those steps (estimated means, bonuses, posteriors...), plus
        ``theta``, ``V_inv`` and ``width_scale`` for a shared LinUCB model of dimension <= 3,
        and ``ucb_all``, ``mean_all``, ``bucket_of``, ``champions``, ``chosen_bucket`` for a policy with arm buckets.
    policy : Policy
        The policy at the end of the run.
    mean_rewards : ndarray (n_actions, reward_dim) or None
        True mean rewards, for single-user problems with fixed means.
    action_features : ndarray (n_actions, n_features) or None
        Action features, for problems with a fixed action set.
    users : ndarray (horizon,) or None
        User of each step (multi-user problems).
    extra : dict
        Environment parameters useful to figures (``theta`` of a linear problem).
    action_names : list of str or None
        Names of the actions, when the environment has them (real data).
    """

    label: str
    actions: np.ndarray
    rewards: np.ndarray
    regret: np.ndarray
    weights: np.ndarray
    snapshot_steps: np.ndarray
    snapshots: Dict[str, np.ndarray]
    policy: Policy
    mean_rewards: Optional[np.ndarray] = None
    action_features: Optional[np.ndarray] = None
    users: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    action_names: Optional[List[str]] = None

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.policy.spec.n_actions)


def trace(env: Environment, policy: Policy, horizon: int, seed: Optional[int] = 0, every: int = 1,
          weights: Any = None) -> Trace:
    """Run ``policy`` once on ``env`` and record what the explanatory figures need.

    Parameters
    ----------
    env : Environment
        The problem.
    policy : Policy
        The policy (a fresh copy is used).
    horizon : int
        Number of steps.
    seed : int, optional
        Seed of the run.
    every : int
        Record the policy diagnostics every ``every`` steps (and at step 1).
    weights : array of shape (reward_dim,), optional
        Scalarization weights (default: uniform).
    """
    horizon = check_int("horizon", horizon, low=1)
    every = check_int("every", every, low=1)
    weights = check_weights(weights, env.spec.reward_dim)
    spec = env.spec.with_horizon(horizon)
    env_seed, policy_seed = np.random.SeedSequence(seed).spawn(2)
    obs, _ = env.reset(seed=env_seed)
    first_obs = obs
    policy = policy.clone()
    policy.reset(spec, weights, seed=policy_seed)

    actions = np.empty(horizon, dtype=np.int64)
    rewards = np.empty((horizon, spec.reward_dim))
    regret = np.empty(horizon)
    users = np.empty(horizon, dtype=np.int64) if spec.n_users > 1 else None
    snapshot_steps: List[int] = []
    snapshots: Dict[str, List[Any]] = {}
    total_regret = 0.0
    for t in range(horizon):
        action = policy.predict(obs)
        if (t + 1) % every == 0 or t == 0:
            # Snapshot at decision time: after predict (bonuses just computed), before update.
            snapshot_steps.append(t + 1)
            state = {k: np.copy(v) for k, v in policy.diagnostics().items()}
            state.update(_linear_state(policy))
            state.update(_bucket_state(policy))
            for k, v in state.items():
                snapshots.setdefault(k, []).append(v)
        next_obs, reward, _, truncated, _ = env.step(action)
        policy.update(obs, action, reward)
        actions[t], rewards[t] = action, reward
        total_regret += env.best_mean_reward(weights, obs) - float(weights @ reward)
        regret[t] = total_regret
        if users is not None:
            users[t] = obs.user_id
        obs = next_obs
        if truncated and t < horizon - 1:
            raise ValueError(f"The environment can serve only {t + 1} steps; reduce `horizon`.")

    consistent = {k: np.asarray(v) for k, v in snapshots.items()
                  if len(v) == len(snapshot_steps) and len({np.shape(x) for x in v}) == 1}
    fixed_means = env.static_means and spec.n_users == 1
    return Trace(label=policy.label, actions=actions, rewards=rewards, regret=regret, weights=weights,
                 snapshot_steps=np.array(snapshot_steps), snapshots=consistent, policy=policy,
                 mean_rewards=np.array(env.mean_rewards(first_obs)) if fixed_means else None,
                 action_features=None if first_obs.action_features is None or not env.static_means
                 else np.array(first_obs.action_features),
                 users=users, extra={"theta": np.array(env.theta)} if hasattr(env, "theta") else {},
                 action_names=getattr(env, "action_names", None))


def _linear_state(policy: Policy) -> Dict[str, Any]:
    """theta, V^-1 and the width scale of a shared LinUCB model (for the ellipsoid figure)."""
    inner = getattr(policy, "inner", None) or policy
    models = getattr(inner, "models", None)
    if not models or getattr(inner, "user_model", None) != "shared" or inner.dim > 3 or 0 not in models:
        return {}
    model = models[0]
    return {"theta": model.theta.copy(), "V_inv": model.V_inv.copy(), "width_scale": inner.alpha * inner._beta(model.n_obs)}


def _bucket_state(policy: Policy) -> Dict[str, Any]:
    """The last two-step bucket selection: UCB of every action, buckets, champions, chosen bucket."""
    inner = getattr(policy, "inner", None) or policy
    buckets = getattr(inner, "_last_buckets", None)
    if buckets is None or buckets.last_inputs is None:
        return {}
    X, theta, V_inv, width_scale, champions = buckets.last_inputs
    ucb_all = X @ theta + width_scale * weighted_norms(X, V_inv)  # what LinUCB without buckets would compare
    bucket_of = np.empty(X.shape[0], dtype=np.int64)
    for g, members in enumerate(buckets.groups):
        bucket_of[members] = g
    return {"ucb_all": np.asarray(ucb_all, dtype=np.float64), "mean_all": np.asarray(X @ theta, dtype=np.float64),
            "bucket_of": bucket_of,
            "champions": champions.copy(), "chosen_bucket": int(buckets.last_bucket)}
