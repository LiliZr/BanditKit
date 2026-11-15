"""Base class of all policies: declared hyperparameters and the reset / predict / update cycle."""
from __future__ import annotations

import difflib
import inspect
import warnings
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .._utils import make_rng
from .._validation import ParamSpec, check_weights
from ..spec import EnvSpec, Observation

__all__ = ["Policy", "ParameterWarning"]

_FAMILY_TEXT = {"mab": "classical or multi-objective MAB", "linear": "linear", "multi_user": "multi-user linear"}


class ParameterWarning(UserWarning):
    """A hyperparameter keeps an arbitrary default, or a constant of the theory had to be assumed."""


def _as_matrix(name: str, value: Any) -> np.ndarray:
    """Finite 2-D float array (the same object when it already is one)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"`{name}` must be 2-D (n_rows, n_features), got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` contains NaN or infinite values.")
    return arr


def _as_indices(name: str, value: Any, n: int) -> np.ndarray:
    """1-D integer array with values in [0, n)."""
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be 1-D, got shape {arr.shape}.")
    if arr.size and not np.all(np.mod(arr, 1) == 0):
        raise ValueError(f"`{name}` must hold integers.")
    arr = arr.astype(np.int64)
    if arr.size and (arr.min() < 0 or arr.max() >= n):
        raise ValueError(f"`{name}` must lie in [0, {n}), got values in [{arr.min()}, {arr.max()}].")
    return arr


class Policy:
    """Base class of bandit algorithms.

    Hyperparameters are keyword arguments of the constructor. Each subclass
    declares them in its ``params`` table (type, bounds, default, one-line doc);
    they are validated at construction, and the ``Parameters`` section of the
    class docstring is generated from the table. Checks that depend on the
    problem (``n_components < n_features``, horizon needed...) run in :meth:`reset`.

    Two ways to use a policy.

    In a simulation, with the method names of Gymnasium / Stable-Baselines3::

        policy.reset(spec, weights=None, seed=0)   # once per run
        action = policy.predict(obs)               # every step
        policy.update(obs, action, reward)         # every step

    On data, with the method names of scikit-learn::

        policy.fit(X, y)                 # X: played actions, y: their rewards
        action = policy.predict(A)       # A: candidate actions (nothing for a MAB)
        policy.partial_fit([action], [reward], ...)

    ``reward`` is the reward vector of shape (reward_dim,); policies learn from
    the scalarized reward ``weights @ reward`` (uniform weights by default).
    """

    #: Problem families the policy handles: subset of {"mab", "linear", "multi_user"}.
    families: ClassVar[FrozenSet[str]] = frozenset()
    #: Hyperparameters of this class (merged with those of parent classes).
    params: ClassVar[Dict[str, ParamSpec]] = {
        "name": ParamSpec("any", default=None, allow_none=True, doc="Display name (default: class name plus techniques)."),
    }
    #: Algorithm family shown in figures (same color for a family and its variants).
    base_name: ClassVar[str] = ""
    #: Generate the signature and docstring Parameters from ``params`` (False for generic wrappers).
    document_params: ClassVar[bool] = True
    #: Hyperparameters whose default value is arbitrary (no theoretical value): name -> reason.
    #: Reported by :meth:`parameter_notes` when left at the default. A subclass removes an
    #: inherited entry by mapping it to None.
    arbitrary_defaults: ClassVar[Dict[str, Optional[str]]] = {}
    #: Why this policy has no proven regret bound (None: it has one). See :attr:`missing_guarantee`.
    no_guarantee: ClassVar[Optional[str]] = None

    #: Parameters searched by ``Experiment(param_grid=True)`` when the policy has them.
    default_search_params: ClassVar[Tuple[str, ...]] = ("alpha", "lambda_")

    #: Kinds of notes: value with no theoretical justification, value taken from the
    #: problem, value assumed because the theory needs a quantity the problem does not give.
    NOTE_KINDS = ("arbitrary", "from_problem", "assumed")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Generate the signature and the Parameters section of the docstring from ``params``."""
        super().__init_subclass__(**kwargs)
        if not cls.document_params:
            return
        specs = cls._all_params()
        cls.__signature__ = inspect.Signature(
            [inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=s.default) for k, s in specs.items()])
        lines = ["", "Parameters", "----------"]
        for key, spec in specs.items():
            lines.append(f"{key} : {spec.type_text()}, default={spec.default!r}")
            lines.append(f"    {spec.doc}")
        cls.__doc__ = inspect.cleandoc(cls.__doc__ or "") + "\n" + "\n".join(lines)

    def __init__(self, **kwargs: Any) -> None:
        specs = self._all_params()
        unknown = [k for k in kwargs if k not in specs]
        if unknown:
            hints = []
            for k in unknown:
                close = difflib.get_close_matches(k, specs, n=1)
                hints.append(f"`{k}` is not a parameter of {type(self).__name__}."
                             + (f" Did you mean `{close[0]}`?" if close else ""))
            raise TypeError(" ".join(hints) + f" Valid parameters: {', '.join(sorted(specs))}.")
        self._params = {key: spec.validate(key, kwargs.get(key, spec.default)) for key, spec in specs.items()}
        for key, value in self._params.items():
            setattr(self, key, value)
        self._check_params()
        self.spec: Optional[EnvSpec] = None
        self._ready = False

    # ------------------------------------------------------------- parameters
    @classmethod
    def _all_params(cls) -> Dict[str, ParamSpec]:
        merged: Dict[str, ParamSpec] = {}
        for klass in reversed(cls.__mro__):
            merged.update(vars(klass).get("params", {}))
        return merged

    def _check_params(self) -> None:
        """Checks between hyperparameters (raise ValueError). Override in subclasses."""

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Hyperparameters as a dict, as in scikit-learn (``deep`` is accepted for compatibility)."""
        return dict(self._params)

    def set_params(self, **params: Any) -> "Policy":
        """Change hyperparameters (validated again). Call :meth:`reset` afterwards."""
        self.__init__(**{**self.get_params(), **params})
        return self

    def clone(self) -> "Policy":
        """New, unfitted policy with the same hyperparameters."""
        return type(self)(**self.get_params())

    # ---------------------------------------------------------------- display
    @property
    def techniques(self) -> Tuple[str, ...]:
        """Efficiency techniques in use, e.g. ("RP", "AB"). Used in labels and figure markers."""
        return ()

    @property
    def label(self) -> str:
        """Display name: the ``name`` parameter, or the class name followed by the techniques."""
        if self._params.get("name"):
            return str(self._params["name"])
        return "_".join((type(self).__name__,) + self.techniques)

    def __repr__(self) -> str:
        specs = self._all_params()
        changed = [f"{k}={v!r}" for k, v in self._params.items() if k != "name" and v != specs[k].default]
        return f"{type(self).__name__}({', '.join(changed)})"

    # ------------------------------------------------------------- life cycle
    def reset(self, spec: EnvSpec, weights: Any = None, seed: Any = None) -> None:
        """Prepare a new run.

        Parameters
        ----------
        spec : EnvSpec
            Description of the problem (``env.spec``, with the horizon if known).
        weights : array of shape (reward_dim,), optional
            Scalarization weights (default: uniform 1 / reward_dim).
        seed : int or numpy Generator, optional
            Seed of the policy's own randomness (ties, sampling, projections).
        """
        if not isinstance(spec, EnvSpec):
            raise TypeError(f"`spec` must be an EnvSpec, got {type(spec).__name__}.")
        if spec.family not in self.families:
            handled = ", ".join(_FAMILY_TEXT[f] for f in sorted(self.families))
            raise ValueError(f"{type(self).__name__} handles {handled} problems; "
                             f"the environment is a {_FAMILY_TEXT[spec.family]} problem.")
        self.spec = spec
        self.weights = check_weights(weights, spec.reward_dim)
        self.rng = make_rng(seed)
        self._notes: List[Tuple[str, str]] = []
        self._start_counts(spec)
        self._check_spec(spec)
        self._reset()
        specs, unused = self._all_params(), self._unused_params()
        for key, reason in self._all_arbitrary().items():
            if key in self._params and key not in unused and self._params[key] == specs[key].default:
                self._note("arbitrary", f"{key}={self._params[key]!r} ({reason})")
        self._ready = True

    @classmethod
    def _all_arbitrary(cls) -> Dict[str, str]:
        merged: Dict[str, Optional[str]] = {}
        for klass in reversed(cls.__mro__):
            merged.update(vars(klass).get("arbitrary_defaults", {}))
        return {k: v for k, v in merged.items() if v is not None}

    def _unused_params(self) -> FrozenSet[str]:
        """Hyperparameters the current setting does not use (never reported as arbitrary)."""
        return frozenset()

    @property
    def missing_guarantee(self) -> Optional[str]:
        """Why no regret bound is proven for this policy with its current hyperparameters (None: one is).

        Heuristic options (random projection, arm buckets, k-means users) and policies whose
        paper proves no bound return a short reason, shown in the warning of an experiment.
        """
        return self.no_guarantee

    @property
    def radius_source(self) -> Optional[str]:
        """Where the default confidence width comes from, when not from the policy's own paper.

        For example "the OFUL radius (Abbasi-Yadkori et al., 2011)" for LinUCB. Shown in the warning.
        """
        return None

    def _note(self, kind: str, text: str) -> None:
        """Record how a hyperparameter was set for this run (see :meth:`parameter_notes`)."""
        self._notes.append((kind, text))

    def parameter_notes(self) -> List[Tuple[str, str]]:
        """How the hyperparameters were set at the last :meth:`reset`, as (kind, text) pairs.

        ``kind`` is ``"arbitrary"`` (default with no theoretical value), ``"from_problem"``
        (constant of the theoretical bound read from the environment) or ``"assumed"``
        (constant the theory needs but the environment does not give).
        """
        return list(getattr(self, "_notes", []))

    def _start_counts(self, spec: EnvSpec) -> None:
        """Fitted attributes shared by both uses (scikit-learn names)."""
        self.n_samples_seen_ = 0  # number of updates since the last reset
        if spec.n_features is not None:
            self.n_features_in_ = spec.n_features
        self._candidates: Optional[np.ndarray] = None  # default candidate actions of predict()
        self._checked: Optional[np.ndarray] = None     # last candidate array checked

    def predict(self, obs: Any = None, *, user: Optional[int] = None) -> int:
        """Action to play.

        Parameters
        ----------
        obs : Observation, array of shape (n_candidates, n_features), or None
            The observation of the environment (simulation). Or, on data, the
            features of the candidate actions of a linear problem (default: the
            ``action_features`` given to :meth:`fit`); nothing for a MAB.
        user : int, optional
            User to serve, for a multi-user problem with candidate actions.

        Returns
        -------
        int
            Index of the action: an arm of the MAB, or a row of the candidate actions.
        """
        if not self._ready:
            raise RuntimeError(f"{type(self).__name__} is not ready: call fit(X, y) or reset(spec) before predict().")
        if isinstance(obs, Observation):
            if user is not None:
                raise ValueError("`user` comes with the observation; pass it only with candidate actions.")
        else:
            obs = self._candidate_observation(obs, user)
        return self._predict(obs)

    def update(self, obs: Observation, action: int, reward: Any) -> None:
        """Learn from the reward vector observed after playing ``action`` at ``obs``."""
        if not self._ready:
            raise RuntimeError(f"Call {type(self).__name__}.reset(spec) before update().")
        reward = np.asarray(reward, dtype=np.float64).reshape(-1)
        if reward.size != self.spec.reward_dim:
            raise ValueError(f"The reward must have {self.spec.reward_dim} component(s), got shape {reward.shape}.")
        self._update(obs, int(action), reward, float(self.weights @ reward))
        self.n_samples_seen_ += 1

    # ------------------------------------------------- scikit-learn style use
    def fit(self, X: Any, y: Any, *, users: Any = None, action_features: Any = None, n_actions: Optional[int] = None,
            n_users: Optional[int] = None, horizon: Optional[int] = None, weights: Any = None,
            seed: Any = None) -> "Policy":
        """Forget what was learned, then learn from logged data, as scikit-learn's ``fit``.

        Parameters
        ----------
        X : array
            The played actions. MAB: their indices, shape (n_samples,). Linear
            problem: their features, shape (n_samples, n_features), or their row
            indices in ``action_features``. X can be empty (cold start).
        y : array of shape (n_samples,) or (n_samples, reward_dim)
            Their rewards (reward vectors for a multi-objective problem).
        users : array of shape (n_samples,), optional
            User of each sample (multi-user linear problem).
        action_features : array of shape (n_actions, n_features), optional
            Candidate actions of a linear problem, used by ``predict()`` when it
            gets none. Arm buckets (``n_buckets``) need them.
        n_actions : int, optional
            Number of arms of a MAB (default: largest index in X plus one).
        n_users : int, optional
            Number of users (default: largest index in ``users`` plus one).
        horizon : int, optional
            Total number of steps T, for the policies whose bound uses it (UCB-V2, EUCBV, SCLUB).
        weights : array of shape (reward_dim,), optional
            Scalarization weights of a multi-objective problem (default: uniform).
        seed : int, optional
            Seed of the policy's randomness (ties, sampling, projections).

        Returns
        -------
        self
        """
        X, Y = self._data_arrays(X, y)
        candidates = None if action_features is None else _as_matrix("action_features", action_features)
        user_ids = None if users is None else _as_indices("users", users, np.iinfo(np.int64).max)
        if user_ids is not None and user_ids.size != Y.shape[0]:
            raise ValueError(f"`users` has {user_ids.size} entries for {Y.shape[0]} samples.")
        if X.ndim == 2 or candidates is not None:
            n_features = X.shape[1] if X.ndim == 2 else candidates.shape[1]
            if candidates is not None and candidates.shape[1] != n_features:
                raise ValueError(f"X has {n_features} features but `action_features` has {candidates.shape[1]}.")
            if n_users is None:
                n_users = int(user_ids.max()) + 1 if user_ids is not None and user_ids.size else 1
            n_known = candidates.shape[0] if candidates is not None else max(1, X.shape[0])
            spec = EnvSpec(n_actions=n_known, n_features=n_features, reward_dim=Y.shape[1], n_users=n_users,
                           horizon=horizon)
        else:
            if user_ids is not None:
                raise ValueError("`users` applies to multi-user linear problems; a 1-D X holds the arms of a MAB.")
            if n_actions is None:
                if X.size == 0:
                    raise ValueError("X is empty: give the number of arms with n_actions=.")
                n_actions = int(np.max(X)) + 1
            spec = EnvSpec(n_actions=n_actions, reward_dim=Y.shape[1], horizon=horizon)
        self._check_data_family(spec, candidates)
        self.reset(spec, weights=weights, seed=seed)
        self._candidates = candidates
        flagged = [f"{text} [{kind}]" for kind, text in self.parameter_notes() if kind != "from_problem"]
        if flagged:
            warnings.warn(f"{self.label}: " + "; ".join(flagged) + ". Set these values in the constructor, "
                          "or search them with bk.Experiment(param_grid=True).", ParameterWarning, stacklevel=2)
        return self._learn(X, Y, user_ids)

    def partial_fit(self, X: Any, y: Any, *, users: Any = None, action_features: Any = None, **first_call: Any
                    ) -> "Policy":
        """Learn from more samples without forgetting, as scikit-learn's ``partial_fit``.

        Same X, y and ``users`` as :meth:`fit`. The first call (policy neither
        fitted nor reset) is a :meth:`fit` and takes its other arguments.
        ``action_features`` sets the candidate actions that index-valued X refers to.
        """
        if not self._ready:
            return self.fit(X, y, users=users, action_features=action_features, **first_call)
        if first_call:
            raise TypeError(f"{', '.join(sorted(first_call))}: only the first call (a fit) takes these arguments.")
        X, Y = self._data_arrays(X, y)
        if action_features is not None:
            self._candidates = _as_matrix("action_features", action_features)
        user_ids = None if users is None else _as_indices("users", users, self.spec.n_users)
        return self._learn(X, Y, user_ids)

    @staticmethod
    def _data_arrays(X: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X)
        Y = np.asarray(y, dtype=np.float64)
        Y = Y.reshape(-1, 1) if Y.ndim <= 1 else Y
        if Y.ndim != 2:
            raise ValueError(f"`y` must have shape (n_samples,) or (n_samples, reward_dim), got {np.shape(y)}.")
        if X.ndim not in (1, 2) or X.shape[0] != Y.shape[0]:
            raise ValueError(f"X has shape {X.shape} and y {np.shape(y)}: they need the same number of samples.")
        if not np.all(np.isfinite(Y)):
            raise ValueError("`y` contains NaN or infinite values.")
        return X, Y

    def _check_data_family(self, spec: EnvSpec, candidates: Optional[np.ndarray]) -> None:
        name = type(self).__name__
        if spec.family not in self.families:
            if "mab" in self.families:
                need = "X holds the indices of the played arms, shape (n_samples,)"
            elif "linear" in self.families:
                need = ("X holds the features of the played actions, shape (n_samples, n_features), "
                        "or their indices with action_features=")
            else:
                need = "it needs several users: pass users= (and n_users= if some users have no data yet)"
            handled = ", ".join(_FAMILY_TEXT[f] for f in sorted(self.families))
            raise ValueError(f"{name} handles {handled} problems: {need}.")
        if spec.family != "mab" and self.get_params().get("n_buckets") is not None and candidates is None:
            raise ValueError(f"{name}(n_buckets=...) groups the candidate actions: pass them with action_features=.")

    def _learn(self, X: np.ndarray, Y: np.ndarray, user_ids: Optional[np.ndarray]) -> "Policy":
        """One :meth:`update` per sample."""
        spec = self.spec
        if spec.family == "mab":
            if X.ndim != 1:
                raise ValueError(f"A MAB policy takes the indices of the played arms, shape (n_samples,); got {X.shape}.")
            for action, reward in zip(_as_indices("X", X, spec.n_actions), Y):
                self.update(Observation(t=self.n_samples_seen_), action, reward)
            return self
        if X.ndim == 1:
            if self._candidates is None:
                raise ValueError("A 1-D X holds row indices of the candidate actions: give them with action_features=.")
            actions = _as_indices("X", X, self._candidates.shape[0])
            # The candidate set is the action set of every sample: buckets and caches are built on it.
            samples = ((self._candidates, a) for a in actions)
        else:
            features = _as_matrix("X", X)
            if features.shape[1] != spec.n_features:
                raise ValueError(f"X has {features.shape[1]} features; the policy was fitted with {spec.n_features}.")
            # Each played action alone: nothing (buckets, norms) is built on the whole log.
            samples = ((features[i:i + 1], 0) for i in range(features.shape[0]))
        n = Y.shape[0]
        if user_ids is None:
            if spec.n_users > 1 and n:
                raise ValueError("Several users: give the user of each sample with users=.")
            user_ids = np.zeros(n, dtype=np.int64)
        elif user_ids.size and user_ids.max() >= spec.n_users:
            raise ValueError(f"`users` must lie in [0, {spec.n_users}); fit with n_users= to allow more users.")
        for (action_set, action), user, reward in zip(samples, user_ids, Y):
            self.update(Observation(t=self.n_samples_seen_, user_id=int(user), action_features=action_set),
                        action, reward)
        return self

    def _candidate_observation(self, candidates: Any, user: Optional[int]) -> Observation:
        """Observation built from scikit-learn style arguments of :meth:`predict`."""
        spec, t = self.spec, self.n_samples_seen_
        if spec.family == "mab":
            if candidates is not None or user is not None:
                raise ValueError(f"{type(self).__name__} plays a MAB: call predict() without candidate actions.")
            return Observation(t=t)
        if candidates is None:
            if self._candidates is None:
                raise ValueError("Give the candidate actions: predict(A) with A of shape (n_actions, n_features), "
                                 "or action_features= to fit.")
            candidates = self._candidates
        elif candidates is not self._checked:
            candidates = _as_matrix("candidate actions", candidates)
            if candidates.shape[1] != spec.n_features:
                raise ValueError(f"The candidate actions have {candidates.shape[1]} features; "
                                 f"the policy was fitted with {spec.n_features}.")
            self._checked = candidates
        if spec.n_users > 1:
            if user is None:
                raise ValueError("Several users: say which one with predict(A, user=...).")
            if not 0 <= int(user) < spec.n_users:
                raise ValueError(f"`user` must lie in [0, {spec.n_users}), got {user}.")
        elif user not in (None, 0):
            raise ValueError("`user` applies to multi-user problems (fit with users=).")
        return Observation(t=t, user_id=int(user or 0), action_features=candidates)

    # --------------------------------------------------------- for subclasses
    def _check_spec(self, spec: EnvSpec) -> None:
        """Checks of hyperparameters against the problem (raise ValueError)."""

    def _reset(self) -> None:
        raise NotImplementedError

    def _predict(self, obs: Observation) -> int:
        raise NotImplementedError

    def _update(self, obs: Observation, action: int, reward: np.ndarray, scalar_reward: float) -> None:
        raise NotImplementedError

    def diagnostics(self) -> Dict[str, Any]:
        """Internal state worth recording (estimates, bonuses, number of clusters...)."""
        return {}

    def last_bucket(self) -> int:
        """Bucket of the last predicted action (-1 for policies without buckets)."""
        return -1
