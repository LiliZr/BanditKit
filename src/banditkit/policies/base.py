"""Base class of all policies: declared hyperparameters and the reset / predict / update cycle."""
from __future__ import annotations

import difflib
import inspect
from typing import Any, ClassVar, Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np

from .._utils import make_rng
from .._validation import ParamSpec, check_weights
from ..spec import EnvSpec, Observation

__all__ = ["Policy"]

_FAMILY_TEXT = {"mab": "classical or multi-objective MAB", "linear": "linear", "multi_user": "multi-user linear"}


class Policy:
    """Base class of bandit algorithms.

    Hyperparameters are keyword arguments of the constructor. Each subclass
    declares them in its ``params`` table (type, bounds, default, one-line doc);
    they are validated at construction, and the ``Parameters`` section of the
    class docstring is generated from the table. Checks that depend on the
    problem (``n_components < n_features``, horizon needed...) run in :meth:`reset`.

    Life cycle, with the method names of Gymnasium / Stable-Baselines3::

        policy.reset(spec, weights=None, seed=0)   # once per run
        action = policy.predict(obs)               # every step
        policy.update(obs, action, reward)         # every step

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
    #: Reported by :meth:`parameter_notes` when left at the default.
    arbitrary_defaults: ClassVar[Dict[str, str]] = {}

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

    def get_params(self) -> Dict[str, Any]:
        """Hyperparameters as a dict, as in scikit-learn."""
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
        self._check_spec(spec)
        self._reset()
        specs = self._all_params()
        for key, reason in self._all_arbitrary().items():
            if key in self._params and self._params[key] == specs[key].default:
                self._note("arbitrary", f"{key}={self._params[key]!r} ({reason})")
        self._ready = True

    @classmethod
    def _all_arbitrary(cls) -> Dict[str, str]:
        merged: Dict[str, str] = {}
        for klass in reversed(cls.__mro__):
            merged.update(vars(klass).get("arbitrary_defaults", {}))
        return merged

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

    def predict(self, obs: Observation) -> int:
        """Action to play at this step."""
        if not self._ready:
            raise RuntimeError(f"Call {type(self).__name__}.reset(spec) before predict().")
        return self._predict(obs)

    def update(self, obs: Observation, action: int, reward: Any) -> None:
        """Learn from the reward vector observed after playing ``action`` at ``obs``."""
        if not self._ready:
            raise RuntimeError(f"Call {type(self).__name__}.reset(spec) before update().")
        reward = np.asarray(reward, dtype=np.float64).reshape(-1)
        if reward.size != self.spec.reward_dim:
            raise ValueError(f"The reward must have {self.spec.reward_dim} component(s), got shape {reward.shape}.")
        self._update(obs, int(action), reward, float(self.weights @ reward))

    def partial_fit(self, observations: Iterable[Observation], actions: Iterable[int], rewards: Iterable[Any]) -> "Policy":
        """Learn from logged data (warm start), one :meth:`update` per row, as scikit-learn's partial_fit."""
        for obs, action, reward in zip(observations, actions, rewards, strict=True):
            self.update(obs, action, reward)
        return self

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
