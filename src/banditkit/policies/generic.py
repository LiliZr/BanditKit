"""Family-agnostic policies, resolved once the problem is known.

``UCB()`` becomes UCB1 on a MAB and LinUCB on a linear problem; likewise for
``ThompsonSampling()`` and ``EpsilonGreedy()``. A parameter that does not apply
to the resolved policy raises a ValueError naming the family it belongs to.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple, Type

from ..spec import EnvSpec, Observation
from .base import _FAMILY_TEXT, Policy
from .linear import LinGreedy, LinTS, LinUCB
from .mab import UCB1
from .mab import EpsilonGreedy as MABEpsilonGreedy
from .thompson import BernoulliThompsonSampling, GaussianThompsonSampling

__all__ = ["UCB", "ThompsonSampling", "EpsilonGreedy"]


class GenericPolicy(Policy):
    """Wrapper that builds the concrete policy of the problem family in ``reset``."""

    families = frozenset({"mab", "linear", "multi_user"})
    document_params = False  # accepts the parameters of the concrete class
    #: Concrete class for each family.
    targets: ClassVar[Dict[str, Type[Policy]]] = {}

    def __init__(self, **params: Any) -> None:
        # Parameters are validated by the concrete class in reset().
        self._params = dict(params)
        self.inner: Optional[Policy] = None
        self.spec = None
        self._ready = False

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(self._params)

    def _target(self, spec: EnvSpec) -> Type[Policy]:
        return self.targets[spec.family]

    def reset(self, spec: EnvSpec, weights: Any = None, seed: Any = None) -> None:
        if not isinstance(spec, EnvSpec):
            raise TypeError(f"`spec` must be an EnvSpec, got {type(spec).__name__}.")
        target = self._target(spec)
        accepted = target._all_params()
        for key in self._params:
            if key not in accepted:
                owners = sorted({fam for fam, cls in self.targets.items() if key in cls._all_params()})
                where = " or ".join(_FAMILY_TEXT[f] for f in owners) or "no"
                raise ValueError(f"`{key}` does not apply to {target.__name__} ({_FAMILY_TEXT[spec.family]} problem); "
                                 f"it applies to {where} problems.")
        self.inner = target(**self._params)
        self.inner.reset(spec, weights, seed)
        self.spec, self.weights, self._ready = spec, self.inner.weights, True
        self._start_counts(spec)

    def _predict(self, obs: Observation) -> int:
        return self.inner._predict(obs)

    def _update(self, obs, action, reward, scalar_reward) -> None:
        self.inner._update(obs, action, reward, scalar_reward)

    def diagnostics(self) -> Dict[str, Any]:
        return self.inner.diagnostics() if self.inner is not None else {}

    def last_bucket(self) -> int:
        return self.inner.last_bucket() if self.inner is not None else -1

    def parameter_notes(self):
        return self.inner.parameter_notes() if self.inner is not None else []

    @property
    def techniques(self) -> Tuple[str, ...]:
        return self.inner.techniques if self.inner is not None else ()

    @property
    def base_name(self) -> str:  # type: ignore[override]
        return self.inner.base_name if self.inner is not None else type(self).__name__

    @property
    def label(self) -> str:
        if self._params.get("name"):
            return str(self._params["name"])
        return self.inner.label if self.inner is not None else type(self).__name__

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in self._params.items())})"


class UCB(GenericPolicy):
    """UCB1 on a MAB, LinUCB on a linear or multi-user problem."""

    targets = {"mab": UCB1, "linear": LinUCB, "multi_user": LinUCB}


class ThompsonSampling(GenericPolicy):
    """Beta-Bernoulli TS on a MAB with known reward range, Gaussian TS on other MABs, LinTS on linear problems."""

    targets = {"mab": BernoulliThompsonSampling, "linear": LinTS, "multi_user": LinTS}

    def _target(self, spec: EnvSpec) -> Type[Policy]:
        if spec.family == "mab" and spec.reward_range is None and "reward_range" not in self._params:
            return GaussianThompsonSampling
        return super()._target(spec)


class EpsilonGreedy(GenericPolicy):
    """Epsilon-greedy on the empirical means (MAB) or on the ridge estimate (LinGreedy)."""

    targets = {"mab": MABEpsilonGreedy, "linear": LinGreedy, "multi_user": LinGreedy}
