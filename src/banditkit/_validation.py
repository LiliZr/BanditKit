"""Strict validation helpers shared by environments, policies and experiments.

Every public entry point of banditkit validates its inputs with these helpers,
so that errors surface early with a message naming the parameter, the value
received and what was expected.
"""
from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "check_int",
    "check_float",
    "check_bool",
    "check_choice",
    "check_array",
    "check_weights",
    "ParamSpec",
]


def _fmt(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def check_int(
    name: str,
    value: Any,
    low: Optional[int] = None,
    high: Optional[int] = None,
    allow_none: bool = False,
) -> Optional[int]:
    """Return ``value`` as a Python int, checking type and bounds (inclusive)."""
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"`{name}` is required, got None.")
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"`{name}` must be an integer, got {_fmt(value)} ({type(value).__name__}).")
    value = int(value)
    if low is not None and value < low:
        raise ValueError(f"`{name}` must be >= {low}, got {value}.")
    if high is not None and value > high:
        raise ValueError(f"`{name}` must be <= {high}, got {value}.")
    return value


def check_float(
    name: str,
    value: Any,
    low: Optional[float] = None,
    high: Optional[float] = None,
    low_inclusive: bool = True,
    high_inclusive: bool = True,
    allow_none: bool = False,
) -> Optional[float]:
    """Return ``value`` as a finite Python float, checking type and bounds."""
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"`{name}` is required, got None.")
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"`{name}` must be a real number, got {_fmt(value)} ({type(value).__name__}).")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"`{name}` must be finite, got {value}.")
    if low is not None:
        if low_inclusive and value < low:
            raise ValueError(f"`{name}` must be >= {low}, got {value}.")
        if not low_inclusive and value <= low:
            raise ValueError(f"`{name}` must be > {low}, got {value}.")
    if high is not None:
        if high_inclusive and value > high:
            raise ValueError(f"`{name}` must be <= {high}, got {value}.")
        if not high_inclusive and value >= high:
            raise ValueError(f"`{name}` must be < {high}, got {value}.")
    return value


def check_bool(name: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"`{name}` must be a bool, got {_fmt(value)} ({type(value).__name__}).")
    return bool(value)


def check_choice(name: str, value: Any, choices: Iterable[Any], allow_none: bool = False) -> Any:
    choices = tuple(choices)
    if value is None and allow_none:
        return None
    if value not in choices:
        raise ValueError(f"`{name}` must be one of {choices}, got {_fmt(value)}.")
    return value


def check_array(
    name: str,
    value: Any,
    ndim: Optional[int | Sequence[int]] = None,
    shape: Optional[Sequence[Optional[int]]] = None,
    dtype: Any = np.float64,
    finite: bool = True,
    nonneg: bool = False,
    low: Optional[float] = None,
    high: Optional[float] = None,
    allow_none: bool = False,
    min_size: int = 1,
) -> Optional[np.ndarray]:
    """Convert ``value`` to an ndarray and validate it.

    ``shape`` may contain ``None`` for free dimensions, e.g. ``(None, 3)``.
    """
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"`{name}` is required, got None.")
    try:
        arr = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"`{name}` must be convertible to an array of {np.dtype(dtype).name}: {exc}") from None
    if ndim is not None:
        allowed = (ndim,) if isinstance(ndim, int) else tuple(ndim)
        if arr.ndim not in allowed:
            raise ValueError(
                f"`{name}` must have {' or '.join(map(str, allowed))} dimension(s), "
                f"got shape {arr.shape}."
            )
    if shape is not None:
        if arr.ndim != len(shape) or any(s is not None and s != a for s, a in zip(shape, arr.shape)):
            expected = "(" + ", ".join("*" if s is None else str(s) for s in shape) + ")"
            raise ValueError(f"`{name}` must have shape {expected}, got {arr.shape}.")
    if arr.size < min_size:
        raise ValueError(f"`{name}` must contain at least {min_size} element(s), got shape {arr.shape}.")
    if np.issubdtype(arr.dtype, np.number):
        if finite and not np.all(np.isfinite(arr)):
            raise ValueError(f"`{name}` must contain only finite values (no NaN or inf).")
        if nonneg and np.any(arr < 0):
            raise ValueError(f"`{name}` must be non-negative, found minimum {arr.min()}.")
        if low is not None and np.any(arr < low):
            raise ValueError(f"`{name}` values must be >= {low}, found minimum {arr.min()}.")
        if high is not None and np.any(arr > high):
            raise ValueError(f"`{name}` values must be <= {high}, found maximum {arr.max()}.")
    return arr


def check_weights(weights: Any, reward_dim: int) -> np.ndarray:
    """Validate scalarization weights for a reward of dimension ``reward_dim``.

    ``None`` gives uniform weights 1/D. Weights must be finite, non-negative and
    not all zero. They are not renormalized: the scale of the scalarized reward
    is the user's choice.
    """
    if weights is None:
        return np.full(reward_dim, 1.0 / reward_dim)
    w = check_array("weights", weights, ndim=(0, 1), nonneg=True)
    w = np.atleast_1d(w).astype(np.float64)
    if w.shape != (reward_dim,):
        raise ValueError(
            f"`weights` must have length reward_dim={reward_dim}, got shape {w.shape}."
        )
    if not np.any(w > 0):
        raise ValueError("`weights` must contain at least one positive value.")
    return w


@dataclass(frozen=True)
class ParamSpec:
    """Declaration of one policy hyperparameter: type, default, bounds and a one-line doc.

    ``kind`` is ``"int"``, ``"float"``, ``"bool"``, ``"choice"`` or ``"any"``.
    ``low`` / ``high`` are bounds (inclusive unless ``low_inclusive`` / ``high_inclusive`` is False).
    ``special`` lists extra accepted strings (e.g. ``"sqrt"`` for a number of buckets).
    """

    kind: str
    default: Any = None
    low: Optional[float] = None
    high: Optional[float] = None
    low_inclusive: bool = True
    high_inclusive: bool = True
    choices: tuple = ()
    allow_none: bool = False
    special: tuple = ()
    doc: str = ""

    def type_text(self) -> str:
        """Human-readable type and range, used in generated docstrings."""
        if self.kind == "choice":
            text = "{" + ", ".join(repr(c) for c in self.choices) + "}"
        else:
            text = {"int": "int", "float": "float", "bool": "bool", "any": "object"}[self.kind]
            if self.low is not None or self.high is not None:
                lo = "" if self.low is None else f"{self.low:g} {'<=' if self.low_inclusive else '<'} "
                hi = "" if self.high is None else f" {'<=' if self.high_inclusive else '<'} {self.high:g}"
                text += f" ({lo}x{hi})"
        if self.special:
            text += " or " + " or ".join(repr(s) for s in self.special)
        if self.allow_none and self.kind != "any":
            text += " or None"
        return text

    def validate(self, name: str, value: Any) -> Any:
        if value is None and self.allow_none:
            return None
        if isinstance(value, str) and value in self.special:
            return value
        if self.kind == "int":
            return check_int(name, value, low=None if self.low is None else int(np.ceil(self.low)),
                             high=None if self.high is None else int(self.high))
        if self.kind == "float":
            return check_float(name, value, low=self.low, high=self.high,
                               low_inclusive=self.low_inclusive, high_inclusive=self.high_inclusive)
        if self.kind == "bool":
            return check_bool(name, value)
        if self.kind == "choice":
            return check_choice(name, value, self.choices)
        if self.kind == "any":
            return value
        raise RuntimeError(f"Unknown ParamSpec kind {self.kind!r} for `{name}`.")
