"""Figures. ``banditkit.plot`` needs matplotlib; the interactive figures need plotly.

Submodules load on first use, so ``banditkit.plot.style`` (the visual encoding,
also used by the dashboard) works without matplotlib.
"""
import importlib
from typing import Any

from .style import encode

_PUBLICATION = ("BANDS", "curves", "dispersion", "panel", "pareto_panel", "performance_profile", "search_landscape",
                "sensitivity", "tradeoff")

__all__ = ["curves", "panel", "pareto_panel", "tradeoff", "sensitivity", "search_landscape", "dispersion",
           "performance_profile", "encode", "BANDS", "interactive"]


def __getattr__(name: str) -> Any:
    if name in _PUBLICATION:
        return getattr(importlib.import_module(".publication", __name__), name)
    if name == "interactive":
        return importlib.import_module(".interactive", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
