"""Figures. ``banditkit.plot`` needs matplotlib; the interactive figures need plotly."""
from .publication import (BANDS, curves, dispersion, panel, pareto_panel, performance_profile, search_landscape,
                          sensitivity, tradeoff)
from . import interactive
from .style import encode

__all__ = ["curves", "panel", "pareto_panel", "tradeoff", "sensitivity", "search_landscape", "dispersion",
           "performance_profile", "encode", "BANDS", "interactive"]
