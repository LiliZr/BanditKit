"""Visual encoding shared by every figure.

* color  = algorithm family (UCB1, LinUCB, CLUB...), fixed order of appearance;
* marker = combination of techniques (none, RP, AB, RP+AB, ...);
* fill   = filled for the plain algorithm, hollow for a variant with techniques.

Colors come from a categorical palette validated for color-vision deficiencies
(adjacent pairs). Past 8 families, colors restart with a different line style,
so identity never relies on color alone.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]
LINESTYLES = ["-", "--", ":", "-."]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"


def encode(results, overrides: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    """Style of each policy of ``results``: color, marker, filled, linestyle.

    ``overrides`` maps a label to style keys to replace (e.g. ``{"LinUCB": {"color": "k"}}``).
    """
    styles_meta = results.meta.get("styles", {})
    families: Dict[str, int] = {}
    combos: Dict[tuple, int] = {(): 0}
    seen: Dict[tuple, int] = {}  # policies sharing family and techniques (BernoulliTS / GaussianTS)
    out: Dict[str, Dict[str, Any]] = {}
    for label in results.labels:
        info = styles_meta.get(label, {})
        fam = info.get("base_name") or label
        tech = tuple(info.get("techniques", ()))
        fi = families.setdefault(fam, len(families))
        mi = combos.setdefault(tech, len(combos))
        twin = seen.get((fam, tech), 0)  # 0 for the first one, then 1, 2... get another line style
        seen[(fam, tech)] = twin + 1
        out[label] = {
            "color": PALETTE[fi % len(PALETTE)],
            "linestyle": LINESTYLES[(fi // len(PALETTE) + twin) % len(LINESTYLES)],
            "marker": MARKERS[mi % len(MARKERS)],
            "filled": not tech,
        }
        if overrides and label in overrides:
            out[label].update(overrides[label])
    return out


def rc_params() -> Dict[str, Any]:
    """Quiet matplotlib defaults: hairline grid, no top/right spines, readable text."""
    return {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_SECONDARY,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,  # editable text in PDF
        "ps.fonttype": 42,
    }
