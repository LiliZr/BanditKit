"""Results folders and the dashboard.

A results folder holds one experiment::

    results/<name>/
        config.json    the configuration that produced it (rerun: banditkit run <name>/config.json)
        results.npz    every run (Results.load)
        summary.json   means and 95% intervals, read by the dashboard

:func:`build_dashboard` writes one self-contained HTML page from any number of
folders. The page lists them, draws the selected one, and can load more
``summary.json`` files from disk without rebuilding anything.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from . import __version__
from .plot.style import encode
from .results import Results, _to_json

__all__ = ["summarize", "save_run", "load_summary", "find_runs", "build_dashboard", "SUMMARY_FILE"]

SUMMARY_FILE = "summary.json"

#: Metrics shown by the dashboard, in this order, with their axis titles.
METRIC_TITLES = {"regret": "Cumulative regret", "pseudo_regret": "Cumulative pseudo-regret",
                 "reward": "Cumulative reward", "pareto_regret": "Cumulative Pareto regret",
                 "pareto_share": "Pareto-optimal pulls", "unfairness": "Unfairness",
                 "cpu_time": "CPU time (s)", "wall_time": "Wall time (s)", "memory_mb": "Peak memory (MB)"}
_DASH = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}
_SYMBOL = {"o": "circle", "s": "square", "^": "triangle-up", "D": "diamond", "v": "triangle-down", "P": "cross",
           "X": "x", "*": "star", "h": "hexagon", "<": "triangle-left", ">": "triangle-right"}


def _round(values: np.ndarray) -> List[Optional[float]]:
    """JSON-friendly floats (6 significant digits, NaN as null)."""
    return [None if not math.isfinite(v) else float(f"{v:.6g}") for v in np.asarray(values, dtype=float).ravel()]


def _mean_ci(values: np.ndarray):
    """Mean over axis 0 and its 95% interval (normal approximation), ignoring NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=0)
        n = np.sum(~np.isnan(values), axis=0)
        half = 1.96 * np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1)) if values.shape[0] > 1 \
            else np.zeros_like(mean)
    half = np.where(n > 1, half, 0.0)
    return mean, mean - half, mean + half


def _changed(info: Dict[str, Any]) -> Dict[str, Any]:
    """Hyperparameters that differ from the defaults of the policy class (all of them if the class is unknown)."""
    from . import policies as _policies

    params = dict(info.get("params", {}))
    params.pop("name", None)
    cls = getattr(_policies, info.get("class", ""), None)
    if cls is None or not hasattr(cls, "_all_params"):
        return params
    defaults = {k: spec.default for k, spec in cls._all_params().items()}
    return {k: v for k, v in params.items() if k not in defaults or v != defaults[k]}


def summarize(results: Results, name: str, description: str = "", command: Optional[str] = None,
              max_points: int = 200) -> Dict[str, Any]:
    """What the dashboard shows of ``results``: problem, policies, curves (mean and 95% interval), final values.

    Parameters
    ----------
    results : Results
    name : str
        Name of the folder, shown in the dashboard.
    description : str
        One or two sentences on the experiment.
    command : str, optional
        Command line that reruns it.
    max_points : int
        Curves are thinned to at most this many logged steps.
    """
    meta = results.meta
    steps = np.asarray(results.steps)
    keep = np.unique(np.linspace(0, steps.size - 1, min(max_points, steps.size)).round().astype(int))
    styles = encode(results)
    metrics = [m for m in METRIC_TITLES if all(m in results.curves[label] for label in results.labels)]
    metrics = [m for m in metrics if any(np.isfinite(results.curve(lb, m)).any() for lb in results.labels)]
    curves: Dict[str, Dict[str, Any]] = {}
    finals: Dict[str, Dict[str, Any]] = {}
    for metric in metrics:
        curves[metric], finals[metric] = {}, {}
        last = results.final(metric)
        for label in results.labels:
            mean, low, high = _mean_ci(results.curve(label, metric)[:, keep])
            curves[metric][label] = {"mean": _round(mean), "low": _round(low), "high": _round(high)}
            f_mean, f_low, f_high = _mean_ci(last[label][:, None])
            finals[metric][label] = {"mean": _round(f_mean)[0], "low": _round(f_low)[0], "high": _round(f_high)[0],
                                     "values": _round(last[label])}
    notes = meta.get("parameter_notes", {})
    search = meta.get("param_search") or {}
    policies = []
    for label in results.labels:
        look = styles[label]
        info = meta.get("policies", {}).get(label, {})
        policies.append({
            "label": label, "class": info.get("class", label), "params": _changed(info),
            "notes": notes.get(label, []), "best_params": (search.get(label) or {}).get("best_params"),
            "color": look["color"], "dash": _DASH.get(look["linestyle"], "solid"),
            "symbol": _SYMBOL.get(look["marker"], "circle") + ("" if look["filled"] else "-open"),
        })
    spec = meta.get("spec", {})
    return json.loads(json.dumps({
        "schema": 1,
        "name": name,
        "description": description,
        "command": command,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "banditkit_version": meta.get("banditkit_version", __version__),
        "problem": {"family": meta.get("family"), "horizon": meta.get("horizon"), "n_runs": results.n_runs,
                    "n_instances": meta.get("n_instances", 1), "seed": meta.get("seed"),
                    **{k: spec.get(k) for k in ("n_actions", "n_features", "reward_dim", "n_users")},
                    **({"n_actions": "{} to {}".format(*meta["n_actions_range"])}
                       if meta.get("n_actions_range") and len(set(meta["n_actions_range"])) > 1 else {})},
        "policies": policies,
        "steps": _round(steps[keep]),
        "metric_titles": {m: METRIC_TITLES[m] for m in metrics},
        "curves": curves,
        "final": finals,
    }, default=_to_json))


def save_run(results: Results, folder: Union[str, Path], config: Optional[Dict[str, Any]] = None,
             description: str = "", command: Optional[str] = None) -> Path:
    """Write ``results.npz``, ``summary.json`` and (if given) ``config.json`` into ``folder``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    results.save(folder / "results.npz")
    if config is not None:
        (folder / "config.json").write_text(json.dumps(config, indent=2, default=_to_json) + "\n")
    summary = summarize(results, folder.name, description=description, command=command)
    (folder / SUMMARY_FILE).write_text(json.dumps(summary, separators=(",", ":")) + "\n")
    return folder


def load_summary(folder: Union[str, Path]) -> Dict[str, Any]:
    """Summary of a results folder: ``summary.json``, or computed from ``results.npz`` if missing."""
    folder = Path(folder)
    path = folder / SUMMARY_FILE
    if path.exists():
        return json.loads(path.read_text())
    npz = folder / "results.npz" if folder.is_dir() else folder
    if not npz.exists():
        raise FileNotFoundError(f"{folder} has neither {SUMMARY_FILE} nor results.npz.")
    return summarize(Results.load(npz), folder.stem if folder.is_file() else folder.name)


def find_runs(results_dir: Union[str, Path] = "results") -> List[Path]:
    """Results folders under ``results_dir`` (those with summary.json or results.npz), sorted by name."""
    root = Path(results_dir)
    if not root.exists():
        return []
    found = {p.parent for p in root.rglob(SUMMARY_FILE)} | {p.parent for p in root.rglob("results.npz")}
    return sorted(found)


def build_dashboard(folders: Optional[Sequence[Union[str, Path]]] = None, results_dir: Union[str, Path] = "results",
                    out: Union[str, Path] = "dashboard.html", title: str = "banditkit results",
                    offline: bool = False) -> Path:
    """Write the dashboard: one HTML page for the given folders (default: every folder of ``results_dir``).

    Parameters
    ----------
    folders : sequence of paths, optional
        Results folders to show, in this order.
    results_dir : path
        Searched when ``folders`` is None.
    out : path
        HTML file to write. Open it in a browser; no server is needed.
    title : str
        Title of the page.
    offline : bool
        Embed plotly.js (about 4 MB, needs the plotly package) instead of loading it from a CDN.
    """
    paths = [Path(f) for f in folders] if folders else find_runs(results_dir)
    summaries = []
    for path in paths:
        summary = load_summary(path)
        summary.setdefault("folder", str(path))
        summaries.append(summary)
    if offline:
        from plotly.offline import get_plotlyjs

        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    else:
        plotly_tag = f'<script src="{PLOTLY_CDN}"></script>'
    data = json.dumps(summaries, separators=(",", ":")).replace("</", "<\\/")
    page = (Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
            .replace("{{TITLE}}", title).replace("{{PLOTLY}}", plotly_tag).replace("{{DATA}}", data))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


#: plotly.js used by the dashboard when it is not embedded.
PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"

