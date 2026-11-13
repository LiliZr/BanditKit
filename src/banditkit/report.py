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

__all__ = ["summarize", "save_run", "load_summary", "find_runs", "build_dashboard", "SUMMARY_FILE",
           "METRIC_TITLES"]

SUMMARY_FILE = "summary.json"

#: Metrics shown by the dashboard, in this order, with their axis titles.
METRIC_TITLES = {"regret": "Cumulative regret", "pseudo_regret": "Cumulative pseudo-regret",
                 "reward": "Cumulative reward", "pareto_regret": "Cumulative Pareto regret",
                 "pareto_share": "Share of Pareto-optimal pulls", "unfairness": "Unfairness",
                 "cpu_time": "CPU time (s)", "wall_time": "Wall time (s)", "memory_mb": "Peak memory (MB)",
                 "n_user_clusters": "Number of user clusters", "n_buckets": "Number of buckets"}
#: Group of each metric in the dashboard: performance, multi-objective, cost or policy state.
METRIC_GROUPS = {"regret": "performance", "pseudo_regret": "performance", "reward": "performance",
                 "pareto_regret": "multi-objective", "pareto_share": "multi-objective",
                 "unfairness": "multi-objective", "cpu_time": "cost", "wall_time": "cost", "memory_mb": "cost"}
_HIGHER_IS_BETTER = {"reward", "pareto_share"}
#: Curves that are not metrics (per-action arrays, cumulative reward vectors).
_NOT_SCALAR = {"reward_vec"}
_DASH = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}
_SYMBOL = {"o": "circle", "s": "square", "^": "triangle-up", "D": "diamond", "v": "triangle-down", "P": "cross",
           "X": "x", "*": "star", "h": "hexagon", "<": "triangle-left", ">": "triangle-right"}


def _round(values: np.ndarray) -> List[Optional[float]]:
    """JSON-friendly floats (6 significant digits, NaN as null)."""
    return [None if not math.isfinite(v) else float(f"{v:.6g}") for v in np.asarray(values, dtype=float).ravel()]


def _stats(values: np.ndarray) -> Dict[str, Any]:
    """Mean over axis 0, its 95% interval (normal approximation), standard deviation and count, ignoring NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=0)
        n = np.sum(~np.isnan(values), axis=0)
        sd = np.nanstd(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
    sd = np.where(n > 1, sd, 0.0)
    half = 1.96 * sd / np.sqrt(np.maximum(n, 1))
    counts = np.atleast_1d(n).astype(int)
    return {"mean": _round(mean), "low": _round(mean - half), "high": _round(mean + half), "sd": _round(sd),
            "n": int(counts[0]) if np.all(counts == counts[0]) else counts.tolist()}


def _metric_title(name: str) -> str:
    if name in METRIC_TITLES:
        return METRIC_TITLES[name]
    if name.startswith("env_"):
        return name[4:].replace("_", " ").upper() if len(name) <= 8 else name[4:].replace("_", " ").capitalize()
    return name.replace("_", " ").capitalize()


def _metric_group(name: str) -> str:
    if name in METRIC_GROUPS:
        return METRIC_GROUPS[name]
    return "performance" if name.startswith("env_") else "policy state"


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
    labels = results.labels
    names = [m for m in METRIC_TITLES if all(m in results.curves[lb] for lb in labels)]
    names += sorted({m for lb in labels for m in results.curves[lb]} - set(names) - _NOT_SCALAR)
    metrics = []
    for m in names:
        if not all(m in results.curves[lb] and np.ndim(results.curves[lb][m]) == 2 for lb in labels):
            continue  # per-action diagnostics, or a curve some policies do not have
        values = np.concatenate([results.curve(lb, m).ravel() for lb in labels])
        values = values[np.isfinite(values)]
        if values.size == 0 or (_metric_group(m) == "policy state" and np.all(values == values[0])):
            continue  # nothing measured, or a policy state that never changes (no buckets...)
        metrics.append(m)
    curves: Dict[str, Dict[str, Any]] = {}
    finals: Dict[str, Dict[str, Any]] = {}
    for metric in metrics:
        curves[metric], finals[metric] = {}, {}
        last = results.final(metric)
        for label in labels:
            curves[metric][label] = _stats(results.curve(label, metric)[:, keep])
            stats = _stats(last[label][:, None])
            finals[metric][label] = {k: (v[0] if isinstance(v, list) else v) for k, v in stats.items()}
            finals[metric][label]["values"] = _round(last[label])
    notes = meta.get("parameter_notes", {})
    search = meta.get("param_search") or {}
    policies = []
    for label in labels:
        look = styles[label]
        info = meta.get("policies", {}).get(label, {})
        policies.append({
            "label": label, "class": info.get("class", label), "params": _changed(info),
            "notes": notes.get(label, []), "best_params": (search.get(label) or {}).get("best_params"),
            "color": look["color"], "dash": _DASH.get(look["linestyle"], "solid"),
            "symbol": _SYMBOL.get(look["marker"], "circle") + ("" if look["filled"] else "-open"),
        })
    searched = {}
    for label, info in search.items():
        if info and info.get("searched") and info.get("table"):
            searched[label] = {"metric": info.get("metric", "regret"), "search_runs": info.get("search_runs"),
                               "best": info.get("best_params", {}),
                               "rows": [{"params": row["params"], "mean": _round([row["mean"]])[0],
                                         "std": _round([row.get("std", float("nan"))])[0]} for row in info["table"]]}
    spec = meta.get("spec", {})
    return json.loads(json.dumps({
        "schema": 2,
        "name": name,
        "description": description,
        "command": command,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "banditkit_version": meta.get("banditkit_version", __version__),
        "problem": {"family": meta.get("family"), "horizon": meta.get("horizon"), "n_runs": results.n_runs,
                    "n_instances": meta.get("n_instances", 1), "seed": meta.get("seed"),
                    "log_interval": meta.get("log_interval"), "track": meta.get("track", []),
                    "measure_memory": meta.get("measure_memory", False),
                    **{k: spec.get(k) for k in ("n_actions", "n_features", "reward_dim", "n_users")},
                    **({"n_actions": "{} to {}".format(*meta["n_actions_range"])}
                       if meta.get("n_actions_range") and len(set(meta["n_actions_range"])) > 1 else {})},
        "policies": policies,
        "steps": _round(steps[keep]),
        "metric_titles": {m: _metric_title(m) for m in metrics},
        "metric_groups": {m: _metric_group(m) for m in metrics},
        "higher_is_better": [m for m in metrics if m in _HIGHER_IS_BETTER or m.startswith("env_")],
        "curves": curves,
        "final": finals,
        "search": searched,
        "notices": meta.get("notices") or [],
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

