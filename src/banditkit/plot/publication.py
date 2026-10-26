"""Publication figures (matplotlib). Every function returns ``(fig, ax)``; save with ``fig.savefig("fig.pdf")``.

Keyword arguments shared by all figures:

* ``ax``: matplotlib Axes to draw on (default: a new figure of size ``figsize``);
* ``include``: labels of the policies to draw, in this order (default: all);
* ``labels``: legend names, e.g. ``{"LinUCB_RP_AB": "LinUCB + RP + AB"}``;
* ``overrides``: per-policy style, e.g. ``{"LinUCB": {"color": "black", "marker": "o"}}``;
* ``style``: matplotlib or SciencePlots style (``"science"``, ``"ieee"``...; needs
  ``pip install SciencePlots``). None: banditkit's defaults.

Encoding (see :mod:`banditkit.plot.style`): color = algorithm family, marker shape =
combination of techniques, hollow marker = variant with techniques.
"""
from __future__ import annotations

import contextlib
import warnings
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .style import INK_SECONDARY, encode, rc_params

__all__ = ["curves", "panel", "pareto_panel", "tradeoff", "sensitivity", "search_landscape", "dispersion",
           "performance_profile", "BANDS", "AXIS_LABELS"]

#: Axis titles of the metrics.
AXIS_LABELS = {"regret": "Cumulative regret", "pseudo_regret": "Cumulative pseudo-regret",
               "pareto_regret": "Cumulative Pareto regret", "reward": "Cumulative reward",
               "pareto_share": "Pareto-optimal pulls", "unfairness": "Unfairness",
               "cpu_time": "CPU time (s)", "wall_time": "Wall time (s)", "memory_mb": "Peak memory (MB)"}
#: Uncertainty bands: standard deviation, 95% confidence interval of the mean, 10-90% quantiles, none.
BANDS = ("std", "ci95", "quantiles", None)
_SCIENCEPLOTS_STYLES = ("science", "ieee", "nature", "notebook", "no-latex", "grid", "scatter", "high-vis")


# ------------------------------------------------------------------- helpers
@contextlib.contextmanager
def _style_context(style: Optional[str]):
    import matplotlib.pyplot as plt

    if style is None:
        with plt.rc_context(rc_params()):
            yield
        return
    if style in _SCIENCEPLOTS_STYLES:
        try:
            import scienceplots  # noqa: F401  (importing it registers the styles)
        except ImportError:
            raise ImportError(f"style={style!r} needs SciencePlots: pip install SciencePlots") from None
    with plt.style.context(style):
        yield


def _figure(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize, layout="constrained")


def _policies(results, include: Optional[Sequence[str]]) -> list:
    if include is None:
        return results.labels
    unknown = [p for p in include if p not in results.labels]
    if unknown:
        raise KeyError(f"Unknown policies {unknown}. Available: {results.labels}")
    return list(include)


def mean_and_band(values: np.ndarray, band: Optional[str]):
    """Mean over runs (axis 0) and the lower / upper edges of the band (None if no band)."""
    if band not in BANDS:
        raise ValueError(f"`band` must be one of {BANDS}, got {band!r}.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns after a timeout
        mean = np.nanmean(values, axis=0)
        if band is None or values.shape[0] < 2:
            return mean, None, None
        if band == "quantiles":
            return mean, np.nanquantile(values, 0.1, axis=0), np.nanquantile(values, 0.9, axis=0)
        half_width = np.nanstd(values, axis=0, ddof=1)
        if band == "ci95":
            half_width = 1.96 * half_width / np.sqrt(np.sum(~np.isnan(values), axis=0))
        return mean, mean - half_width, mean + half_width


def _final_label(metric: str) -> str:
    """'Final cumulative regret', 'Final CPU time (s)': lowercase the first letter unless it starts an acronym."""
    label = AXIS_LABELS.get(metric, metric)
    if len(label) > 1 and label[1].islower():
        label = label[0].lower() + label[1:]
    return "Final " + label


def _marker(look: Dict[str, Any], size: float = 5.5) -> Dict[str, Any]:
    return dict(marker=look["marker"], markersize=size, markeredgecolor=look["color"], markeredgewidth=1.2,
                markerfacecolor=look["color"] if look["filled"] else "white")


#: Symbols of searched hyperparameters in legend tables.
PARAM_SYMBOLS = {"alpha": "α", "lambda_": "λ", "epsilon": "ε", "tau": "τ"}


def _fmt_param(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.0e}" if value != 0 and (abs(value) < 0.01 or abs(value) >= 1000) else f"{value:g}"
    return str(value)


def _legend_labels(results, names: Sequence[str], labels: Optional[Mapping[str, str]]):
    """Legend labels, and a legend title when a parameter search ran.

    After ``Experiment(param_grid=...)`` the legend becomes a small table: one row
    per policy (color and marker, name, best value of each searched parameter).
    """
    shown = [(labels or {}).get(n, n) for n in names]
    best = results.best_params if hasattr(results, "best_params") else {}
    keys = sorted({k for n in names for k in best.get(n, {})})
    if not keys:
        return shown, None
    cells = [[_fmt_param(best.get(n, {}).get(k)) for k in keys] for n in names]
    heads = [PARAM_SYMBOLS.get(k, k) for k in keys]
    name_w = max(len(x) for x in shown)
    col_w = [max(len(h), *(len(row[c]) for row in cells)) for c, h in enumerate(heads)]
    rows = [name.ljust(name_w) + "".join("  " + v.rjust(w) for v, w in zip(row, col_w)) for name, row in zip(shown, cells)]
    title = "best".ljust(name_w) + "".join("  " + h.rjust(w) for h, w in zip(heads, col_w))
    return rows, title


def _figure_legend(fig, handles, names, title: Optional[str]) -> None:
    """Shared legend of a multi-panel figure: below the panels, or a parameter table on the right."""
    if title:
        mono = {"family": "monospace", "size": 8}
        fig.legend(handles, names, loc="outside right upper", title=title, prop=mono, title_fontproperties=mono,
                   alignment="left")
    else:
        fig.legend(handles, names, loc="outside lower center", ncols=min(len(names), 4))


def _legend(ax, title: Optional[str] = None) -> None:
    """Legend for 2+ entries, moved outside on the right past 4 entries so it never hides data.

    With a ``title`` (parameter table), the legend is set in a monospaced font so columns align.
    """
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) < 2 and title is None:
        return
    kwargs = dict(title=title, prop={"family": "monospace", "size": 8}, title_fontproperties={
        "family": "monospace", "size": 8}, alignment="left") if title else {}
    if len(handles) > 4 or title:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), **kwargs)
    else:
        ax.legend(**kwargs)


# ------------------------------------------------------------------- figures
def curves(results, metric: str = "regret", band: Optional[str] = "std", log_x: bool = False, log_y: bool = False,
           xlabel: str = "Round", title: Optional[str] = None, n_markers: int = 8, ax=None, include=None,
           labels: Optional[Mapping[str, str]] = None, overrides=None, style: Optional[str] = None,
           figsize=(5.0, 3.4)):
    """A metric over time: mean over runs, one line per policy, with an uncertainty band.

    Parameters
    ----------
    metric : str
        Curve to draw (``results.metrics()`` lists them).
    band : {"std", "ci95", "quantiles", None}
        Band around the mean: one standard deviation, 95% confidence interval of
        the mean, 10-90% quantiles, or none.
    log_x, log_y : bool
        Logarithmic axes.
    n_markers : int
        Markers per line; they are staggered between policies so overlapping curves stay distinct.
    """
    looks = encode(results, overrides)
    names = _policies(results, include)
    shown, legend_title = _legend_labels(results, names, labels)
    x = results.steps
    every = max(1, x.size // max(n_markers, 1))
    with _style_context(style):
        fig, ax = _figure(ax, figsize)
        for i, label in enumerate(names):
            look = looks[label]
            mean, low, high = mean_and_band(results.curve(label, metric), band)
            if log_x:  # markers evenly spaced on the log axis, staggered between policies
                shift = (i / len(names)) / max(n_markers, 1)
                targets = np.exp(np.log(x[0]) + (np.arange(n_markers) / max(n_markers, 1) + shift + 0.5 / n_markers)
                                 * (np.log(x[-1]) - np.log(x[0])))
                markevery = sorted(set(np.minimum(np.searchsorted(x, targets), x.size - 1).tolist()))
            else:
                markevery = ((i * every // len(names)) % every, every)
            ax.plot(x, mean, color=look["color"], linestyle=look["linestyle"], markevery=markevery,
                    label=shown[i], **_marker(look))
            if low is not None:
                ax.fill_between(x, low, high, color=look["color"], alpha=0.12, linewidth=0)
        ax.set(xlabel=xlabel, ylabel=AXIS_LABELS.get(metric, metric), xlim=(x[0], x[-1]))
        ax.set_xscale("log" if log_x else "linear")
        ax.set_yscale("log" if log_y else "linear")
        if title:
            ax.set_title(title, loc="left")
        _legend(ax, legend_title)
    return fig, ax


def panel(results, metrics: Sequence[str] = ("regret", "reward", "cpu_time", "memory_mb"), band: Optional[str] = "std",
          include=None, labels=None, overrides=None, style: Optional[str] = None, figsize=(9.5, 6.5)):
    """Grid of :func:`curves` (2 x 2 by default) with one shared legend. Metrics never recorded are skipped."""
    import matplotlib.pyplot as plt

    first = _policies(results, include)[0]
    metrics = [m for m in metrics if m in results.curves[first] and not np.all(np.isnan(results.curve(first, m)))]
    n_cols = 2 if len(metrics) > 1 else 1
    with _style_context(style):
        fig, axes = plt.subplots(int(np.ceil(len(metrics) / n_cols)), n_cols, figsize=figsize, layout="constrained",
                                 squeeze=False)
        for ax, metric in zip(axes.flat, metrics):
            curves(results, metric, band=band, ax=ax, include=include, labels=labels, overrides=overrides, style=style)
            if ax.get_legend():
                ax.get_legend().remove()
        for ax in list(axes.flat)[len(metrics):]:
            ax.set_visible(False)
        handles, names = axes.flat[0].get_legend_handles_labels()
        _figure_legend(fig, handles, names, _legend_labels(results, _policies(results, include), labels)[1])
    return fig, axes


def pareto_panel(results, band: Optional[str] = "std", log_x: bool = True, include=None, labels=None, overrides=None,
                 style: Optional[str] = None, figsize=(11.0, 3.6)):
    """Multi-objective MAB, the three measures of Drugan and Nowe (2013, fig. 3), over time.

    Share of the pulls on Pareto optimal arms (higher is better), cumulative Pareto
    regret (lower is better), and unfairness (spread of the pulls among Pareto
    optimal arms, 0 when they are pulled evenly). Needs
    ``Experiment(track=["pareto_share", "pareto_regret", "unfairness"])``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    metrics = ("pareto_share", "pareto_regret", "unfairness")
    first = _policies(results, include)[0]
    missing = [m for m in metrics if m not in results.curves[first]]
    if missing:
        raise ValueError(f"pareto_panel needs Experiment(track={list(metrics)}); missing {missing}.")
    with _style_context(style):
        fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")
        for ax, metric in zip(axes, metrics):
            curves(results, metric, band=band, log_x=log_x, ax=ax, include=include, labels=labels,
                   overrides=overrides, style=style)
            if ax.get_legend():
                ax.get_legend().remove()
        axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[0].set_ylim(0, 1.02)
        # Unfairness spans orders of magnitude and starts at 0: log axis, values below 1 drawn at the bottom.
        axes[2].set_yscale("log", nonpositive="clip")
        axes[2].set_ylim(bottom=1.0)
        handles, names = axes[0].get_legend_handles_labels()
        _figure_legend(fig, handles, names, _legend_labels(results, _policies(results, include), labels)[1])
    return fig, axes


def search_landscape(results, policy: str, x: str = "alpha", by: Optional[str] = None, log_x: bool = True, ax=None,
                     style: Optional[str] = None, figsize=(5.0, 3.4)):
    """Result of the parameter search of one policy: search metric against ``x``, one line per value of ``by``.

    Needs ``Experiment(param_grid=...)``. Error bars: one standard deviation over
    the search runs. The star marks the combination kept.
    """
    search = (results.meta.get("param_search") or {}).get(policy)
    if not search or not search.get("table"):
        raise ValueError(f"No parameter search for {policy!r}; run the Experiment with `param_grid`.")
    table = search["table"]
    if x not in table[0]["params"]:
        raise ValueError(f"{x!r} was not searched for {policy!r}; searched: {list(table[0]['params'])}.")
    if by is None:
        others = [k for k in table[0]["params"] if k != x]
        by = others[0] if others else None
    groups: Dict[Any, list] = {}
    for row in table:
        groups.setdefault(row["params"].get(by) if by else None, []).append(row)
    best = search["best_params"]
    with _style_context(style):
        fig, ax = _figure(ax, figsize)
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("Blues")
        for k, (value, rows) in enumerate(sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0]))):
            rows = sorted(rows, key=lambda r: r["params"][x])
            xs = [r["params"][x] for r in rows]
            color = cmap(0.35 + 0.6 * k / max(len(groups) - 1, 1))
            ax.errorbar(xs, [r["mean"] for r in rows], yerr=[r["std"] for r in rows], color=color, marker="o",
                        markersize=4, capsize=0, linewidth=1.5,
                        label=f"{PARAM_SYMBOLS.get(by, by)} = {_fmt_param(value)}" if by else None)
        ax.plot([best[x]], [next(r["mean"] for r in table if r["params"] == best)], marker="*", markersize=14,
                color=INK_SECONDARY, linestyle="none", label="Best", zorder=5)
        ax.set(xlabel=PARAM_SYMBOLS.get(x, x), ylabel=f"{_final_label(search.get('metric', 'regret'))} (search)")
        if log_x:
            ax.set_xscale("log")
        _legend(ax)
    return fig, ax


def tradeoff(results, x: str = "cpu_time", y: str = "regret", front: bool = True, log_x: bool = False, ax=None,
             include=None, labels=None, overrides=None, style: Optional[str] = None, figsize=(5.0, 3.6)):
    """Final ``y`` against final ``x`` (mean ± std over runs), one point per policy.

    With ``front=True``, the policies that no other policy beats on both axes
    (lower is better on both) are joined by a line: the compromise front.
    """
    looks = encode(results, overrides)
    names = _policies(results, include)
    final_x, final_y = results.final(x), results.final(y)
    points = []
    with _style_context(style):
        fig, ax = _figure(ax, figsize)
        for label in names:
            mx, my = np.nanmean(final_x[label]), np.nanmean(final_y[label])
            ax.errorbar(mx, my, xerr=np.nanstd(final_x[label]), yerr=np.nanstd(final_y[label]),
                        color=looks[label]["color"], elinewidth=1, capsize=0, linestyle="none",
                        label=(labels or {}).get(label, label), **_marker(looks[label], size=7))
            points.append((mx, my))
        if front and len(points) > 1:
            P = np.array(points)
            non_dominated = [i for i in range(len(P)) if not np.any(np.all(P <= P[i], axis=1) & np.any(P < P[i], axis=1))]
            F = P[sorted(non_dominated, key=lambda i: P[i, 0])]
            ax.plot(F[:, 0], F[:, 1], color=INK_SECONDARY, linewidth=0.9, zorder=0, label="Compromise front")
        ax.set(xlabel=AXIS_LABELS.get(x, x), ylabel=_final_label(y))
        ax.set_xscale("log" if log_x else "linear")
        _legend(ax)
    return fig, ax


def sensitivity(results_by_value: Mapping[Any, Any], param: str, metric: str = "regret", band: Optional[str] = "std",
                log_x: bool = False, ax=None, include=None, labels=None, overrides=None, style: Optional[str] = None,
                figsize=(5.0, 3.4)):
    """Final ``metric`` against the value of a hyperparameter.

    Parameters
    ----------
    results_by_value : dict {value: Results}
        One experiment per tested value of the hyperparameter, with the same policy labels.
    param : str
        Name of the hyperparameter (x-axis title).
    """
    values = list(results_by_value)
    first = results_by_value[values[0]]
    looks = encode(first, overrides)
    with _style_context(style):
        fig, ax = _figure(ax, figsize)
        for label in _policies(first, include):
            finals = np.array([results_by_value[v].final(metric)[label] for v in values])  # (n_values, n_runs)
            mean, low, high = mean_and_band(finals.T, band)
            ax.plot(values, mean, color=looks[label]["color"], linestyle=looks[label]["linestyle"],
                    label=(labels or {}).get(label, label), **_marker(looks[label]))
            if low is not None:
                ax.fill_between(values, low, high, color=looks[label]["color"], alpha=0.12, linewidth=0)
        ax.set(xlabel=param, ylabel=_final_label(metric))
        ax.set_xscale("log" if log_x else "linear")
        _legend(ax)
    return fig, ax


def dispersion(results, metric: str = "regret", kind: str = "box", ax=None, include=None, labels=None, overrides=None,
               style: Optional[str] = None, figsize=(5.0, 3.4)):
    """Spread over runs of the final ``metric``, per policy.

    ``kind``: ``"box"`` (box plots), ``"violin"`` (densities with the runs as dots)
    or ``"strip"`` (one dot per run).
    """
    if kind not in ("box", "violin", "strip"):
        raise ValueError("`kind` must be 'box', 'violin' or 'strip'.")
    looks = encode(results, overrides)
    names = _policies(results, include)
    finals = results.final(metric)
    data = [finals[p][~np.isnan(finals[p])] for p in names]
    positions = np.arange(1, len(names) + 1)
    with _style_context(style):
        fig, ax = _figure(ax, figsize)
        if kind == "box":
            boxes = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                               medianprops=dict(color="white", linewidth=1.5))
            for patch, p in zip(boxes["boxes"], names):
                patch.set(facecolor=looks[p]["color"], edgecolor=looks[p]["color"], alpha=0.85)
        else:
            if kind == "violin":
                violins = ax.violinplot(data, positions=positions, showmedians=True)
                for body, p in zip(violins["bodies"], names):
                    body.set(facecolor=looks[p]["color"], edgecolor=looks[p]["color"], alpha=0.35)
            jitter = np.random.default_rng(0)
            for x, values, p in zip(positions, data, names):
                ax.scatter(x + jitter.uniform(-0.12, 0.12, values.size), values, s=14, color=looks[p]["color"], zorder=3)
        ax.set_xticks(positions, [(labels or {}).get(p, p) for p in names], rotation=30, ha="right")
        ax.set_ylabel(_final_label(metric))
        ax.grid(axis="x", visible=False)
    return fig, ax


def performance_profile(results_by_dataset: Mapping[str, Any], metric: str = "regret", tau_max: Optional[float] = None,
                        ax=None, include=None, labels=None, overrides=None, style: Optional[str] = None,
                        figsize=(5.0, 3.4)):
    """Performance profiles (Dolan and Moré, 2002) of the policies over several datasets.

    For each policy: the fraction of datasets where its mean final ``metric`` is at
    most tau times the best one on that dataset, as tau grows (log axis). Lower
    metrics are better. ``results.by_instance()`` gives one entry per instance.

    Parameters
    ----------
    results_by_dataset : dict {dataset name: Results}
        One experiment per dataset, with the same policy labels.
    tau_max : float, optional
        Right end of the x-axis (default: the largest ratio observed).
    """
    datasets = list(results_by_dataset)
    first = results_by_dataset[datasets[0]]
    looks = encode(first, overrides)
    names = _policies(first, include)
    scores = np.array([[np.nanmean(results_by_dataset[d].final(metric)[p]) for p in names] for d in datasets])
    best = scores.min(axis=1, keepdims=True)
    ratios = scores / np.where(best > 0, best, 1.0)
    taus = np.geomspace(1.0, max(tau_max or float(np.nanmax(ratios)) * 1.05, 1.01), 400)
    with _style_context(style):
        fig, ax = _figure(ax, figsize)
        for j, label in enumerate(names):
            fraction = (ratios[:, j][None, :] <= taus[:, None]).mean(axis=1)
            # Markers at evenly spaced positions on the log axis, staggered between policies.
            marks = np.unique(np.linspace(j * 40 // len(names), taus.size - 1, 6).astype(int))
            ax.step(taus, fraction, where="post", color=looks[label]["color"], linestyle=looks[label]["linestyle"],
                    markevery=list(marks), label=(labels or {}).get(label, label), **_marker(looks[label]))
        ax.set(xlabel="τ (ratio to the best)", ylabel="Share of datasets", ylim=(-0.02, 1.02))
        ax.set_xscale("log")
        _legend(ax)
    return fig, ax
