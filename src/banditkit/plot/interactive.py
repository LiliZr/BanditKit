"""Explanatory figures (plotly): how a policy decides, step after step.

Every function takes a :class:`~banditkit.trace.Trace` (one detailed run, from
``bk.trace(env, policy, horizon)``) and returns a ``plotly.graph_objects.Figure``.
Save one with ``fig.write_html(path)`` (interactive) or ``fig.write_image(path)``
(static, needs the kaleido package). Figures with a slider also have a Play button.
"""
from __future__ import annotations

from math import lgamma
from typing import Optional, Sequence

import numpy as np

from .._utils import pareto_front
from .style import GRID, INK, INK_SECONDARY, PALETTE

__all__ = ["allocation", "estimates", "why_this_arm", "beliefs", "prob_best",
           "regret_by_arm", "ellipsoid", "scalarization", "ucbmin_choice", "clusters", "bucket_choice"]

OTHER = "#b4b2aa"   # color of folded actions
MAX_SHOWN = 7       # actions drawn individually; the others are folded into "Other"


# ------------------------------------------------------------------- helpers
def _go():
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Interactive figures need plotly: pip install plotly") from None
    return go


def _style(fig, title: str, x_title: str = "", y_title: str = "", height: int = 420):
    """Shared layout: left-aligned title, light grid, legend below the plot (above it in animated figures)."""
    animated = bool(fig.layout.sliders)
    legend = dict(orientation="h", x=0, y=1.0, yanchor="bottom", itemsizing="constant") if animated \
        else dict(orientation="h", y=-0.18, itemsizing="constant")
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=15, color=INK)), template="plotly_white",
                      height=height, margin=dict(l=60, r=20, t=90 if animated else 60, b=130 if animated else 50),
                      legend=legend,
                      font=dict(family="Inter, system-ui, sans-serif", size=12, color=INK_SECONDARY))
    if not animated:
        fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title=x_title, gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(title=y_title, gridcolor=GRID, linecolor=GRID, zeroline=False)
    return fig


SYMBOLS = ("circle", "square", "diamond", "triangle-up", "triangle-down", "cross", "x", "star", "pentagon", "hexagon")


def _marks(x, i: int, n: int, size: int = 8, count: int = 10) -> dict:
    """Markers of the i-th of n lines: about ``count`` per line, evenly spread, shifted from line to line."""
    x = np.asarray(x, dtype=float)
    sizes = np.zeros(x.size)
    if x.size:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        step = (hi - lo) / count if hi > lo else 1.0
        targets = lo + step * (np.arange(count) + (i % max(n, 1) + 0.5) / max(n, 1))
        sizes[np.unique(np.abs(x[None, :] - targets[:, None]).argmin(axis=1))] = size
    return dict(size=sizes, symbol=SYMBOLS[i % len(SYMBOLS)])


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _most_played(trace, k: int = MAX_SHOWN) -> np.ndarray:
    """The k most played actions, in index order."""
    counts = np.bincount(trace.actions, minlength=trace.n_actions)
    return np.sort(np.argsort(-counts, kind="stable")[:k])


def _arm_name(trace, arm: int) -> str:
    """Name of an action: its real name when the environment gives one, else "Arm i"."""
    names = getattr(trace, "action_names", None)
    return names[arm] if names else f"Arm {arm}"


def _color_order(trace) -> np.ndarray:
    """Arms that get a color: the best by true mean when known (same colors in every run), else the most played."""
    truth = _true_means(trace)
    if truth is not None:
        return np.argsort(-truth, kind="stable")[:MAX_SHOWN]
    return _most_played(trace, MAX_SHOWN)


def _arm_color(trace, arm: int) -> str:
    """Fixed color of an arm across all figures and runs (gray outside the MAX_SHOWN colored arms)."""
    shown = list(_color_order(trace))
    return PALETTE[shown.index(arm)] if arm in shown else OTHER


def _as_list(traces) -> list:
    """One trace or several runs of a policy, as a non-empty list."""
    traces = list(traces) if isinstance(traces, (list, tuple)) else [traces]
    if not traces:
        raise ValueError("No trace given.")
    return traces


def _runs_suffix(traces) -> str:
    return f" ({len(traces)} runs)" if len(traces) > 1 else ""


def _require(trace, *keys: str) -> None:
    missing = [k for k in keys if k not in trace.snapshots]
    if missing:
        raise ValueError(f"{trace.label} does not record {missing}; this figure needs a policy that does.")


def _true_means(trace):
    """Scalarized true mean of each action, or None if the problem has no fixed means."""
    return None if trace.mean_rewards is None else trace.mean_rewards @ trace.weights


def _frame_indices(trace, max_frames: int) -> np.ndarray:
    """Up to ``max_frames`` snapshot indices spread over the run."""
    n = trace.snapshot_steps.size
    return np.unique(np.linspace(0, n - 1, min(n, max_frames)).round().astype(int))


def _animate(go, trace, frames_of, max_frames: int, layout_of=None):
    """Figure whose data at snapshot i is ``frames_of(i)``, with a step slider and Play / Pause buttons.

    ``layout_of(i)``, if given, returns the layout changes of frame i (axis ranges, annotations).
    """
    idx = _frame_indices(trace, max_frames)
    steps = [int(trace.snapshot_steps[i]) for i in idx]
    frames = [go.Frame(data=frames_of(i), name=str(s), layout=layout_of(i) if layout_of else None)
              for i, s in zip(idx, steps)]
    fig = go.Figure(data=frames_of(idx[-1]), frames=frames)
    if layout_of:
        fig.update_layout(layout_of(idx[-1]))
    fig.update_layout(
        hovermode="closest",
        sliders=[dict(active=len(steps) - 1, currentvalue=dict(prefix="Round "), x=0.16, len=0.84, y=-0.16,
                      yanchor="top", pad=dict(t=0), steps=[
            dict(method="animate", label=str(s), args=[[str(s)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
            for s in steps])],
        updatemenus=[dict(type="buttons", direction="left", showactive=False, x=0, y=-0.16, xanchor="left",
                          yanchor="top", buttons=[
            dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True)]),
            dict(label="Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])])
    return fig, idx


# ------------------------------------------------------------------- figures
def allocation(traces, n_windows: int = 40):
    """Share of pulls of each arm per time window, as stacked areas (any MAB policy).

    The usual "traffic allocation" view of bandit A/B tests: the best arm should
    take a growing share. Give several traces (runs of one policy) to average them.
    ``n_windows`` sets the number of time windows.
    """
    go = _go()
    traces = _as_list(traces)
    trace = traces[0]
    edges = np.linspace(0, trace.horizon, n_windows + 1).astype(int)
    centers = (edges[:-1] + edges[1:]) / 2
    shares = np.mean([[np.bincount(tr.actions[a:b], minlength=tr.n_actions) / max(b - a, 1)
                       for a, b in zip(edges[:-1], edges[1:])] for tr in traces], axis=0)
    shown = np.sort(_color_order(trace))
    fig = go.Figure()
    for arm in shown:
        fig.add_scatter(x=centers, y=shares[:, arm], name=_arm_name(trace, arm), stackgroup="share", mode="lines",
                        line=dict(width=0.5, color=_arm_color(trace, arm)))
    others = np.setdiff1d(np.arange(trace.n_actions), shown)
    if others.size:
        fig.add_scatter(x=centers, y=shares[:, others].sum(axis=1), name=f"Other ({others.size} arms)",
                        stackgroup="share", mode="lines", line=dict(width=0.5, color=OTHER))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return _style(fig, f"{trace.label} · Allocation{_runs_suffix(traces)}", "Round", "Share of pulls")


def estimates(trace, show_bonus: bool = False):
    """Estimated mean of each arm over time; true means dotted.

    ``show_bonus`` adds a band up to mean + bonus (index policies: UCB1, UCB-V,
    UCB-min...). Trace with ``every`` > 1 for long runs.
    """
    go = _go()
    _require(trace, "mean")
    steps, mean = trace.snapshot_steps, trace.snapshots["mean"]
    bonus, truth = trace.snapshots.get("bonus") if show_bonus else None, _true_means(trace)
    fig = go.Figure()
    shown = _most_played(trace, 6)
    for i, arm in enumerate(shown):
        color = _arm_color(trace, arm)
        if bonus is not None:
            upper = np.where(np.isfinite(bonus[:, arm]), mean[:, arm] + bonus[:, arm], np.nan)
            fig.add_scatter(x=steps, y=upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=steps, y=mean[:, arm], mode="lines", line=dict(width=0), fill="tonexty",
                            fillcolor=_rgba(color, 0.15), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=steps, y=mean[:, arm], mode="lines+markers", name=_arm_name(trace, arm),
                        line=dict(color=color, width=2), marker=dict(color=color, **_marks(steps, i, len(shown))))
        if truth is not None:
            fig.add_hline(y=float(truth[arm]), line=dict(color=color, width=1, dash="dot"))
    # Legend keys for the two encodings shared by all arms.
    if truth is not None:
        fig.add_scatter(x=[None], y=[None], mode="lines", name="True mean", line=dict(color=INK_SECONDARY, dash="dot"))
    if bonus is not None:
        fig.add_scatter(x=[None], y=[None], mode="lines", name="Up to mean + bonus",
                        line=dict(width=8, color=_rgba(INK_SECONDARY, 0.2)))
    # Y range from the last 95% of the run: the first bonuses are huge and would flatten the plot.
    late = steps >= 0.05 * steps[-1]
    top = mean[late][:, shown] + (np.nan_to_num(bonus[late][:, shown], posinf=0.0) if bonus is not None else 0.0)
    fig.update_yaxes(range=[min(0.0, float(np.nanmin(mean[late][:, shown]))), float(np.nanmax(top)) * 1.05])
    return _style(fig, f"{trace.label} · Estimated means", "Round", "Mean")


def why_this_arm(trace, max_frames: int = 60, max_arms: int = 20):
    """At each step, estimated mean + exploration bonus of each arm; the played arm is orange.

    Shows that an index policy plays the highest index, and how exploration
    (the bonus) shrinks over time. Needs an index policy. ``max_frames`` steps
    are shown on the slider; ``max_arms`` most played arms are drawn.
    """
    go = _go()
    _require(trace, "mean", "bonus")
    mean, bonus, truth = trace.snapshots["mean"], trace.snapshots["bonus"], _true_means(trace)
    arms = _most_played(trace, max_arms)
    names = [_arm_name(trace, a) for a in arms]

    def bars(i):
        played = int(trace.actions[int(trace.snapshot_steps[i]) - 1])
        color = [PALETTE[1] if a == played else PALETTE[0] for a in arms]
        data = [go.Bar(x=names, y=mean[i, arms], showlegend=False, marker=dict(color=color)),
                go.Bar(x=names, y=np.where(np.isfinite(bonus[i, arms]), bonus[i, arms], np.nan), showlegend=False,
                       marker=dict(color=color, opacity=0.35))]
        if truth is not None:
            data.append(go.Scatter(x=names, y=truth[arms], mode="markers", name="True mean",
                                   marker=dict(symbol="line-ew", size=22, line=dict(width=2, color=INK))))
        return data

    fig, idx = _animate(go, trace, bars, max_frames)
    for name, color in (("Mean", PALETTE[0]), ("Bonus", _rgba(PALETTE[0], 0.35)), ("Played arm", PALETTE[1])):
        fig.add_bar(x=[None], y=[None], name=name, marker_color=color)  # legend keys only
    shown_index = mean[idx][:, arms] + np.where(np.isfinite(bonus[idx][:, arms]), bonus[idx][:, arms], np.nan)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(range=[min(0.0, float(np.nanmin(mean[idx][:, arms]))), float(np.nanmax(shown_index)) * 1.05])
    return _style(fig, f"{trace.label} · Index = mean + bonus", "", "Index", height=480)


def beliefs(trace, n_panels: int = 6, grid_size: int = 300):
    """Posterior densities of Thompson Sampling at ``n_panels`` steps (small multiples)."""
    _go()  # clear error message if plotly is missing
    from plotly.subplots import make_subplots

    beta_posterior = "posterior_alpha" in trace.snapshots
    if not beta_posterior:
        _require(trace, "posterior_mean", "posterior_std")
    idx = _frame_indices(trace, n_panels)
    arms = _most_played(trace, 5)
    fig = make_subplots(rows=1, cols=len(idx), shared_yaxes=True,
                        subplot_titles=[f"Round {int(trace.snapshot_steps[i])}" for i in idx])
    if beta_posterior:
        x = np.linspace(0.001, 0.999, grid_size)
    else:
        mu, sd = trace.snapshots["posterior_mean"][:, arms], trace.snapshots["posterior_std"][:, arms]
        x = np.linspace(float((mu - 3 * sd).min()), float((mu + 3 * sd).max()), grid_size)
    for col, i in enumerate(idx, start=1):
        for arm in arms:
            if beta_posterior:
                a, b = trace.snapshots["posterior_alpha"][i, arm], trace.snapshots["posterior_beta"][i, arm]
                density = np.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1) * np.log(x) + (b - 1) * np.log1p(-x))
            else:
                m, s = trace.snapshots["posterior_mean"][i, arm], trace.snapshots["posterior_std"][i, arm]
                density = np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
            fig.add_scatter(x=x, y=density, mode="lines", name=_arm_name(trace, arm), legendgroup=str(arm), showlegend=col == 1,
                            line=dict(color=_arm_color(trace, arm), width=2), row=1, col=col)
    _style(fig, f"{trace.label} · Posterior densities", height=360)
    fig.update_layout(hovermode="closest")
    return fig


def prob_best(trace, n_samples: int = 4000, seed: int = 0):
    """Posterior probability that each arm is the best, over time (Thompson Sampling).

    Estimated by drawing ``n_samples`` samples from the posteriors at each snapshot.
    """
    go = _go()
    rng = np.random.default_rng(seed)
    K = trace.n_actions
    probs = np.zeros((trace.snapshot_steps.size, K))
    for i in range(probs.shape[0]):
        if "posterior_alpha" in trace.snapshots:
            draws = rng.beta(trace.snapshots["posterior_alpha"][i], trace.snapshots["posterior_beta"][i], size=(n_samples, K))
        else:
            _require(trace, "posterior_mean", "posterior_std")
            draws = trace.snapshots["posterior_mean"][i] + trace.snapshots["posterior_std"][i] * rng.standard_normal((n_samples, K))
        probs[i] = np.bincount(draws.argmax(axis=1), minlength=K) / n_samples
    fig = go.Figure()
    shown = _most_played(trace)
    for i, arm in enumerate(shown):
        color = _arm_color(trace, arm)
        fig.add_scatter(x=trace.snapshot_steps, y=probs[:, arm], mode="lines+markers", name=_arm_name(trace, arm),
                        line=dict(color=color, width=2), marker=dict(color=color, **_marks(trace.snapshot_steps, i, len(shown))))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    return _style(fig, f"{trace.label} · Probability of being the best arm", "Round", "Probability")


def regret_by_arm(traces, max_arms: int = 20):
    """Where the regret comes from: gap of each arm times its number of pulls (pseudo-regret).

    Give several traces (runs of one policy) to average the pulls.
    """
    go = _go()
    traces = _as_list(traces)
    trace = traces[0]
    truth = _true_means(trace)
    if truth is None:
        raise ValueError("regret_by_arm needs a problem with fixed mean rewards.")
    counts = np.mean([np.bincount(tr.actions, minlength=tr.n_actions) for tr in traces], axis=0)
    gaps = truth.max() - truth
    loss = gaps * counts
    order = [a for a in np.argsort(-loss)[:max_arms] if loss[a] > 0]
    fig = go.Figure(go.Bar(x=[_arm_name(trace, a) for a in order], y=loss[order], marker_color=PALETTE[0],
                           customdata=np.c_[counts[order], gaps[order]],
                           hovertemplate="%{x}<br>regret %{y:.1f}<br>pulls %{customdata[0]:.0f}<br>gap %{customdata[1]:.3f}<extra></extra>"))
    fig.update_layout(hovermode="closest")
    return _style(fig, f"{trace.label} · Regret per arm{_runs_suffix(traces)}", "", "Gap × pulls")


def ellipsoid(trace, max_frames: int = 40):
    """Linear bandit in 2D: the confidence ellipsoid around the estimate shrinks towards theta.

    Needs a shared LinUCB on 2-dimensional features (or ``n_components=2``).
    """
    go = _go()
    _require(trace, "theta", "V_inv", "width_scale")
    theta, V_inv, scale = trace.snapshots["theta"], trace.snapshots["V_inv"], trace.snapshots["width_scale"]
    if theta.shape[1] != 2 or trace.action_features is None:
        raise ValueError("The ellipsoid figure needs a fixed set of 2-dimensional action features.")
    X = trace.action_features
    true_theta = trace.extra.get("theta")
    circle = np.stack([np.cos(np.linspace(0, 2 * np.pi, 120)), np.sin(np.linspace(0, 2 * np.pi, 120))])

    def scene(i):
        eigval, eigvec = np.linalg.eigh(V_inv[i])
        boundary = theta[i][:, None] + scale[i] * (eigvec * np.sqrt(np.clip(eigval, 0, None))) @ circle
        played = int(trace.actions[int(trace.snapshot_steps[i]) - 1])
        data = [go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", name="Arms", marker=dict(color=OTHER, size=7)),
                go.Scatter(x=[X[played, 0]], y=[X[played, 1]], mode="markers", name="Played arm",
                           marker=dict(color=PALETTE[1], size=12, line=dict(width=2, color="white"))),
                go.Scatter(x=boundary[0], y=boundary[1], mode="lines", name="Confidence set", fill="toself",
                           fillcolor=_rgba(PALETTE[0], 0.12), line=dict(color=PALETTE[0], width=2)),
                go.Scatter(x=[theta[i, 0]], y=[theta[i, 1]], mode="markers", name="Estimate",
                           marker=dict(color=PALETTE[0], size=10))]
        if true_theta is not None and true_theta.shape[-1] == 2:
            target = np.atleast_2d(trace.weights @ true_theta)
            data.append(go.Scatter(x=target[:, 0], y=target[:, 1], mode="markers", name="True θ",
                                   marker=dict(symbol="star", color=INK, size=14)))
        return data

    fig, idx = _animate(go, trace, scene, max_frames)
    limit = float(max(np.abs(X).max(), 0 if true_theta is None else np.abs(true_theta).max())) * 1.25
    fig.update_xaxes(range=[-limit, limit])
    fig.update_yaxes(range=[-limit, limit], scaleanchor="x", scaleratio=1)
    return _style(fig, f"{trace.label} · Confidence ellipsoid", "Feature 1", "Feature 2", height=560)


def scalarization(problem, weights=None):
    """Two objectives: which Pareto optimal arms a linear scalarization can ever choose (Drugan and Nowe, 2013, fig. 1).

    Left: the arms at their true means. Filled markers are Pareto optimal; a ring
    marks the Pareto optimal arms that no linear weights select (they lie in a
    concave part of the front). Right: the scalarized mean ``w1 mu_1 + (1 - w1) mu_2``
    of each Pareto optimal arm against the weight w1; a policy with linear weights
    plays the arm on the upper envelope. ``problem``: an environment, a trace or an
    array of means (n_actions, 2). ``weights`` (optional) marks the weights of an experiment.
    """
    _go()  # clear error if plotly is missing
    from plotly.subplots import make_subplots

    if hasattr(problem, "mean_rewards") and callable(problem.mean_rewards):  # an environment
        means = problem.mean_rewards(problem.reset(seed=0)[0])
    elif hasattr(problem, "mean_rewards"):                                  # a trace
        means = problem.mean_rewards
    else:
        means = problem
    means = np.asarray(means, dtype=float)
    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("scalarization needs the mean rewards of exactly two objectives, shape (n_actions, 2).")
    K = means.shape[0]
    front = pareto_front(means)
    w1 = np.linspace(0.0, 1.0, 201)
    values = np.outer(w1, means[:, 0]) + np.outer(1.0 - w1, means[:, 1])  # (n_weights, K)
    supported = np.unique(values[:, front].argmax(axis=1))
    supported = front[supported]
    colors = {a: PALETTE[k % len(PALETTE)] for k, a in enumerate(front)}

    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.12)
    F = means[front][np.argsort(means[front][:, 0])]
    fig.add_scatter(x=F[:, 0], y=F[:, 1], mode="lines", name="Pareto front", hoverinfo="skip",
                    line=dict(color=INK_SECONDARY, width=1.5, shape="vh"), row=1, col=1)
    for a in range(K):
        optimal, reached = a in front, a in supported
        color = colors.get(a, OTHER)
        status = ("Pareto optimal, chosen by some linear weights" if reached else
                  "Pareto optimal, never chosen by linear weights" if optimal else "dominated")
        fig.add_scatter(x=[means[a, 0]], y=[means[a, 1]], mode="markers+text", text=[f"Arm {a}"], showlegend=False,
                        textposition="top center", textfont=dict(size=11, color=INK_SECONDARY),
                        marker=dict(size=15 if optimal else 11, color=color if optimal else "white",
                                    line=dict(width=4 if optimal and not reached else 1.5,
                                              color=INK if optimal and not reached else color)),
                        hovertemplate=f"Arm {a}: {status}<br>objective 1: %{{x:.3f}}<br>objective 2: %{{y:.3f}}<extra></extra>",
                        row=1, col=1)
    fig.add_scatter(x=w1, y=values[:, front].max(axis=1), mode="lines", name="Best linear choice",
                    line=dict(color=_rgba(INK_SECONDARY, 0.25), width=10), hoverinfo="skip", row=1, col=2)
    for a in front:
        fig.add_scatter(x=w1, y=values[:, a], mode="lines", name=f"Arm {a}", showlegend=False,
                        line=dict(color=colors[a], width=2, dash="solid" if a in supported else "dot"),
                        hovertemplate=f"Arm {a}<br>w1 = %{{x:.2f}}<br>scalarized mean %{{y:.3f}}<extra></extra>",
                        row=1, col=2)
        k = int(np.argmax(values[:, a])) if a in supported else len(w1) // 2
        fig.add_annotation(x=w1[k], y=values[k, a], text=f"Arm {a}", showarrow=False, yshift=10,
                           font=dict(size=11, color=colors[a]), row=1, col=2)
    if weights is not None:
        w = float(np.asarray(weights, dtype=float)[0] / np.sum(weights))
        fig.add_vline(x=w, line=dict(color=INK_SECONDARY, width=1, dash="dash"), row=1, col=2)
        fig.add_annotation(x=w, y=1.0, yref="y2 domain", text="w used", showarrow=False,
                           xanchor="left", font=dict(size=11, color=INK_SECONDARY), row=1, col=2)
    for name, marker in (("Pareto optimal", dict(color=INK_SECONDARY)),
                         ("Never chosen by linear weights", dict(color=INK_SECONDARY, line=dict(width=4, color=INK))),
                         ("Dominated", dict(color="white", line=dict(width=1.5, color=INK_SECONDARY)))):
        fig.add_scatter(x=[None], y=[None], mode="markers", name=name, marker=dict(size=12, **marker), row=1, col=1)

    _style(fig, "Linear scalarization", height=500)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="Mean, objective 1", row=1, col=1)
    fig.update_yaxes(title="Mean, objective 2", row=1, col=1)
    fig.update_xaxes(title="Weight w1 (w2 = 1 − w1)", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title="w1 μ1 + w2 μ2", row=1, col=2)
    return fig


def ucbmin_choice(trace, max_arms: int = 30):
    """UCB-min: per arm and traced round, whether the variance bonus or the UCB1 bonus was used."""
    go = _go()
    _require(trace, "variance_bonus_used")
    used = trace.snapshots["variance_bonus_used"].astype(float).T
    arms = _most_played(trace, max_arms)
    light = "#e7f1fb"
    fig = go.Figure(go.Heatmap(
        z=used[arms], x=trace.snapshot_steps, y=[_arm_name(trace, a) for a in arms], zmin=0, zmax=1,
        colorscale=[[0, light], [0.5, light], [0.5, PALETTE[0]], [1, PALETTE[0]]],
        colorbar=dict(tickvals=[0.25, 0.75], ticktext=["UCB1", "Variance"], len=0.35, thickness=14,
                      title=dict(text="Bonus", side="top")),
        hovertemplate="%{y}, round %{x}<extra></extra>"))
    fig.update_layout(hovermode="closest")
    return _style(fig, f"{trace.label} · Bonus used", "Round")


def clusters(traces: Sequence, n_true_groups: int = None):
    """Number of user clusters over time (CLUB, SCLUB, LinUCB with k-means).

    Traces of the same policy are averaged (mean and one standard deviation over
    the runs). ``n_true_groups`` draws the number of groups of the problem, when known.
    """
    go = _go()
    by_label: dict = {}
    for tr in _as_list(traces):
        _require(tr, "n_user_clusters")
        by_label.setdefault(tr.label, []).append(tr)
    fig = go.Figure()
    for j, (label, runs) in enumerate(by_label.items()):
        color = PALETTE[j % len(PALETTE)]
        steps = runs[0].snapshot_steps
        values = np.array([tr.snapshots["n_user_clusters"] for tr in runs], dtype=float)
        mean = values.mean(axis=0)
        if len(runs) > 1:  # band: mean +- one standard deviation over the runs
            std = values.std(axis=0)
            fig.add_scatter(x=np.r_[steps, steps[::-1]], y=np.r_[mean + std, (mean - std)[::-1]], fill="toself",
                            mode="lines", line=dict(width=0), fillcolor=_rgba(color, 0.15), hoverinfo="skip",
                            showlegend=False)
        fig.add_scatter(x=steps, y=mean, mode="lines+markers", name=label + _runs_suffix(runs),
                        line=dict(color=color, width=2), marker=dict(color=color, **_marks(steps, j, len(by_label))))
    if n_true_groups is not None:
        fig.add_hline(y=n_true_groups, line=dict(color=INK_SECONDARY, width=1, dash="dot"),
                      annotation_text="True groups", annotation_position="top left")
    return _style(fig, "User clusters", "Round", "Clusters")


def bucket_choice(trace, max_frames: int = 40):
    """Arm buckets: what is evaluated at each traced round; scales to many arms.

    Left: one column per bucket, ranked by estimated mean at the last rebuild
    (bucket 0 first). The gray bar spans the UCBs of the bucket's arms; the
    diamond is its champion. Step 1 compares the champions and picks a bucket
    (shaded). Right, same UCB scale: the arms of that bucket. Step 2 plays the
    best UCB among them and the champions (orange). The black ring is the arm
    LinUCB without buckets would play; a gray bar above the orange dot hides a
    better UCB that was not evaluated.
    Needs a linear policy with ``n_buckets`` (LinUCB, CBRAP, CLUB, SCLUB).
    """
    go = _go()
    _require(trace, "ucb_all", "mean_all", "bucket_of", "champions", "chosen_bucket")
    ucb, mean = trace.snapshots["ucb_all"], trace.snapshots["mean_all"]
    bucket_of, champions = trace.snapshots["bucket_of"], trace.snapshots["champions"]
    chosen = trace.snapshots["chosen_bucket"].astype(int)
    K, G = ucb.shape[1], champions.shape[1]
    accent, ring = PALETTE[1], dict(size=18, color="rgba(0,0,0,0)", line=dict(width=2, color=INK))
    dot = dict(size=12, color=accent, line=dict(width=1.5, color="white"))

    def scene(i):
        u, b = ucb[i], bucket_of[i]
        lo, hi = np.full(G, np.inf), np.full(G, -np.inf)
        np.minimum.at(lo, b, u)
        np.maximum.at(hi, b, u)
        members = np.flatnonzero(b == chosen[i])
        members = members[np.argsort(-mean[i, members], kind="stable")]  # right panel: by estimated mean
        rank = {a: r + 1 for r, a in enumerate(members.tolist())}
        played = int(trace.actions[int(trace.snapshot_steps[i]) - 1])
        best = int(np.argmax(u))

        def place(arm):
            """(left x, right x) of an arm: in the zoom if it belongs to the chosen bucket, else at its bucket."""
            return ([], [rank[arm]]) if arm in rank else ([float(b[arm])], [])

        (best_l, best_r), (played_l, played_r) = place(best), place(played)
        hover = "arm %{customdata}<br>UCB %{y:.3f}<extra></extra>"
        return [
            go.Scatter(x=np.repeat(np.arange(G, dtype=float), 3), y=np.c_[lo, hi, np.full(G, np.nan)].ravel(),
                       mode="lines", hoverinfo="skip", showlegend=False,
                       line=dict(color=_rgba(INK_SECONDARY, 0.45), width=3)),
            go.Scatter(x=np.arange(G), y=u[champions[i]], mode="markers", showlegend=False, customdata=champions[i],
                       marker=dict(size=8, symbol="diamond", color=INK_SECONDARY), hovertemplate=hover),
            go.Scatter(x=np.arange(1, members.size + 1), y=u[members], mode="markers", showlegend=False,
                       xaxis="x2", customdata=members, marker=dict(size=6, color=PALETTE[0]), hovertemplate=hover),
            go.Scatter(x=best_l, y=[u[best]] * len(best_l), mode="markers", showlegend=False, marker=ring,
                       customdata=[best] * len(best_l), hovertemplate=hover),
            go.Scatter(x=best_r, y=[u[best]] * len(best_r), mode="markers", showlegend=False, marker=ring,
                       xaxis="x2", customdata=[best] * len(best_r), hovertemplate=hover),
            go.Scatter(x=played_l, y=[u[played]] * len(played_l), mode="markers", showlegend=False, marker=dot,
                       customdata=[played] * len(played_l), hovertemplate=hover),
            go.Scatter(x=played_r, y=[u[played]] * len(played_r), mode="markers", showlegend=False, marker=dot,
                       xaxis="x2", customdata=[played] * len(played_r), hovertemplate=hover),
        ]

    def layout_of(i):
        lo, hi = float(ucb[i].min()), float(ucb[i].max())
        pad = 0.06 * (hi - lo) if hi > lo else 0.05 * max(abs(hi), 1.0)
        size = int(np.sum(bucket_of[i] == chosen[i]))
        return dict(yaxis=dict(range=[lo - pad, hi + pad]),
                    xaxis2=dict(range=[0, size + 1], title=f"Bucket {chosen[i]}: arms by estimated mean"),
                    shapes=[dict(type="rect", xref="x", yref="paper", x0=chosen[i] - 0.5, x1=chosen[i] + 0.5,
                                 y0=0, y1=1, line=dict(width=0), layer="below", fillcolor=_rgba(accent, 0.22))])

    fig, idx = _animate(go, trace, scene, max_frames, layout_of=layout_of)
    # Legend keys only: frames update the first seven traces.
    for name, kind in (("UCB range", "bar"), ("Champion", dict(size=8, symbol="diamond", color=INK_SECONDARY)),
                       ("Chosen bucket", dict(size=6, color=PALETTE[0])), ("Played arm", dot),
                       ("LinUCB without buckets", ring)):
        if kind == "bar":
            fig.add_scatter(x=[None], y=[None], mode="lines", name=name, line=dict(color=_rgba(INK_SECONDARY, 0.45), width=3))
        else:
            fig.add_scatter(x=[None], y=[None], mode="markers", name=name, marker=kind)
    per_round = G + int(round(K / G))
    _style(fig, f"{trace.label} · {K:,} arms, {G} buckets: ≈{per_round} UCBs per round instead of {K:,}", height=540)
    fig.update_layout(
        xaxis=dict(domain=[0, 0.66], range=[-1, G], showgrid=False, title="Bucket (by estimated mean)"),
        xaxis2=dict(domain=[0.72, 1], anchor="y", showgrid=False, gridcolor=GRID, linecolor=GRID, zeroline=False,
                    title=f"Bucket {chosen[idx[-1]]}: arms by estimated mean"),
        yaxis=dict(title="UCB"))
    return fig
