"""Explanatory charts of a results folder: what happens inside the policies, on one traced run.

``banditkit run`` writes them to ``explain.json`` next to the results, and the dashboard
shows them under "Inside the policies". They need plotly (``pip install plotly``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

__all__ = ["EXPLAIN_FILE", "explain", "save_explain"]

EXPLAIN_FILE = "explain.json"

TITLES = {
    "why": "Why this arm",
    "beliefs": "Posterior beliefs",
    "prob_best": "Probability of being the best arm",
    "ucbmin": "Bonus chosen by UCB-min",
    "allocation": "Pulls over time",
    "regret_by_arm": "Regret per arm",
    "scalarization": "Pareto front and scalarization",
    "ellipsoid": "Confidence ellipsoid",
    "buckets": "Arm buckets",
    "clusters": "User clusters over time",
}
CAPTIONS = {
    "why": "At each round, each arm scores its estimated mean plus an exploration bonus. The played arm (orange) "
           "has the highest score. Use the slider or Play.",
    "beliefs": "Posterior density of each arm at a few rounds. Thompson Sampling draws one sample per arm and "
               "plays the largest.",
    "prob_best": "Posterior probability that each arm is the best one. It is the probability that Thompson "
                 "Sampling plays the arm.",
    "ucbmin": "Per arm and traced round: the bonus UCB-min used, the variance-aware one or the UCB1 one.",
    "allocation": "Share of the pulls of each arm per time window. Exploration first, then the best arms take over.",
    "regret_by_arm": "Where the regret comes from: the gap of each arm times its number of pulls (pseudo-regret).",
    "scalarization": "Left: the arms at their true means; filled markers are Pareto optimal. Right: the scalarized "
                     "mean of each Pareto optimal arm against the weight. A linear scalarization only reaches the "
                     "arms of the upper envelope.",
    "ellipsoid": "On a 2-dimensional version of the problem (12 actions). The confidence set around the estimate "
                 "shrinks towards the true parameter as the policy plays. Use the slider or Play.",
    "buckets": "Left: one column per bucket (UCB range in gray, champion as a diamond). Step 1 picks the bucket of "
               "the best champion. Right: the arms of that bucket; step 2 plays the best UCB among them and the "
               "champions (orange). The black ring is the arm LinUCB without buckets would play.",
    "clusters": ("Number of user clusters each policy keeps over time. The dotted line is the true number of "
                 "groups. When a policy keeps far more clusters than that (one model per user), the axis is logarithmic."),
}


def _figure(fig, kind: str, policy: Optional[str]) -> Dict[str, Any]:
    """A plotly figure as plain JSON, without its title (the dashboard shows it) and template (themed there)."""
    fig.update_layout(title=None, template=None, height=None)
    return {"kind": kind, "policy": policy, "title": TITLES[kind], "caption": CAPTIONS[kind],
            "figure": json.loads(fig.to_json())}


def _try(make: Callable[[], Any]):
    """The figure, or None when the policy does not record what it needs."""
    try:
        return make()
    except (KeyError, ValueError, IndexError, TypeError, AttributeError):
        return None


def explain(env, policies: Dict[str, Any], weights=None, horizon: int = 2000, seed: int = 0,
            n_true_groups: Optional[int] = None) -> List[Dict[str, Any]]:
    """Explanatory charts of ``policies`` on ``env``, from one traced run per policy.

    The traced run lasts at most 2000 steps (50,000 on multi-user problems, where clusters form late).

    Returns a list of ``{"kind", "policy", "title", "caption", "figure"}`` (``policy`` is None for a
    chart of the problem). Which charts: MAB policies get the indices of the arms ("why this
    arm", or the posteriors for Thompson Sampling), the allocation of the pulls and the regret
    per arm; UCB-min gets its choice of bonus; a two-objective problem gets its Pareto front;
    linear policies get the confidence ellipsoid on a 2-dimensional version of the problem;
    policies with arm buckets get the bucket selection; multi-user problems get the number of
    user clusters over time.
    """
    from .instances import sample_instances
    from .plot import interactive as ip
    from .trace import trace

    spec = env.spec
    horizon = int(min(horizon, 50_000 if spec.family == "multi_user" else 2000))  # clustering needs the whole run
    every = max(1, horizon // 60)
    out: List[Dict[str, Any]] = []

    def add(kind: str, policy: Optional[str], fig) -> None:
        if fig is not None:
            out.append(_figure(fig, kind, policy))

    traces = {}
    for label, policy in policies.items():
        tr = _try(lambda: trace(env, policy, horizon, seed=seed, every=every, weights=weights))
        if tr is not None:
            tr.label = label
            traces[label] = tr

    if spec.family == "mab":
        if spec.reward_dim == 2:
            add("scalarization", None, _try(lambda: ip.scalarization(env, weights)))
        for label, tr in traces.items():
            if type(policies[label]).__name__ == "Random":
                continue
            add("why", label, _try(lambda: ip.why_this_arm(tr, max_frames=30)))
            add("beliefs", label, _try(lambda: ip.beliefs(tr)))
            add("prob_best", label, _try(lambda: ip.prob_best(tr)))
            add("ucbmin", label, _try(lambda: ip.ucbmin_choice(tr)))
            add("allocation", label, _try(lambda: ip.allocation([tr])))
            add("regret_by_arm", label, _try(lambda: ip.regret_by_arm([tr])))
    else:
        if spec.family == "linear":
            plane = sample_instances("linear", 1, n_actions=12, n_features=2, noise_std=0.1, seed=seed)[0]
            for label, policy in policies.items():
                tr2 = _try(lambda: trace(plane, policy, 300, seed=seed, every=10))
                add("ellipsoid", label, _try(lambda: ip.ellipsoid(tr2, max_frames=30)) if tr2 is not None else None)
        for label, tr in traces.items():
            add("buckets", label, _try(lambda: ip.bucket_choice(tr, max_frames=30)))
        clustered = [tr for tr in traces.values() if "n_user_clusters" in tr.snapshots]
        if spec.family == "multi_user" and clustered:
            add("clusters", None, _try(lambda: ip.clusters(clustered, n_true_groups)))
    return out


def save_explain(config: Dict[str, Any], folder: Union[str, Path], labels: Optional[List[str]] = None) -> Path:
    """Write ``explain.json`` in a results folder, from the configuration of its run.

    ``labels``: the labels of the policies in the results (default: read from ``summary.json``),
    so that the charts carry the same names as the rest of the dashboard.
    """
    from .cli import build_env, build_policies

    folder = Path(folder)
    envs = build_env(config["env"])
    env = envs[0] if isinstance(envs, list) else envs
    built = build_policies(config["policies"])
    summary_path = folder / "summary.json"
    saved = json.loads(summary_path.read_text())["policies"] if summary_path.exists() else []
    if labels is None and saved:
        labels = [p["label"] for p in saved]
    if labels is None or len(labels) != len(built):
        labels = [p.label for p in built]
    policies = dict(zip(labels, built))
    for entry in saved:  # a searched policy is explained with the values the search kept
        if entry.get("best_params") and entry["label"] in policies:
            policies[entry["label"]] = policies[entry["label"]].clone().set_params(**entry["best_params"])
    settings = config.get("experiment", {})
    groups = config["env"].get("n_clusters") if config["env"].get("kind") == "multi_user" else None
    items = explain(env, policies, weights=settings.get("weights"), horizon=settings.get("horizon", 2000),
                    seed=settings.get("seed") or 0, n_true_groups=groups)
    path = folder / EXPLAIN_FILE
    path.write_text(json.dumps(items, separators=(",", ":")))
    return path
