"""Outputs of an experiment: curves per policy and per run, summaries, export and persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

__all__ = ["Results", "CORE_METRICS"]

#: Curves recorded for every policy.
CORE_METRICS = ("regret", "reward", "cpu_time", "wall_time", "memory_mb")

_LATEX_NAMES = {"regret": "Regret", "pseudo_regret": "Pseudo-regret", "reward": "Reward",
                "cpu_time": "CPU time (s)", "wall_time": "Wall time (s)", "memory_mb": "Peak memory (MB)",
                "pareto_regret": "Pareto regret", "pareto_share": "Pareto optimal pulls", "unfairness": "Unfairness"}


class Results:
    """Outputs of :meth:`Experiment.run <banditkit.Experiment.run>`.

    Attributes
    ----------
    steps : ndarray of shape (n_points,)
        Logged steps (1-based); every curve has one value per logged step.
    curves : dict
        ``curves[label][metric]`` is an array of shape (n_runs, n_points), or
        (n_runs, n_points, ...) for per-action diagnostics. ``regret``, ``reward``,
        ``cpu_time`` and ``wall_time`` are cumulative; ``memory_mb`` is the peak
        memory since the start of the run (NaN unless measured). Values are NaN
        after a run stopped on timeout.
    arrays : dict
        ``arrays[label][name]`` per-run arrays: ``action_counts`` (n_runs, n_actions),
        ``n_steps`` (n_runs,), and ``actions`` / ``bucket`` (n_runs, horizon) if tracked.
    meta : dict
        Settings of the experiment (environment, horizon, seed, weights, policies...).
    """

    def __init__(self, steps: np.ndarray, curves: Dict[str, Dict[str, np.ndarray]],
                 arrays: Dict[str, Dict[str, np.ndarray]], meta: Dict[str, Any]) -> None:
        self.steps = np.asarray(steps)
        self.curves = curves
        self.arrays = arrays
        self.meta = meta

    # ---------------------------------------------------------------- access
    @property
    def labels(self) -> List[str]:
        """Policy labels, in the order of the experiment."""
        return list(self.curves)

    @property
    def n_runs(self) -> int:
        """Total number of runs per policy (all instances together)."""
        return int(self.curve(self.labels[0]).shape[0])

    @property
    def n_instances(self) -> int:
        """Number of problem instances (1 for a single environment)."""
        return int(self.meta.get("n_instances", 1))

    def by_instance(self) -> Dict[int, "Results"]:
        """One Results per instance (e.g. for :func:`banditkit.plot.performance_profile`)."""
        first = self.labels[0]
        index = self.arrays[first].get("instance", np.zeros(self.n_runs, dtype=int))
        out = {}
        for i in np.unique(index):
            rows = np.flatnonzero(index == i)
            curves = {label: {m: a[rows] for m, a in named.items()} for label, named in self.curves.items()}
            arrays = {label: {m: a[rows] for m, a in named.items()} for label, named in self.arrays.items()}
            out[int(i)] = Results(self.steps, curves, arrays, {**self.meta, "n_instances": 1, "instance": int(i)})
        return out

    def instance_means(self, metric: str = "regret") -> Dict[str, np.ndarray]:
        """Final ``metric`` averaged over the runs of each instance: array (n_instances,) per policy."""
        out = {}
        for label in self.labels:
            final = self.final(metric)[label]
            index = self.arrays[label].get("instance", np.zeros(final.size, dtype=int))
            out[label] = np.array([np.nanmean(final[index == i]) for i in np.unique(index)])
        return out

    def compare(self, a: str, b: str, metric: str = "regret") -> Dict[str, Any]:
        """Paired comparison of two policies on the final ``metric`` (lower is better for regrets).

        With several instances the pairs are the instance means; with one instance,
        the runs (paired by their common random numbers). Returns the mean
        difference a - b, its 95% confidence interval, and the share of pairs where
        a has the lower value.
        """
        if self.n_instances > 1:
            diff = self.instance_means(metric)[a] - self.instance_means(metric)[b]
            unit = "instances"
        else:
            diff = self.final(metric)[a] - self.final(metric)[b]
            unit = "runs"
        diff = diff[~np.isnan(diff)]
        half = 1.96 * np.std(diff, ddof=1) / np.sqrt(diff.size) if diff.size > 1 else float("nan")
        return {"mean_difference": float(diff.mean()), "ci95": (float(diff.mean() - half), float(diff.mean() + half)),
                "a_lower_share": float(np.mean(diff < 0)), "n_pairs": int(diff.size), "pairs": unit}

    @property
    def best_params(self) -> Dict[str, Dict[str, Any]]:
        """Hyperparameters chosen by the search of ``Experiment(param_grid=...)``, per policy ({} if none)."""
        search = self.meta.get("param_search") or {}
        return {label: dict(info.get("best_params", {})) for label, info in search.items()}

    def search_table(self):
        """Every combination tried by the search, with its mean and std of the search metric (pandas DataFrame)."""
        import pandas as pd

        search = self.meta.get("param_search") or {}
        rows = [{"policy": label, **row["params"], "mean": row["mean"], "std": row["std"],
                 "best": row["params"] == info["best_params"]}
                for label, info in search.items() for row in info.get("table", [])]
        return pd.DataFrame(rows)

    def metrics(self, label: Optional[str] = None) -> List[str]:
        """Curves available for a policy (the first one by default)."""
        return list(self.curves[label or self.labels[0]])

    def curve(self, label: str, metric: str = "regret") -> np.ndarray:
        """Array (n_runs, n_points) of ``metric`` for policy ``label``."""
        if label not in self.curves:
            raise KeyError(f"No policy {label!r}. Available: {self.labels}")
        if metric not in self.curves[label]:
            raise KeyError(f"No metric {metric!r} for {label!r}. Available: {self.metrics(label)}")
        return self.curves[label][metric]

    def final(self, metric: str = "regret") -> Dict[str, np.ndarray]:
        """Last logged value of ``metric`` of each run, per policy (the last non-NaN one)."""
        out = {}
        for label in self.labels:
            c = self.curve(label, metric)
            valid = ~np.isnan(c)
            last = np.where(valid.any(axis=1), valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1), -1)
            out[label] = np.where(last >= 0, c[np.arange(c.shape[0]), np.maximum(last, 0)], np.nan)
        return out

    # ------------------------------------------------------------- summaries
    def summary(self, metrics: Sequence[str] = ("regret", "reward", "cpu_time", "memory_mb")):
        """Mean and standard deviation over runs of the final values, one row per policy (pandas DataFrame)."""
        import pandas as pd

        rows = []
        for label in self.labels:
            row: Dict[str, Any] = {"policy": label}
            for m in metrics:
                if m in self.curves[label]:
                    v = self.final(m)[label]
                    has_values = bool(np.any(~np.isnan(v)))
                    row[f"{m}_mean"] = float(np.nanmean(v)) if has_values else np.nan
                    row[f"{m}_std"] = float(np.nanstd(v)) if has_values else np.nan
            row["n_runs"] = self.n_runs
            rows.append(row)
        return pd.DataFrame(rows).set_index("policy")

    def to_latex(self, metrics: Sequence[str] = ("regret", "cpu_time", "memory_mb"), digits: int = 2) -> str:
        """Booktabs table of final values (mean ± std over runs), ready to paste in a paper."""
        table = self.summary(metrics)
        cols = [m for m in metrics if f"{m}_mean" in table.columns and table[f"{m}_mean"].notna().any()]
        lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
                 "Policy & " + " & ".join(_LATEX_NAMES.get(m, m) for m in cols) + " \\\\", "\\midrule"]
        for label, row in table.iterrows():
            cells = [f"${row[f'{m}_mean']:.{digits}f} \\pm {row[f'{m}_std']:.{digits}f}$" for m in cols]
            lines.append(_latex_escape(str(label)) + " & " + " & ".join(cells) + " \\\\")
        return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])

    def to_dataframe(self, metrics: Optional[Iterable[str]] = None, every: int = 1):
        """Long format DataFrame (policy, run, step, metric, value) of the scalar curves, for pandas or seaborn."""
        import pandas as pd

        idx = np.arange(0, self.steps.size, max(1, int(every)))
        frames = []
        for label, curves in self.curves.items():
            for m, arr in curves.items():
                if arr.ndim != 2 or (metrics is not None and m not in metrics):
                    continue
                frames.append(pd.DataFrame({"policy": label, "run": np.repeat(np.arange(arr.shape[0]), idx.size),
                                            "step": np.tile(self.steps[idx], arr.shape[0]), "metric": m,
                                            "value": arr[:, idx].reshape(-1)}))
        return pd.concat(frames, ignore_index=True)

    # ----------------------------------------------------------- persistence
    def save(self, path: Union[str, Path]) -> Path:
        """Save to a compressed ``.npz`` file (arrays and metadata, no pickle)."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, np.ndarray] = {"steps": self.steps,
                                          "meta": np.array(json.dumps(self.meta, default=_to_json))}
        for kind, store in (("curves", self.curves), ("arrays", self.arrays)):
            for label, named in store.items():
                for name, arr in named.items():
                    payload[f"{kind}\x1f{label}\x1f{name}"] = arr
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Results":
        """Load results saved with :meth:`save`."""
        curves: Dict[str, Dict[str, np.ndarray]] = {}
        arrays: Dict[str, Dict[str, np.ndarray]] = {}
        with np.load(Path(path), allow_pickle=False) as data:
            for key in data.files:
                if "\x1f" in key:
                    kind, label, name = key.split("\x1f")
                    (curves if kind == "curves" else arrays).setdefault(label, {})[name] = data[key]
            return cls(data["steps"], curves, arrays, json.loads(str(data["meta"])))

    @classmethod
    def merge(cls, *results: "Results") -> "Results":
        """Concatenate the runs of several Results logged at the same steps (e.g. from several machines)."""
        if not results:
            raise ValueError("Nothing to merge.")
        if any(not np.array_equal(r.steps, results[0].steps) for r in results[1:]):
            raise ValueError("Results must share the same logged steps to be merged.")
        curves: Dict[str, Dict[str, np.ndarray]] = {}
        arrays: Dict[str, Dict[str, np.ndarray]] = {}
        for res in results:
            for store, merged in ((res.curves, curves), (res.arrays, arrays)):
                for label, named in store.items():
                    for name, arr in named.items():
                        prev = merged.setdefault(label, {}).get(name)
                        merged[label][name] = arr if prev is None else np.concatenate([prev, arr])
        meta = {**results[0].meta, "merged_from": len(results)}
        return cls(results[0].steps, curves, arrays, meta)

    def plot(self, metric: str = "regret", **kwargs: Any):
        """Shortcut for :func:`banditkit.plot.curves`."""
        from .plot import curves

        return curves(self, metric=metric, **kwargs)

    def __repr__(self) -> str:
        return f"Results(policies={self.labels}, n_runs={self.n_runs}, horizon={self.meta.get('horizon')})"


def _latex_escape(text: str) -> str:
    for char, escaped in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#")):
        text = text.replace(char, escaped)
    return text


def _to_json(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return repr(obj)
