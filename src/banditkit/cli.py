"""Command line: run an experiment from a configuration file, build the dashboard.

    banditkit run configs/linear.json --generate-dashboard
    banditkit run results/*/config.json         # rerun every experiment of results/
    banditkit dashboard                         # every folder of results/
    banditkit dashboard results/linear results/my-run --out my-dashboard.html

``python -m banditkit`` does the same without installing the script.

A configuration is a JSON file::

    {
      "description": "One or two sentences shown in the dashboard.",
      "env": {"kind": "linear", "n_instances": 10, "n_actions": 1000, "n_features": 20, "seed": 0},
      "policies": [{"policy": "LinUCB"}, {"policy": "LinUCB", "n_buckets": "sqrt"}],
      "experiment": {"horizon": 3000, "n_runs": 2, "seed": 0}
    }

``env.kind`` is one of ``bernoulli``, ``gaussian``, ``linear``, ``multi_user`` (synthetic,
see :func:`banditkit.sample_instances`), ``make_env`` (keywords of :func:`banditkit.make_env`),
``npz`` (``path`` of a file read by :func:`banditkit.datasets.load_npz`) or ``dataset``
(``function``: a function of :mod:`banditkit.datasets`, with its keywords). Each policy
gives its class (``policy``), an optional ``label`` and its hyperparameters.
``experiment`` holds the keywords of :class:`banditkit.Experiment`.
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["main", "build_env", "build_policies", "run_config"]

SYNTHETIC = ("bernoulli", "gaussian", "linear", "multi_user")


def build_env(spec: Dict[str, Any]):
    """Environment (or list of instances) described by the ``env`` entry of a configuration."""
    import banditkit as bk

    spec = dict(spec)
    kind = spec.pop("kind", None)
    if kind in SYNTHETIC:
        return bk.sample_instances(kind, spec.pop("n_instances", 1), **spec)
    if kind == "make_env":
        return bk.make_env(**spec)
    if kind == "npz":
        from .datasets import load_npz

        return load_npz(spec.pop("path"), **spec)
    if kind == "dataset":
        from . import datasets

        name = spec.pop("function")
        if name not in datasets.__all__:
            raise ValueError(f"Unknown dataset function {name!r}; available: {sorted(datasets.__all__)}.")
        env = getattr(datasets, name)(**spec)
        return env.to_env() if hasattr(env, "to_env") else env
    raise ValueError(f"Unknown env kind {kind!r}; use one of {SYNTHETIC + ('make_env', 'npz', 'dataset')}.")


def build_policies(entries: Sequence[Dict[str, Any]]) -> list:
    """Policies described by the ``policies`` entry of a configuration."""
    import banditkit as bk

    policies = []
    for entry in entries:
        entry = dict(entry)
        name = entry.pop("policy")
        if name not in bk.__all__:
            raise ValueError(f"Unknown policy {name!r}.")
        label = entry.pop("label", None)
        if label is not None:
            entry["name"] = label
        policies.append(getattr(bk, name)(**entry))
    return policies


def run_config(config: Dict[str, Any], folder: Path, n_jobs: Optional[int] = None,
               overrides: Optional[Dict[str, Any]] = None, command: Optional[str] = None, explain: bool = True) -> Path:
    """Run the experiment of ``config`` and save it as a results folder (with its explanatory charts)."""
    import banditkit as bk
    from .report import save_run

    settings = {**config.get("experiment", {}), **(overrides or {})}
    if n_jobs is not None:
        settings["n_jobs"] = n_jobs
    config = {**config, "experiment": {k: v for k, v in settings.items() if k != "n_jobs"}}
    results = bk.Experiment(build_env(config["env"]), build_policies(config["policies"]), **settings).run()
    folder = save_run(results, folder, config=config, description=config.get("description", ""), command=command)
    if explain:
        _explain(config, folder, results.labels)
    return folder


def _explain(config: Dict[str, Any], folder: Path, labels=None) -> None:
    """explain.json of a results folder (the "Inside the policies" charts), when plotly is installed."""
    try:
        import plotly  # noqa: F401
    except ImportError:
        print("Install plotly for the explanatory charts of the dashboard (pip install plotly).")
        return
    from .explain import save_explain

    save_explain(config, folder, labels)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="banditkit", description="Run bandit experiments and view their results.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the experiments of JSON configurations and save results folders.")
    run.add_argument("configs", nargs="+", type=Path, metavar="config",
                     help="JSON configurations (a results folder's config.json works too).")
    run.add_argument("--name", help="Name of the results folder (default: name of the configuration file; "
                                    "one configuration only).")
    run.add_argument("--results-dir", type=Path, default=Path("results"), help="Parent of the results folders.")
    run.add_argument("--horizon", type=int, help="Override the horizon.")
    run.add_argument("--n-runs", type=int, help="Override the number of runs per instance.")
    run.add_argument("--n-jobs", type=int, help="Parallel processes.")
    run.add_argument("--generate-dashboard", action="store_true", help="Rebuild the dashboard at the end.")
    run.add_argument("--dashboard", type=Path, default=Path("dashboard.html"), help="Dashboard file to write.")

    dash = sub.add_parser("dashboard", help="Build the dashboard from results folders.")
    dash.add_argument("folders", nargs="*", type=Path, help="Results folders (default: every folder of --results-dir).")
    dash.add_argument("--results-dir", type=Path, default=Path("results"), help="Where to look for results folders.")
    dash.add_argument("--out", type=Path, default=Path("dashboard.html"), help="HTML file to write.")
    dash.add_argument("--offline", action="store_true", help="Embed plotly.js (works without internet).")
    dash.add_argument("--open", action="store_true", help="Open the dashboard in the browser.")
    dash.add_argument("--explain", action="store_true",
                      help="Recompute the explanatory charts (explain.json) of the folders from their config.json.")
    run.add_argument("--no-explain", action="store_true", help="Skip the explanatory charts (explain.json).")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    from .report import build_dashboard

    if args.command == "run":
        if args.name and len(args.configs) > 1:
            raise SystemExit("--name works with one configuration only.")
        overrides = {k: v for k, v in (("horizon", args.horizon), ("n_runs", args.n_runs)) if v is not None}
        for path in args.configs:
            config = json.loads(path.read_text())
            name = args.name or (path.parent.name if path.name == "config.json" else path.stem)
            folder = args.results_dir / name
            command = f"python -m banditkit run {folder.as_posix()}/config.json"
            run_config(config, folder, n_jobs=args.n_jobs, overrides=overrides, command=command,
                       explain=not args.no_explain)
            print(f"Results saved in {folder}/")
        if args.generate_dashboard:
            out = build_dashboard(results_dir=args.results_dir, out=args.dashboard)
            print(f"Dashboard: {out}")
        return 0

    if args.explain:
        from .report import find_runs

        for folder in args.folders or find_runs(args.results_dir):
            if (Path(folder) / "config.json").is_file():
                _explain(json.loads((Path(folder) / "config.json").read_text()), Path(folder))
    out = build_dashboard(args.folders or None, results_dir=args.results_dir, out=args.out, offline=args.offline)
    print(f"Dashboard: {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
