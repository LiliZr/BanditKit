"""Run policies on an environment over several runs, and measure regret, reward, time and memory."""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import __version__
from ._validation import check_bool, check_float, check_int, check_weights
from .envs.base import Environment
from .policies.base import Policy
from .results import CORE_METRICS, Results
from .spec import EnvSpec

__all__ = ["Experiment", "TRACKABLE", "DEFAULT_PARAM_GRID"]

_BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

#: Optional measurements, enabled with ``Experiment(track=...)``.
TRACKABLE = {
    "pseudo_regret": "cumulative regret computed with mean rewards (no noise)",
    "pareto_regret": "cumulative Pareto regret (multi-objective problems)",
    "pareto_share": "share of the pulls so far that went to Pareto optimal actions (multi-objective problems)",
    "unfairness": "spread of the pulls among Pareto optimal actions, Drugan and Nowe (2013); fixed means only",
    "reward_vec": "cumulative reward per objective, shape (n_runs, n_points, reward_dim)",
    "actions": "action played at every step, shape (n_runs, horizon)",
    "bucket": "bucket of the played action at every step, shape (n_runs, horizon)",
    "clusters": "scalar diagnostics of the policy (number of user clusters, of buckets...)",
    "estimates": "per-action diagnostics of the policy (estimated means, bonuses...)",
    "env_metrics": "cumulative dataset metrics returned by the environment (NDCG, CTR...)",
}


#: Grid of ``Experiment(param_grid=True)``. Each policy searches the parameters it has among
#: these and lists in ``default_search_params`` (alpha and lambda_ for UCB-type policies).
DEFAULT_PARAM_GRID: Dict[str, Tuple[float, ...]] = {
    "alpha": tuple(float(v) for v in np.logspace(-4, 2, 7)),     # 1e-4 ... 100
    "lambda_": tuple(float(v) for v in np.logspace(-5, 0, 6)),   # 1e-5 ... 1
    "c": (0.0, 1.0, 3.0),                                        # KL-UCB
    "epsilon": (0.01, 0.02, 0.05, 0.1, 0.2),                     # epsilon-greedy
    "tau": (0.01, 0.03, 0.1, 0.3, 1.0),                          # softmax
}
_HIGHER_IS_BETTER = ("reward", "pareto_share")


class Experiment:
    """Run one or more policies on one problem, or on several instances of a problem.

    Parameters
    ----------
    env : Environment or list of Environment
        The problem. A list gives several instances of the same kind of problem
        (same family, number of features and objectives; same number of actions for
        a MAB), for example drawn with :func:`banditkit.sample_instances`. Each instance gets ``n_runs``
        runs and the results pool them: averages are over instances and runs.
    policies : Policy, list of Policy, or dict {label: Policy}
        Policies to compare. Each run uses a fresh copy (``policy.clone()``).
    horizon : int
        Number of steps T per run (``total_timesteps`` in Stable-Baselines3).
    n_runs : int
        Independent runs per instance. Within a run, all policies see the same
        environment randomness (common random numbers).
    weights : array of shape (reward_dim,), optional
        Scalarization weights of a multi-objective reward (default: uniform).
    seed : int, optional
        Master seed; run seeds are derived from it. None draws one, stored in
        ``results.meta["seed"]`` so the experiment can be replayed.
    n_jobs : int
        Parallel processes (1: sequential, -1: all CPUs), as in scikit-learn.
    log_interval : int, optional
        Curves are recorded every ``log_interval`` steps and at the last step
        (default: T // 5000, about 5000 points per curve). Values at these steps are exact.
    timeout : float, optional
        Seconds of policy CPU time after which a run stops; its curve is NaN after that.
    train_freq : int
        Policies are updated every ``train_freq`` steps with the buffered
        observations (1: online), as in Stable-Baselines3.
    track : iterable of str
        Extra measurements, see :data:`TRACKABLE`.
    measure_memory : bool
        Measure peak memory with ``tracemalloc``. It slows Python code down
        (about fivefold), so compare CPU times only between runs with the same setting.
    param_grid : bool or dict, optional
        How hyperparameters are set. None (default): as given; left at their
        defaults, they give the theoretical confidence bounds (``alpha=1``).
        True: for each policy, search the best combination of :data:`DEFAULT_PARAM_GRID`
        on this problem. A dict ``{param: values}`` sets the grid (one value fixes the
        parameter for every policy that has it); ``{label: {param: values}}`` sets one
        grid per policy. The search is slower: every combination runs ``search_runs``
        times per instance, on other seeds than the reported runs, and the best one
        is then run ``n_runs`` times. See ``results.best_params``.
    search_runs : int
        Runs per instance and per combination during the search.
    search_metric : str
        Final value that ranks the combinations: "regret" (lowest is best),
        "pseudo_regret", "pareto_regret", or "reward" (highest is best).
    checkpoint_dir : str or Path, optional
        Each finished (policy, run) is saved there; a relaunch with the same
        settings reloads it instead of running it again.
    verbose : int
        0: silent. 1 (default): how hyperparameters are set, and the search results.
        2: also one line per finished run.
    callback : callable(dict), optional
        Called after each finished (policy, run) with its summary.
    """

    def __init__(self, env: Union[Environment, Sequence[Environment]],
                 policies: Union[Policy, Sequence[Policy], Dict[str, Policy]], horizon: int,
                 n_runs: int = 1, weights: Any = None, seed: Optional[int] = None, n_jobs: int = 1,
                 log_interval: Optional[int] = None, timeout: Optional[float] = None, train_freq: int = 1,
                 track: Iterable[str] = (), measure_memory: bool = False, param_grid: Any = None,
                 search_runs: int = 3, search_metric: str = "regret",
                 checkpoint_dir: Union[str, Path, None] = None, verbose: int = 1,
                 callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self.envs = self._check_envs(env)
        self.env = self.envs[0]
        self.horizon = check_int("horizon", horizon, low=1)
        for i, e in enumerate(self.envs):
            if e.max_episode_steps is not None and self.horizon > e.max_episode_steps:
                raise ValueError(f"`horizon` ({self.horizon}) exceeds the {e.max_episode_steps} steps "
                                 f"instance {i} can serve.")
        self.n_runs = check_int("n_runs", n_runs, low=1)
        self.weights = check_weights(weights, self.env.spec.reward_dim)
        self.seed = check_int("seed", seed, low=0, allow_none=True)
        self.n_jobs = check_int("n_jobs", n_jobs, low=-1)
        if self.n_jobs == 0:
            raise ValueError("`n_jobs` must be >= 1, or -1 for all CPUs.")
        self._log_interval_given = log_interval is not None
        self.log_interval = check_int("log_interval", log_interval, low=1, allow_none=True) or max(1, self.horizon // 5000)
        self.timeout = check_float("timeout", timeout, low=0.0, low_inclusive=False, allow_none=True)
        self.train_freq = check_int("train_freq", train_freq, low=1)
        self.track = tuple(track)
        unknown = [t for t in self.track if t not in TRACKABLE]
        if unknown:
            raise ValueError(f"Unknown `track` entries {unknown}. Available: {sorted(TRACKABLE)}.")
        for name in ("pareto_regret", "pareto_share", "unfairness"):
            if name in self.track and self.env.spec.reward_dim < 2:
                raise ValueError(f"`{name}` needs a multi-objective environment (reward_dim > 1).")
        if "estimates" in self.track and len({e.spec.n_actions for e in self.envs}) > 1:
            raise ValueError("`estimates` (one value per action) needs instances with the same number of actions.")
        self.measure_memory = check_bool("measure_memory", measure_memory)
        self.search_runs = check_int("search_runs", search_runs, low=1)
        self.search_metric = search_metric
        if search_metric not in ("regret", "pseudo_regret", "pareto_regret", "reward"):
            raise ValueError(f"`search_metric` must be 'regret', 'pseudo_regret', 'pareto_regret' or 'reward', "
                             f"got {search_metric!r}.")
        if search_metric in ("pseudo_regret", "pareto_regret") and search_metric not in self.track:
            self.track = self.track + (search_metric,)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.verbose = check_int("verbose", verbose, low=0)
        if callback is not None and not callable(callback):
            raise TypeError("`callback` must be callable.")
        self.callback = callback
        self.specs = [e.spec.with_horizon(self.horizon) for e in self.envs]
        self.spec = self.specs[0]
        self.policies = self._label_policies(policies)
        self.param_grid = param_grid
        self._grids = self._resolve_grids(param_grid)

    @property
    def n_instances(self) -> int:
        return len(self.envs)

    @staticmethod
    def _check_envs(env: Any) -> List[Environment]:
        envs = list(env) if isinstance(env, (list, tuple)) else [env]
        if not envs:
            raise ValueError("`env` is an empty list.")
        for i, e in enumerate(envs):
            if not isinstance(e, Environment):
                raise TypeError(f"`env` must be an Environment or a list of them; item {i} is {type(e).__name__}.")
        ref = envs[0].spec
        for i, e in enumerate(envs[1:], start=1):
            s = e.spec
            # Linear instances may offer different numbers of actions (the actions come with the
            # observation), e.g. the movies each user rated; MAB instances may not.
            same_actions = s.n_actions == ref.n_actions or ref.family != "mab"
            if (s.family, s.n_features, s.reward_dim) != (ref.family, ref.n_features, ref.reward_dim) \
                    or not same_actions:
                raise ValueError(f"Instance {i} is not the same kind of problem as instance 0: family, n_features "
                                 f"and reward_dim must match, and n_actions for MABs ({s} vs {ref}).")
        return envs

    def _label_policies(self, policies: Any) -> Dict[str, Policy]:
        """Validate every policy on this problem (fail fast) and give each a unique label."""
        if isinstance(policies, Policy):
            policies = [policies]
        if isinstance(policies, dict):
            items = list(policies.items())
            if any(not isinstance(label, str) or not label for label, _ in items):
                raise TypeError("Policy labels must be non-empty strings.")
        elif isinstance(policies, (list, tuple)):
            items = [(None, p) for p in policies]
        else:
            raise TypeError("`policies` must be a Policy, a list of policies or a dict {label: policy}.")
        if not items:
            raise ValueError("`policies` is empty.")
        labeled: Dict[str, Policy] = {}
        self._notes: Dict[str, List[Tuple[str, str]]] = {}
        for label, policy in items:
            if not isinstance(policy, Policy):
                raise TypeError(f"Expected Policy objects, got {type(policy).__name__}.")
            probe = policy.clone()  # generic policies (UCB...) get their concrete name here
            try:
                probe.reset(self.spec, self.weights, seed=0)
            except (ValueError, TypeError) as exc:
                raise type(exc)(f"Policy {label or probe.label!r}: {exc}") from None
            base = label or probe.label
            name, k = base, 2
            while name in labeled:
                name, k = f"{base} ({k})", k + 1
            # Generic policies are replaced by the class they resolve to, so that searches set its parameters.
            labeled[name] = probe.inner.clone() if getattr(probe, "inner", None) is not None else policy
            self._notes[name] = probe.parameter_notes()
        return labeled

    def _resolve_grids(self, param_grid: Any) -> Dict[str, Dict[str, List[Any]]]:
        """Grid of each policy: {label: {param: values}}, restricted to the parameters it has."""
        if param_grid is None or param_grid is False:
            return {}
        if param_grid is True:
            per_policy = None
            common = DEFAULT_PARAM_GRID
        elif isinstance(param_grid, dict) and param_grid and all(isinstance(v, dict) for v in param_grid.values()):
            unknown = [k for k in param_grid if k not in self.policies]
            if unknown:
                raise ValueError(f"`param_grid` names unknown policies {unknown}; labels are {list(self.policies)}.")
            per_policy, common = param_grid, None
        elif isinstance(param_grid, dict):
            per_policy, common = None, param_grid
        else:
            raise TypeError("`param_grid` must be True, a dict {param: values} or a dict {label: {param: values}}.")
        grids = {}
        for label, policy in self.policies.items():
            specs = policy._all_params()
            if per_policy is not None:
                wanted = per_policy.get(label, {})
                unknown = [k for k in wanted if k not in specs]
                if unknown:
                    raise ValueError(f"`param_grid[{label!r}]` has parameters {unknown} that {label} does not have.")
            elif param_grid is True:
                wanted = {k: v for k, v in common.items() if k in specs and k in policy.default_search_params}
            else:
                wanted = {k: v for k, v in common.items() if k in specs}
            grid = {}
            for k, values in wanted.items():
                values = list(np.atleast_1d(values)) if not isinstance(values, (list, tuple)) else list(values)
                if not values:
                    raise ValueError(f"`param_grid` gives no value for {k!r}.")
                grid[k] = [specs[k].validate(k, v.item() if isinstance(v, np.generic) else v) for v in values]
            grids[label] = grid
        return grids

    # ------------------------------------------------------------------- run
    def run(self) -> Results:
        """Run every (policy, instance, run) and return the :class:`~banditkit.Results`."""
        root_seed = np.random.SeedSequence(self.seed)
        main_seq, search_seq = root_seed.spawn(2)
        steps = _logged_steps(self.horizon, self.log_interval)
        configs = [_RunConfig(spec, self.horizon, self.weights, steps, self.timeout, self.train_freq, self.track,
                              self.measure_memory) for spec in self.specs]
        n_workers = (os.cpu_count() or 1) if self.n_jobs == -1 else self.n_jobs
        self._notices_at_start = self.notices()
        if self.verbose:
            self._say_plan()

        policies, search = dict(self.policies), {}
        if self._grids:
            policies, search = self._search(search_seq, configs, n_workers)

        seeds = [[s.spawn(2) for s in inst.spawn(self.n_runs)] for inst in main_seq.spawn(self.n_instances)]
        outputs: Dict[Tuple[str, int, int], Dict[str, np.ndarray]] = {}
        tasks = []
        for label, policy in policies.items():
            for i in range(self.n_instances):
                for r in range(self.n_runs):
                    cached = self._load_checkpoint(label, policy, i, r)
                    if cached is not None:
                        outputs[(label, i, r)] = cached
                    else:
                        tasks.append(((label, i, r), i, policy, *seeds[i][r]))
        total = len(policies) * self.n_instances * self.n_runs
        start = time.perf_counter()

        def done(key, out):
            outputs[key] = out
            label, i, r = key
            self._save_checkpoint(label, policies[label], i, r, out)
            info = {"policy": label, "instance": i, "run": r, "regret": float(np.nanmax(out["regret"])),
                    "cpu_time": float(np.nanmax(out["cpu_time"])), "elapsed": time.perf_counter() - start,
                    "done": len(outputs), "total": total}
            if self.verbose >= 2:
                where = f"instance {i} run {r}" if self.n_instances > 1 else f"run {r}"
                _say(f"[{info['done']}/{total}] {label} {where}: regret {info['regret']:.1f}, "
                     f"policy CPU {info['cpu_time']:.2f}s, elapsed {info['elapsed']:.1f}s")
            if self.callback is not None:
                self.callback(info)

        self._execute(tasks, configs, n_workers, done)
        return self._collect(steps, outputs, policies, search, root_seed)

    def _execute(self, tasks, configs, n_workers: int, done: Callable[[Any, Dict[str, np.ndarray]], None]) -> None:
        """Run ``(key, instance, policy, env_seed, policy_seed)`` tasks, in worker processes if n_workers > 1."""
        if n_workers == 1 or len(tasks) <= 1:
            for key, i, policy, env_seed, policy_seed in tasks:
                done(key, run_single(self.envs[i], policy.clone(), env_seed, policy_seed, configs[i]))
            return
        # One BLAS thread per worker, otherwise n_workers x n_cores threads compete. Spawned
        # workers inherit the environment, so the variables are set around the pool creation.
        saved = {k: os.environ.get(k) for k in _BLAS_THREAD_VARS}
        os.environ.update({k: "1" for k in _BLAS_THREAD_VARS if saved[k] is None})
        try:
            # "spawn": safe with multithreaded BLAS, unlike "fork".
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn")) as pool:
                futures = {pool.submit(run_single, self.envs[i], policy.clone(), env_seed, policy_seed, configs[i]): key
                           for key, i, policy, env_seed, policy_seed in tasks}
                for future in as_completed(futures):
                    done(futures[future], future.result())
        except BrokenProcessPool:
            raise RuntimeError("A worker process failed to start. With n_jobs > 1, run the experiment from a .py "
                               "file (under `if __name__ == \"__main__\":`) or a notebook, with picklable "
                               "environments and policies (no lambdas); or use n_jobs=1.") from None
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)

    # ---------------------------------------------------------------- search
    def _search(self, search_seq, configs, n_workers: int):
        """Run every combination of each policy's grid; keep the best one per policy."""
        seeds = [[s.spawn(2) for s in inst.spawn(self.search_runs)] for inst in search_seq.spawn(self.n_instances)]
        candidates: Dict[str, List[Tuple[Dict[str, Any], Policy]]] = {}
        tasks = []
        for label, policy in self.policies.items():
            grid = self._grids.get(label, {})
            combos = [dict(zip(grid, values)) for values in itertools.product(*grid.values())] if grid else []
            candidates[label] = [(c, policy.clone().set_params(**c)) for c in combos]
            for c_idx, (_, cand) in enumerate(candidates[label]):
                for i in range(self.n_instances):
                    for r in range(self.search_runs):
                        tasks.append(((label, c_idx, i, r), i, cand, *seeds[i][r]))
        finals: Dict[Tuple[str, int], List[float]] = {}

        def done(key, out):
            label, c_idx = key[0], key[1]
            finals.setdefault((label, c_idx), []).append(_last_valid(out[self.search_metric]))

        self._execute(tasks, configs, n_workers, done)
        sign = -1.0 if self.search_metric in _HIGHER_IS_BETTER else 1.0
        policies, search = {}, {}
        for label, policy in self.policies.items():
            if not candidates[label]:
                policies[label] = policy
                search[label] = {"best_params": {}, "table": [], "searched": False}
                continue
            table = []
            for c_idx, (combo, _) in enumerate(candidates[label]):
                values = np.array(finals[(label, c_idx)], dtype=float)
                table.append({"params": combo, "mean": float(np.nanmean(values)), "std": float(np.nanstd(values))})
            best = int(np.nanargmin([sign * row["mean"] for row in table]))
            policies[label] = candidates[label][best][1]
            search[label] = {"best_params": table[best]["params"], "table": table, "searched": True}
            if self.verbose:
                params = ", ".join(f"{k}={_fmt(v)}" for k, v in table[best]["params"].items())
                _say(f"  {label}: best {params} (search {self.search_metric} {table[best]['mean']:.4g} "
                     f"± {table[best]['std']:.2g})")
        return policies, search

    # -------------------------------------------------------------- messages
    def _defaults_in_use(self) -> Dict[str, Tuple[str, Dict[str, Any]]]:
        """Tunable hyperparameters left at their default: {label: (class name, {param: value})}."""
        out = {}
        for label, policy in self.policies.items():
            specs, params = policy._all_params(), policy.get_params()
            left = {k: params[k] for k in policy.default_search_params if k in specs and params[k] == specs[k].default}
            if left:
                out[label] = (type(policy).__name__, left)
        return out

    def _diagnostic_keys(self) -> Tuple[List[str], List[str]]:
        """Scalar and per-action diagnostics the policies can record (``track="clusters"`` / ``"estimates"``)."""
        scalar, per_action = [], []
        for policy in self.policies.values():
            try:
                probe = policy.clone()
                probe.reset(self.spec, self.weights, seed=0)
                diag = probe.diagnostics()
            except Exception:  # a policy that cannot report before its first step reports nothing
                continue
            for k, v in diag.items():
                (per_action if np.ndim(v) > 0 else scalar).append(k)
        return sorted(set(scalar)), sorted(set(per_action))

    def notices(self) -> List[Dict[str, str]]:
        """What to know before reading the results of this run, one dict per item.

        Each item has ``key``, ``title``, ``text`` (what happens, in short sentences), ``default`` (the
        default setting) and ``todo`` (how to change it; empty when there is nothing to change). Printed as
        a numbered warning when the run starts (``verbose >= 1``) and stored in ``results.meta["notices"]``
        (the dashboard shows them as a table).
        """
        out: List[Dict[str, str]] = []

        def add(key: str, title: str, text: str, default: str = "", todo: str = "") -> None:
            out.append({"key": key, "title": title, "text": text, "default": default, "todo": todo})

        def they(labels: List[str]) -> str:
            return "it" if len(labels) == 1 else "they"

        grids = self._grids or {}
        defaults = {}
        for label, (cls, params) in self._defaults_in_use().items():
            arbitrary = {text.split("=", 1)[0] for kind, text in self._notes.get(label, []) if kind == "arbitrary"}
            kept = {k: v for k, v in params.items() if k not in arbitrary}  # arbitrary values are listed below
            if kept and not grids.get(label):
                defaults[label] = (cls, kept)
        if defaults:
            cls, params = next(iter(defaults.values()))
            key = next(iter(params))
            example = f"{cls}({key}={_fmt(_EXAMPLE_VALUES.get(key, params[key]))})"
            groups: Dict[str, List[str]] = {}  # same values, one group: "alpha=1 for UCB1 and KL-UCB"
            for label, (_, p) in defaults.items():
                groups.setdefault(" ".join(f"{k}={_fmt(v)}" for k, v in p.items()), []).append(label)
            if len(defaults) == len(self.policies) and len(groups) == 1:
                used = f"All policies use the values from their papers: {next(iter(groups))}."
            elif len(groups) == 1:
                (v, labels), = groups.items()
                used = (f"{_list(labels)} {_verb(labels, 'uses', 'use')} the values from "
                        f"{_verb(labels, 'its paper', 'their papers')}: {v}.")
            else:
                used = ("Some policies use the values from their papers: "
                        + "; ".join(f"{v} for {_list(labels)}" for v, labels in groups.items()) + ".")
            add("defaults", "Default hyperparameters",
                f"{used} These values are safe in theory. In practice, they often explore too much. "
                "The regret can then be higher than with tuned values.",
                "The values from the papers (param_grid=None).",
                f"Search the best values with param_grid=True. Or set them yourself, e.g. {example}.")
        searched = {label: g for label, g in grids.items() if g}
        if searched:
            n = sum(int(np.prod([len(v) for v in g.values()])) for g in searched.values())
            runs = f"{self.search_runs} run" + ("s" if self.search_runs > 1 else "")
            add("search", "Grid search",
                f"The hyperparameters of {_list(list(searched))} come from a grid search on this problem. "
                f"The search tried {n} combinations, {runs} each, on other seeds than the results. "
                f"Each policy keeps the combination with the best final {self.search_metric}. "
                "These values fit this problem only.",
                "No search: the values from the papers (param_grid=None).",
                'Set your own grid, e.g. param_grid={"alpha": [0.1, 1]}. '
                "Set the runs per combination with search_runs (default 3).")
        assumed: Dict[str, List[str]] = {}
        for label, notes in self._notes.items():
            for kind, text in notes:
                if kind == "assumed":
                    assumed.setdefault(text.split(" (")[0], []).append(label)
        if assumed:
            first, labels = next(iter(assumed.items()))
            param = first.split(":")[0]
            example = f"{type(self.policies[labels[0]]).__name__}({param}=...)"
            values = "; ".join(f"{text.split(': ')[-1]} ({text.split(':')[0]}) for {_list(labels)}"
                               for text, labels in assumed.items())
            add("assumed", "Assumed constants",
                "The problem does not give some constants of the confidence bounds. "
                f"The policies assume them: {values}.",
                "1 when the problem does not give them.",
                f"Give the true values, e.g. {example}.")

        bucketed = [label for label, p in self.policies.items() if p.get_params().get("n_buckets")]
        if bucketed:
            by = ("UCB" if all(self.policies[l].get_params().get("bucket_by_ucb", True) for l in bucketed)
                  else "UCB or estimated mean")
            it = they(bucketed)
            add("buckets", "Arm buckets (heuristic)",
                f"{_list(bucketed)} {_verb(bucketed, 'ranks', 'rank')} the actions by {by} and "
                f"{_verb(bucketed, 'groups', 'group')} them into buckets. "
                f"At each step, {it} {_verb(bucketed, 'searches', 'search')} one bucket only. "
                f"This is faster, but {it} can miss the best action. The regret bound no longer holds.",
                "No buckets: every action is searched (n_buckets=None).",
                "Set n_buckets=None to search every action. Set bucket_by_ucb=False to rank by estimated mean.")
        kmeans = [label for label, p in self.policies.items() if p.get_params().get("user_model") == "kmeans"]
        if kmeans:
            add("kmeans", "K-means users (heuristic)",
                f"{_list(kmeans)} {_verb(kmeans, 'pulls', 'pull')} each user model towards the center of its "
                "k-means cluster. This is a heuristic: there is no regret guarantee.",
                'One model for all users (user_model="shared").',
                'Set user_model="individual" for one model per user.')
        missing, names = [], []
        if "pseudo_regret" not in self.track:
            missing.append("the pseudo-regret"); names.append('"pseudo_regret"')
        if self.spec.reward_dim > 1:
            for name, what in (("pareto_regret", "the Pareto regret"), ("pareto_share", "the share of Pareto-optimal pulls"),
                               ("unfairness", "the unfairness"), ("reward_vec", "the reward per objective")):
                if name not in self.track:
                    missing.append(what); names.append(f'"{name}"')
        scalar, per_action = self._diagnostic_keys()
        # Diagnostics that cannot change on this problem: clusters of a single user, buckets nobody uses.
        if (self.spec.n_users or 1) <= 1:
            scalar = [k for k in scalar if k != "n_user_clusters"]
        if not any(p.get_params().get("n_buckets") for p in self.policies.values()):
            scalar = [k for k in scalar if k != "n_buckets"]
        if scalar and "clusters" not in self.track:
            missing.append(f'{" and ".join(scalar)} over time'); names.append('"clusters"')
        if per_action and "estimates" not in self.track:
            missing.append("the per-action estimates"); names.append('"estimates"')
        if "actions" not in self.track:
            missing.append("the action of every step"); names.append('"actions"')
        if missing or not self.measure_memory:
            what = missing + ([] if self.measure_memory else ["the peak memory"])
            listed = what[0] if len(what) == 1 else (" and ".join(what) if len(what) == 2
                                                     else ", ".join(what[:-1]) + ", and " + what[-1])
            how = ([f"Store them with track=[{', '.join(names)}]."] if names else []) + \
                  ([] if self.measure_memory else ["Measure memory with measure_memory=True (slower)."])
            add("not_stored", "Not stored",
                f"This run does not store {listed}. The charts cannot show them.",
                "Regret, reward, CPU time and wall time only (track=(), measure_memory=False).",
                " ".join(how))

        total = self.n_runs * self.n_instances
        if total == 1:
            add("runs", "One run", "Each policy runs once. So there is no confidence interval.",
                "One run per policy (n_runs=1).", "Increase n_runs, e.g. n_runs=10. Or use several instances.")
        elif total < 5:
            add("runs", "Few runs", f"Each policy runs only {total} times. So the confidence intervals are wide.",
                "One run per policy (n_runs=1).", "Increase n_runs, e.g. n_runs=10.")
        if self.log_interval > 1:
            k = self.log_interval
            why = (f" You set log_interval={k}." if self._log_interval_given
                   else f" This is the default here: T // 5000 = {self.horizon} // 5000 = {k}.")
            add("log_interval", f"Curves stored every {k} steps",
                f"The curves keep one value every {k} steps. Between two kept steps, the line is drawn, "
                f"not measured.{why}",
                "One value every T // 5000 steps (log_interval=None).",
                "Set log_interval=1 to keep every step. The files get larger.")
        if self.timeout is not None:
            add("timeout", "Timeout",
                f"Each run stops after {_fmt(self.timeout)} s of policy CPU time. After that, its curves are empty.",
                "No limit (timeout=None).", "Raise timeout, or set timeout=None.")
        if self.measure_memory:
            add("memory", "Slower timings",
                "Peak memory is measured (measure_memory=True). This makes Python code about 5 times slower. "
                "So the CPU times are higher than in a normal run.",
                "Memory is not measured (measure_memory=False).",
                "Set measure_memory=False for normal CPU times. "
                "Compare CPU times only between runs with the same setting.")
        if self.n_jobs != 1:
            add("n_jobs", "Parallel runs",
                f"The runs are parallel (n_jobs={self.n_jobs}), with one BLAS thread each. "
                "So the CPU times can differ from runs one at a time.",
                "One run at a time (n_jobs=1).",
                "Set n_jobs=1. Compare CPU times only between runs with the same n_jobs.")
        if self.train_freq > 1:
            add("train_freq", "Batched updates",
                f"The policies learn every {self.train_freq} steps, from the observations of these steps. "
                "Between two updates, they do not learn.",
                "The policies learn at every step (train_freq=1).", "Set train_freq=1.")
        if self.checkpoint_dir is not None:
            add("checkpoint", "Checkpoints",
                f"Finished runs are saved in {self.checkpoint_dir}. A new launch loads them instead of running them again.",
                "Nothing is saved during the run (checkpoint_dir=None).",
                "Empty this folder after you change the code or the settings.")
        return out

    def _say_plan(self) -> None:
        where = f" on {self.n_instances} instances" if self.n_instances > 1 else ""
        runs = f"{self.n_runs} run" + ("s" if self.n_runs > 1 else "")
        _say(f"BanditKit: {len(self.policies)} policies, {runs}{where}, horizon {self.horizon}.")
        notices = getattr(self, "_notices_at_start", None) or self.notices()
        if notices:
            _say(_format_notices(notices))
        defaults = self._defaults_in_use()
        _say("Hyperparameters of each policy:")
        for label, notes in self._notes.items():
            grid = (self._grids or {}).get(label)
            if grid:
                n = int(np.prod([len(v) for v in grid.values()]))
                desc = ", ".join(f"{k} in {_fmt(min(v))}..{_fmt(max(v))} ({len(v)})" if len(v) > 1
                                 else f"{k}={_fmt(v[0])}" for k, v in grid.items())
                _say(f"  {label}: grid search, {n} combinations ({desc})")
                continue
            # Defaults, arbitrary or assumed values first (they deserve attention), then values read
            # from the problem.
            noted = {text.split("=", 1)[0] for kind, text in notes if kind == "arbitrary"}
            left = {k: v for k, v in defaults.get(label, ("", {}))[1].items() if k not in noted}
            parts = [f"default {', '.join(f'{k}={_fmt(v)}' for k, v in left.items())}"] if left else []
            parts += [f"{text} [{kind}]" for kind, text in notes if kind != "from_problem"]
            known = [text.split(": ", 1)[-1] for kind, text in notes if kind == "from_problem"]
            if known:
                parts.append(", ".join(known) + " from the environment")
            _say(f"  {label}: " + ("; ".join(parts) if parts else "as given"))

    # ---------------------------------------------------------------- output
    def _collect(self, steps: np.ndarray, outputs: dict, policies: Dict[str, Policy], search: dict,
                 root_seed: np.random.SeedSequence) -> Results:
        """Stack the runs of each policy into a Results object (instance-major order)."""
        curves, arrays, styles = {}, {}, {}
        keys = [(i, r) for i in range(self.n_instances) for r in range(self.n_runs)]
        for label, policy in policies.items():
            runs = [outputs[(label, i, r)] for i, r in keys]
            curves[label] = {k: np.stack([o[k] for o in runs]) for k in runs[0] if not k.startswith("_")}
            arrays[label] = {k[1:]: _stack_padded([o[k] for o in runs]) for k in runs[0] if k.startswith("_")}
            arrays[label]["instance"] = np.array([i for i, _ in keys], dtype=np.int64)
            probe = policy.clone()
            probe.reset(self.spec, self.weights, seed=0)
            styles[label] = {"base_name": probe.base_name or type(probe).__name__,
                             "techniques": list(probe.techniques), "class": type(probe).__name__}
        meta = {
            "banditkit_version": __version__,
            "environment": repr(self.env) if self.n_instances == 1 else [repr(e) for e in self.envs],
            "n_instances": self.n_instances,
            "spec": {k: getattr(self.spec, k) for k in ("n_actions", "n_features", "reward_dim", "n_users", "horizon",
                                                        "reward_range", "feature_norm", "noise_std", "theta_norm")},
            "family": self.spec.family,
            "n_actions_range": [min(s.n_actions for s in self.specs), max(s.n_actions for s in self.specs)],
            "horizon": self.horizon,
            "n_runs": self.n_runs,
            "weights": self.weights.tolist(),
            "seed": root_seed.entropy,
            "log_interval": self.log_interval,
            "timeout": self.timeout,
            "train_freq": self.train_freq,
            "track": list(self.track),
            "measure_memory": self.measure_memory,
            "policies": {label: {"class": type(p).__name__, "params": p.get_params()} for label, p in policies.items()},
            "parameter_notes": {label: [list(n) for n in notes] for label, notes in self._notes.items()},
            "notices": self._notices_at_start,
            "param_search": ({label: {**info, "metric": self.search_metric, "search_runs": self.search_runs}
                              for label, info in search.items()} if search else None),
            "styles": styles,
        }
        return Results(steps, curves, arrays, meta)

    # ------------------------------------------------------------ checkpoints
    def _checkpoint_path(self, label: str, i: int, r: int) -> Optional[Path]:
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", label) / f"instance_{i:03d}_run_{r:04d}.npz"

    def _checkpoint_key(self, policy: Policy, i: int) -> str:
        """Settings that must match for a checkpoint to be reused."""
        return json.dumps({"policy": type(policy).__name__, "params": repr(sorted(policy.get_params().items())),
                           "env": repr(self.envs[i]), "horizon": self.horizon, "seed": self.seed,
                           "n_runs": self.n_runs, "n_instances": self.n_instances, "weights": self.weights.tolist(),
                           "log_interval": self.log_interval, "track": list(self.track),
                           "train_freq": self.train_freq})

    def _load_checkpoint(self, label: str, policy: Policy, i: int, r: int) -> Optional[Dict[str, np.ndarray]]:
        path = self._checkpoint_path(label, i, r)
        if path is None or not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            if str(data["__key"]) != self._checkpoint_key(policy, i):
                return None  # settings changed: run again
            return {k: data[k] for k in data.files if k != "__key"}

    def _save_checkpoint(self, label: str, policy: Policy, i: int, r: int, out: Dict[str, np.ndarray]) -> None:
        path = self._checkpoint_path(label, i, r)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, __key=np.array(self._checkpoint_key(policy, i)), **out)
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written checkpoint


def _stack_padded(arrays: List[np.ndarray]) -> np.ndarray:
    """Stack per-run arrays; 1-D arrays of different lengths (action counts of instances with
    different numbers of actions) are padded with zeros."""
    if len({np.shape(a) for a in arrays}) <= 1:
        return np.stack(arrays)
    if any(np.ndim(a) != 1 for a in arrays):
        raise ValueError("Per-run arrays of different shapes cannot be stacked.")
    out = np.zeros((len(arrays), max(len(a) for a in arrays)), dtype=np.result_type(*arrays))
    for i, a in enumerate(arrays):
        out[i, :len(a)] = a
    return out


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


#: Title of the warning printed when a run starts (see :meth:`Experiment.notices`).
WARNING_TITLE = "/!\\ WARNING /!\\"
_EXAMPLE_VALUES = {"alpha": 0.1, "lambda_": 0.01, "c": 0.0, "epsilon": 0.05, "tau": 0.1}


def _use_color(stream: Any) -> bool:
    """ANSI colors: in a terminal or a notebook, unless NO_COLOR is set (FORCE_COLOR forces them)."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if "ipykernel" in sys.modules:
        return True
    return bool(getattr(stream, "isatty", lambda: False)()) and os.environ.get("TERM") != "dumb"


def _width() -> int:
    import shutil

    return max(60, min(shutil.get_terminal_size((100, 24)).columns, 100))


def _banner(text: str, stream: Any = None) -> str:
    """``text`` on a yellow band (or between two rules when colors are off)."""
    width = max(len(text) + 2, _width())
    if _use_color(stream if stream is not None else sys.stderr):
        return f"\033[1;30;103m{(' ' + text).ljust(width)}\033[0m"
    return f"{'=' * width}\n {text}\n{'=' * width}"


def _format_notices(notices: List[Dict[str, str]], stream: Any = None) -> str:
    """The numbered warning: a band, then for each item what to know, the default and how to change it."""
    import textwrap

    width = _width()
    color = _use_color(stream if stream is not None else sys.stderr)
    n = len(notices)
    lines = [_banner(WARNING_TITLE, stream)]
    pad = len(f" {n}) ")
    for i, notice in enumerate(notices, start=1):
        head = f" {i}) ".rjust(pad)
        title = f"\033[1m{notice['title']}.\033[0m" if color else f"{notice['title']}."
        body = textwrap.fill(f"{notice['title']}. {notice['text']}", width=width, initial_indent=head,
                             subsequent_indent=" " * pad)
        lines.append(body.replace(f"{notice['title']}.", title, 1))
        for label, field in (("Default", "default"), ("How to change", "todo")):
            if notice.get(field):
                line = textwrap.fill(f"{label}: {notice[field]}", width=width, initial_indent=" " * pad,
                                     subsequent_indent=" " * pad)
                lines.append(f"\033[2m{line}\033[0m" if color else line)
    return "\n".join(lines)


def _verb(items: List[str], one: str, several: str) -> str:
    return one if len(items) == 1 else several


def _list(labels: List[str], limit: int = 8) -> str:
    """'A, B and C' (at most ``limit`` names)."""
    if len(labels) > limit:
        return ", ".join(labels[:limit]) + f" and {len(labels) - limit} more"
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


def _fmt(value: Any) -> str:
    """Short display of a hyperparameter value (1e-05, 0.1, 100)."""
    return f"{value:.3g}" if isinstance(value, float) else repr(value)


def _last_valid(curve: np.ndarray) -> float:
    """Last non-NaN value of a curve (a run stopped by the timeout ends with NaN)."""
    valid = curve[~np.isnan(curve)]
    return float(valid[-1]) if valid.size else float("nan")


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _RunConfig:
    spec: EnvSpec
    horizon: int
    weights: np.ndarray
    logged_steps: np.ndarray
    timeout: Optional[float]
    train_freq: int
    track: Tuple[str, ...]
    measure_memory: bool


def _logged_steps(horizon: int, interval: int) -> np.ndarray:
    """1-based steps at which curves are recorded: every ``interval`` steps, plus the last one."""
    steps = np.arange(interval, horizon + 1, interval)
    return steps if steps.size and steps[-1] == horizon else np.append(steps, horizon)


class _RunRecorder:
    """Accumulates the measurements of one run and writes them at the logged steps."""

    def __init__(self, cfg: _RunConfig) -> None:
        self.cfg = cfg
        n_points, spec = cfg.logged_steps.size, cfg.spec
        self.curves = {m: np.full(n_points, np.nan) for m in CORE_METRICS}
        for m in ("pseudo_regret", "pareto_regret", "pareto_share", "unfairness"):
            if m in cfg.track:
                self.curves[m] = np.full(n_points, np.nan)
        if "reward_vec" in cfg.track:
            self.curves["reward_vec"] = np.full((n_points, spec.reward_dim), np.nan)
        self.actions = np.full(cfg.horizon, -1, dtype=np.int32) if "actions" in cfg.track else None
        self.buckets = np.full(cfg.horizon, -1, dtype=np.int32) if "bucket" in cfg.track else None
        self.diagnostics: Dict[str, List[Tuple[int, Any]]] = {}
        self.action_counts = np.zeros(spec.n_actions, dtype=np.int64)
        self.totals = {"regret": 0.0, "pseudo_regret": 0.0, "pareto_regret": 0.0, "reward": 0.0,
                       "pareto_pulls": 0.0}
        self.optimal: Optional[np.ndarray] = None  # Pareto optimal actions, when the means are fixed
        self._fixed_means: Optional[bool] = None
        self.reward_vec = np.zeros(spec.reward_dim)
        self.env_metrics: Dict[str, float] = {}
        self.next_point = 0

    def add_step(self, t: int, env: Environment, obs, action: int, reward: np.ndarray, policy: Policy) -> None:
        w, track = self.cfg.weights, self.cfg.track
        scalar = float(w @ reward)
        best = env.best_mean_reward(w, obs)
        self.totals["reward"] += scalar
        self.totals["regret"] += best - scalar
        if "pseudo_regret" in track:
            self.totals["pseudo_regret"] += best - float(w @ env.mean_reward(action, obs))
        if "pareto_regret" in track or "pareto_share" in track or "unfairness" in track:
            gaps = self._pareto_gaps(env, obs)
            self.totals["pareto_regret"] += float(gaps[action])
            self.totals["pareto_pulls"] += float(gaps[action] <= 1e-12)
        if "env_metrics" in track:
            for k, v in (env.extra_metrics(action, obs) or {}).items():
                self.env_metrics[k] = self.env_metrics.get(k, 0.0) + float(v)
        self.reward_vec += reward
        self.action_counts[action] += 1
        if self.actions is not None:
            self.actions[t] = action
        if self.buckets is not None:
            self.buckets[t] = policy.last_bucket()

    def _pareto_gaps(self, env: Environment, obs) -> np.ndarray:
        """Pareto gaps of the actions, computed once when the means are fixed."""
        if self._fixed_means is None:
            self._fixed_means = bool(env.static_means and self.cfg.spec.n_users == 1)
        if self._fixed_means and self.optimal is not None:
            return self._gaps
        gaps = env.pareto_gaps(obs)
        if self._fixed_means:
            self._gaps, self.optimal = gaps, np.flatnonzero(gaps <= 1e-12)
        return gaps

    def log_if_due(self, t: int, policy: Policy, cpu_time: float, wall_time: float) -> None:
        """Write the current totals if step t + 1 is a logged step."""
        if self.next_point >= self.cfg.logged_steps.size or t + 1 != self.cfg.logged_steps[self.next_point]:
            return
        i, track = self.next_point, self.cfg.track
        for m in ("regret", "reward", "pseudo_regret", "pareto_regret"):
            if m in self.curves:
                self.curves[m][i] = self.totals[m]
        if "pareto_share" in self.curves:
            self.curves["pareto_share"][i] = self.totals["pareto_pulls"] / (t + 1)
        if "unfairness" in self.curves and self.optimal is not None:
            # phi = mean over Pareto optimal arms of (pulls - mean pulls)^2 (0: perfectly even).
            self.curves["unfairness"][i] = float(np.var(self.action_counts[self.optimal]))
        self.curves["cpu_time"][i], self.curves["wall_time"][i] = cpu_time, wall_time
        self.curves["memory_mb"][i] = tracemalloc.get_traced_memory()[1] / 2 ** 20 if self.cfg.measure_memory else np.nan
        if "reward_vec" in track:
            self.curves["reward_vec"][i] = self.reward_vec
        if "clusters" in track or "estimates" in track:
            for k, v in policy.diagnostics().items():
                per_action = np.ndim(v) > 0
                if (per_action and "estimates" in track) or (not per_action and "clusters" in track):
                    self.diagnostics.setdefault(k, []).append((i, np.copy(v)))
        for k, v in self.env_metrics.items():
            self.diagnostics.setdefault("env_" + k, []).append((i, v))
        self.next_point += 1

    def output(self, n_steps: int) -> Dict[str, np.ndarray]:
        """Curves (keys without '_') and per-run arrays (keys starting with '_')."""
        out = dict(self.curves)
        n_points = self.cfg.logged_steps.size
        for k, series in self.diagnostics.items():
            shapes = {np.shape(v) for _, v in series}
            if len(shapes) != 1:
                continue  # variable size (e.g. a Pareto front): not stored
            arr = np.full((n_points,) + shapes.pop(), np.nan)
            for i, v in series:
                arr[i] = v
            out[k] = arr
        out["_action_counts"] = self.action_counts
        out["_n_steps"] = np.array(n_steps)
        if self.actions is not None:
            out["_actions"] = self.actions
        if self.buckets is not None:
            out["_bucket"] = self.buckets
        return out


def run_single(env: Environment, policy: Policy, env_seed: Any, policy_seed: Any, cfg: _RunConfig) -> Dict[str, np.ndarray]:
    """One run of one policy (Gymnasium loop). Module-level so that worker processes can pickle it."""
    obs, _ = env.reset(seed=env_seed)
    policy.reset(cfg.spec, cfg.weights, seed=policy_seed)
    recorder = _RunRecorder(cfg)
    buffer: List[tuple] = []
    cpu_time, n_steps = 0.0, cfg.horizon
    if cfg.measure_memory:
        tracemalloc.start()
    wall_start = time.perf_counter()
    try:
        for t in range(cfg.horizon):
            tic = time.process_time()
            action = policy.predict(obs)
            cpu_time += time.process_time() - tic

            next_obs, reward, terminated, truncated, _ = env.step(action)

            buffer.append((obs, action, reward))
            if len(buffer) >= cfg.train_freq or t == cfg.horizon - 1:
                tic = time.process_time()
                for item in buffer:
                    policy.update(*item)
                cpu_time += time.process_time() - tic
                buffer.clear()

            recorder.add_step(t, env, obs, int(action), reward, policy)
            recorder.log_if_due(t, policy, cpu_time, time.perf_counter() - wall_start)
            obs = next_obs
            if cfg.timeout is not None and cpu_time > cfg.timeout:
                n_steps = t + 1
                break
            if terminated or truncated:
                n_steps = t + 1
                break
    finally:
        if cfg.measure_memory:
            tracemalloc.stop()
    return recorder.output(n_steps)
