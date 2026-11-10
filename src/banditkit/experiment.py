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
        (default: about 2000 points per curve).
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
        self.log_interval = check_int("log_interval", log_interval, low=1, allow_none=True) or max(1, self.horizon // 2000)
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

    def _say_plan(self) -> None:
        where = f" on {self.n_instances} instances" if self.n_instances > 1 else ""
        runs = f"{self.n_runs} run" + ("s" if self.n_runs > 1 else "")
        _say(f"banditkit: {len(self.policies)} policies, {runs}{where}, horizon {self.horizon}.")
        if not self._grids:
            defaults = self._defaults_in_use()
            if defaults:
                _warn_defaults(defaults)
            else:
                _say("Hyperparameters as given.")
            for label, notes in self._notes.items():
                # Defaults, arbitrary or assumed values first (they deserve attention), then values read
                # from the problem.
                noted = {text.split("=", 1)[0] for kind, text in notes if kind == "arbitrary"}
                left = {k: v for k, v in defaults.get(label, ("", {}))[1].items() if k not in noted}
                parts = [f"default {', '.join(f'{k}={_fmt(v)}' for k, v in left.items())}"] if left else []
                parts += [f"{text} [{kind}]" for kind, text in notes if kind != "from_problem"]
                known = [text.split(": ", 1)[-1] for kind, text in notes if kind == "from_problem"]
                if known:
                    parts.append(", ".join(known) + " from the environment")
                if parts:
                    _say(f"  {label}: " + "; ".join(parts))
            return
        _say(f"Hyperparameters: search of the best values for this problem (param_grid, slower). Each combination "
             f"runs {self.search_runs} time(s) per instance, on other seeds than the reported runs; "
             f"criterion: final {self.search_metric}.")
        for label, grid in self._grids.items():
            if not grid:
                _say(f"  {label}: nothing to search, parameters as given")
                continue
            n = int(np.prod([len(v) for v in grid.values()]))
            desc = ", ".join(f"{k} in {_fmt(min(v))}..{_fmt(max(v))} ({len(v)})" if len(v) > 1 else f"{k}={_fmt(v[0])}"
                             for k, v in grid.items())
            _say(f"  {label}: {n} combinations ({desc})")

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


#: Title of the banner printed when policies run with their default hyperparameters.
DEFAULTS_WARNING = "WARNING: you are using default parameters"
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


def _banner(text: str, stream: Any = None) -> str:
    """``text`` on a yellow band (or between two rules when colors are off)."""
    import shutil

    width = max(len(text) + 2, min(shutil.get_terminal_size((80, 24)).columns, 100))
    if _use_color(stream if stream is not None else sys.stderr):
        return f"\033[1;30;103m{(' ' + text).ljust(width)}\033[0m"
    return f"{'=' * width}\n {text}\n{'=' * width}"


def _warn_defaults(defaults: Dict[str, Tuple[str, Dict[str, Any]]]) -> None:
    """Banner and note when policies run with their default (paper) hyperparameters."""
    cls, params = next(iter(defaults.values()))
    key = next(iter(params))
    example = f"{cls}({key}={_fmt(_EXAMPLE_VALUES.get(key, params[key]))})"
    _say(_banner(DEFAULTS_WARNING))
    _say("Defaults are the values of the papers (alpha=1: theoretical confidence bounds).\n"
         "They are safe in theory but often explore too much in practice.\n"
         "Note: if you want values fitted to your problem, run a grid search (param_grid=True, or\n"
         "\"param_grid\": true in the \"experiment\" part of a config file), or set the parameters you want,\n"
         f"e.g. {example}.")


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
