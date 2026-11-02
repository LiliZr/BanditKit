"""Clustering of bandits: CLUB and SCLUB.

Both keep one ridge model per user and one aggregated model per cluster of
users; actions are chosen with the cluster model. A cluster model is
``V_C = lambda I + sum_{u in C} (V_u - lambda I)``, ``b_C = sum_{u in C} b_u``; it is
updated incrementally, and rebuilt only when clusters split or merge.
"""
from __future__ import annotations

import math
import warnings
from collections import Counter, deque
from typing import Any, Dict, Iterable, List, Set

import numpy as np

from .._validation import ParamSpec
from ..spec import EnvSpec, Observation
from ._linear_core import RidgeModel
from .linear import BucketedPolicy

__all__ = ["CLUB", "SCLUB"]

BONUS = ParamSpec("choice", default="paper", choices=("paper", "oful"),
                  doc="'oful': LinUCB radius on the cluster model; 'paper': exploration term of the original paper.")


def confidence_term(n: float) -> float:
    """``sqrt((1 + ln(1 + n)) / (1 + n))``: the F(T) of CLUB and SCLUB, for n observations."""
    return math.sqrt((1.0 + math.log1p(n)) / (1.0 + n))


class ClusteringPolicy(BucketedPolicy):
    """Per-user models and aggregation of user models into cluster models."""

    families = frozenset({"multi_user"})

    def _reset(self) -> None:
        super()._reset()
        self.user_models: Dict[int, RidgeModel] = {}  # created on a user's first update
        self._zero = np.zeros(self.dim, dtype=self._float)
        self._lambda_eye = np.eye(self.dim, dtype=self._float) * self.lambda_
        self._n_updates = 0

    def _user_model(self, user_id: int) -> RidgeModel:
        if user_id not in self.user_models:
            self.user_models[user_id] = RidgeModel(self.dim, self.lambda_, dtype=self._float, keep_V=True)
        return self.user_models[user_id]

    def _user_theta(self, user_id: int) -> np.ndarray:
        model = self.user_models.get(user_id)
        return self._zero if model is None else model.theta

    def _user_n_obs(self, user_id: int) -> int:
        model = self.user_models.get(user_id)
        return 0 if model is None else model.n_obs

    def _aggregate(self, members: Iterable[int], X: np.ndarray) -> RidgeModel:
        """Cluster model built from the models of its members."""
        cluster = RidgeModel(self.dim, self.lambda_, dtype=self._float, keep_V=True)
        for u in members:
            model = self.user_models.get(u)
            if model is not None and model.n_obs:
                cluster.V += model.V - self._lambda_eye
                cluster.b += model.b
                cluster.n_obs += model.n_obs
        cluster.refresh_from_V()
        self._attach_buckets(cluster, X)
        return cluster

    def _absorb(self, target: RidgeModel, source: RidgeModel, sign: float = 1.0) -> None:
        """Add (sign=1) or remove (sign=-1) the observations of ``source`` from ``target``."""
        target.V += sign * (source.V - self._lambda_eye)
        target.b += sign * source.b
        target.n_obs += int(sign) * source.n_obs
        target.refresh_from_V()

    def _width_scale(self, cluster: RidgeModel) -> float:
        return self.alpha * self._beta(cluster.n_obs)


class CLUB(ClusteringPolicy):
    """CLUB (Gentile, Li and Zappella, 2014).

    Users are the nodes of a graph and clusters are its connected components.
    After an update of user i, the edge (i, j) is deleted when
    ``||theta_i - theta_j|| > cluster_alpha * (F(T_i) + F(T_j))``, F = :func:`confidence_term`.

    ``bonus="paper"`` uses the width ``alpha * sqrt(x^T M^-1 x * ln(t + 1))`` of the
    paper's Figure 1. The initial graph is Erdos-Renyi with p = 3 ln(n) / n,
    resampled until connected (the paper's experiments), or complete (its
    analysis; memory grows as n_users^2, so limited to 5000 users).
    """

    base_name = "CLUB"
    params = {
        "cluster_alpha": ParamSpec("float", default=2.0, low=0.0, low_inclusive=False,
                                   doc="Edge deletion threshold alpha_2 of the paper."),
        "bonus": BONUS,
        "initial_graph": ParamSpec("choice", default="erdos_renyi", choices=("erdos_renyi", "complete"),
                                   doc="Initial user graph."),
        "graph_p": ParamSpec("float", default=None, low=0.0, high=1.0, low_inclusive=False, allow_none=True,
                             doc="Edge probability of the Erdos-Renyi graph (None: 3 ln(n) / n)."),
    }

    def _check_spec(self, spec: EnvSpec) -> None:
        super()._check_spec(spec)
        if self.initial_graph == "complete" and spec.n_users > 5000:
            raise ValueError(f"A complete graph on {spec.n_users} users is too large; use initial_graph='erdos_renyi'.")

    def _reset(self) -> None:
        super()._reset()
        n = self.spec.n_users
        self.neighbors: List[Set[int]] = self._initial_graph(n)
        self.component_of = np.zeros(n, dtype=np.int64)    # cluster id of each user
        self.components: Dict[int, Set[int]] = {}          # cluster id -> users
        self.cluster_models: Dict[int, RidgeModel] = {}
        self._next_id = 0
        self._pending_components = self._connected_components(range(n))  # built at the first step

    # ------------------------------------------------------------ graph
    def _initial_graph(self, n: int) -> List[Set[int]]:
        if self.initial_graph == "complete":
            return [set(range(n)) - {u} for u in range(n)]
        p = self.graph_p if self.graph_p is not None else min(1.0, 3.0 * math.log(max(n, 2)) / n)
        for _ in range(20):
            graph = self._erdos_renyi(n, p)
            if len(self._connected_components(range(n), graph)) == 1:
                return graph
        warnings.warn("No connected Erdos-Renyi graph in 20 draws; using a disconnected one.", stacklevel=2)
        return graph

    def _erdos_renyi(self, n: int, p: float) -> List[Set[int]]:
        """G(n, p) as adjacency sets: the number of edges is Binomial, then edges are drawn without repeats."""
        graph: List[Set[int]] = [set() for _ in range(n)]
        n_edges = int(self.rng.binomial(n * (n - 1) // 2, p))
        added = 0
        while added < n_edges:
            us = self.rng.integers(n, size=n_edges - added)
            vs = self.rng.integers(n, size=n_edges - added)
            for u, v in zip(us.tolist(), vs.tolist()):
                if u != v and v not in graph[u] and added < n_edges:
                    graph[u].add(v)
                    graph[v].add(u)
                    added += 1
        return graph

    def _connected_components(self, nodes: Iterable[int], graph=None) -> List[Set[int]]:
        """Connected components of ``nodes`` (breadth-first search)."""
        graph = self.neighbors if graph is None else graph
        remaining, components = set(nodes), []
        while remaining:
            start = remaining.pop()
            component, queue = {start}, deque([start])
            while queue:
                for w in graph[queue.popleft()]:
                    if w in remaining:
                        remaining.discard(w)
                        component.add(w)
                        queue.append(w)
            components.append(component)
        return components

    def _add_component(self, members: Set[int], X: np.ndarray) -> None:
        cid, self._next_id = self._next_id, self._next_id + 1
        self.components[cid] = members
        self.component_of[np.fromiter(members, dtype=np.int64)] = cid
        self.cluster_models[cid] = self._aggregate(members, X)

    def _cluster_model(self, user_id: int, X: np.ndarray) -> RidgeModel:
        if self._pending_components:
            for members in self._pending_components:
                self._add_component(members, X)
            self._pending_components = []
        return self.cluster_models[int(self.component_of[user_id])]

    # ----------------------------------------------------------- bandit
    def _width_scale(self, cluster: RidgeModel) -> float:
        if self.bonus == "paper":
            return self.alpha * math.sqrt(math.log(self._n_updates + 2.0))  # ln(t + 1) with t = 1-based step
        return super()._width_scale(cluster)

    def _predict(self, obs: Observation) -> int:
        X = self._working_features(obs)
        cluster = self._cluster_model(obs.user_id, X)
        return self._ucb_action(X, cluster, cluster.theta, self._width_scale(cluster))

    def _update(self, obs: Observation, action: int, reward: np.ndarray, scalar_reward: float) -> None:
        X = self._working_features(obs)
        u, x = obs.user_id, X[action]
        self._user_model(u).update(x, scalar_reward)
        cluster = self._cluster_model(u, X)
        cluster.update(x, scalar_reward)
        self._n_updates += 1
        self._rebuild_buckets_if_due(cluster, X, self._width_scale(cluster))
        self._delete_far_edges(u, X)

    def _delete_far_edges(self, u: int, X: np.ndarray) -> None:
        neighbors = list(self.neighbors[u])
        if not neighbors:
            return
        thetas = np.stack([self._user_theta(v) for v in neighbors])
        thresholds = self.cluster_alpha * (confidence_term(self._user_n_obs(u))
                                           + np.array([confidence_term(self._user_n_obs(v)) for v in neighbors]))
        far = np.linalg.norm(thetas - self._user_theta(u), axis=1) > thresholds
        if not far.any():
            return
        for v in np.asarray(neighbors)[far].tolist():
            self.neighbors[u].discard(v)
            self.neighbors[v].discard(u)
        # The cluster of u may have split into several connected components.
        cid = int(self.component_of[u])
        parts = self._connected_components(self.components[cid])
        if len(parts) > 1:
            del self.components[cid], self.cluster_models[cid]
            for part in parts:
                self._add_component(part, X)

    def diagnostics(self) -> Dict[str, Any]:
        n_buckets = sum(c.buckets.champions.size for c in self.cluster_models.values() if c.buckets is not None)
        return {"n_user_clusters": len(self.components) or len(self._pending_components), "n_buckets": n_buckets}


class SCLUB(ClusteringPolicy):
    """SCLUB (Li, Chen, Li and Leung, 2019).

    Steps are grouped in phases of 2^s steps. At the start of a phase every user
    becomes unchecked and each cluster stores a pivot estimate. After an update
    of user i, in cluster j:

    * split i out of j if ``||theta_i - pivot_j|| > cluster_alpha (F(T_i) + F(T_pivot))``,
      or (``use_frequency``) if its frequency ``T_i / t`` differs from another
      member's by more than ``2 freq_alpha F(t)``;
    * mark i as checked; once all users of j are checked, merge j with any checked
      cluster j' such that ``||theta_j - theta_j'|| < cluster_alpha / 2 (F(T_j) + F(T_j'))``
      and (``use_frequency``) ``|p_j - p_j'| < freq_alpha F(t)``, with ``p_j = T_j / (|C_j| t)``.

    ``split_merge_freq > 1`` runs these tests every that many steps only (faster;
    not in the paper). ``bonus="paper"`` uses Theorem 1's radius
    ``R sqrt(d ln(1 + T/d) + 2 ln(4 m n_u))``, m the current number of clusters
    (needs the horizon T).
    """

    base_name = "SCLUB"
    params = {
        "cluster_alpha": ParamSpec("float", default=1.0, low=0.0, low_inclusive=False,
                                   doc="Split / merge threshold alpha_theta of the paper."),
        "freq_alpha": ParamSpec("float", default=2.0, low=0.0, low_inclusive=False,
                                doc="Frequency threshold alpha_p of the paper."),
        "use_frequency": ParamSpec("bool", default=True, doc="Also split / merge on user arrival frequencies."),
        "split_merge_freq": ParamSpec("int", default=1, low=1, doc="Run split / merge tests every this many steps."),
        "bonus": BONUS,
    }

    def _check_spec(self, spec: EnvSpec) -> None:
        super()._check_spec(spec)
        if self.bonus == "paper" and spec.horizon is None:
            raise ValueError("SCLUB(bonus='paper') needs the horizon T: fit(..., horizon=T), EnvSpec(horizon=T) "
                             "or an Experiment.")

    def _reset(self) -> None:
        super()._reset()
        n = self.spec.n_users
        self.cluster_of = np.zeros(n, dtype=np.int64)
        self.members: Dict[int, Set[int]] = {0: set(range(n))}
        self.cluster_models: Dict[int, RidgeModel] = {}          # built lazily from members
        self.pivots: Dict[int, tuple] = {0: (self._zero.copy(), 0)}  # (theta, n_obs) at the phase start
        # Multiset of the members' observation counts, for the min / max user frequency in O(1).
        self._member_counts: Dict[int, Counter] = {0: Counter({0: n})}
        self._min_count, self._max_count = {0: 0}, {0: 0}
        self._checked = np.zeros(n, dtype=bool)
        self._n_checked = {0: 0}
        self._next_id = 1
        self._phase, self._phase_end = 1, 2
        self.n_splits = self.n_merges = 0

    # ----------------------------------------------------------- helpers
    def _cluster_model(self, j: int, X: np.ndarray) -> RidgeModel:
        if j not in self.cluster_models:
            self.cluster_models[j] = self._aggregate(self.members[j], X)
        return self.cluster_models[j]

    def _width_scale(self, cluster: RidgeModel) -> float:
        if self.bonus == "oful":
            return super()._width_scale(cluster)
        d, T = self.dim, self.spec.horizon
        inner = d * math.log(1.0 + T / d) + 2.0 * math.log(4.0 * len(self.members) * self.spec.n_users)
        return self.alpha * self._R * math.sqrt(inner)

    def _start_phase_if_due(self, X: np.ndarray) -> None:
        if self._n_updates < self._phase_end:
            return
        self._phase += 1
        self._phase_end = self._n_updates + 2 ** self._phase
        self._checked[:] = False
        self._n_checked = {j: 0 for j in self.members}
        for j in self.members:
            model = self._cluster_model(j, X)
            self.pivots[j] = (model.theta.copy(), model.n_obs)

    # ------------------------------------------------------------ bandit
    def _predict(self, obs: Observation) -> int:
        X = self._working_features(obs)
        cluster = self._cluster_model(int(self.cluster_of[obs.user_id]), X)
        return self._ucb_action(X, cluster, cluster.theta, self._width_scale(cluster))

    def _update(self, obs: Observation, action: int, reward: np.ndarray, scalar_reward: float) -> None:
        X = self._working_features(obs)
        u, x = obs.user_id, X[action]
        j = int(self.cluster_of[u])
        user = self._user_model(u)
        user.update(x, scalar_reward)
        cluster = self._cluster_model(j, X)
        cluster.update(x, scalar_reward)
        self._count_increased(j, user.n_obs - 1)
        self._n_updates += 1
        self._rebuild_buckets_if_due(cluster, X, self._width_scale(cluster))
        test_now = self._n_updates % self.split_merge_freq == 0
        if test_now:
            j = self._split_if_needed(u, j, X)
        if not self._checked[u]:
            self._checked[u] = True
            self._n_checked[j] += 1
        if test_now and self._n_checked[j] == len(self.members[j]):
            self._merge_if_possible(j, X)
        self._start_phase_if_due(X)

    # ------------------------------------------- user count bookkeeping
    def _count_increased(self, j: int, old: int) -> None:
        """A member of cluster j went from ``old`` to ``old + 1`` observations."""
        counts = self._member_counts[j]
        counts[old] -= 1
        emptied = counts[old] == 0
        if emptied:
            del counts[old]
        counts[old + 1] += 1
        self._max_count[j] = max(self._max_count[j], old + 1)
        if emptied and self._min_count[j] == old:
            self._min_count[j] = old + 1  # every remaining count is >= old + 1

    def _remove_count(self, j: int, value: int) -> None:
        counts = self._member_counts[j]
        counts[value] -= 1
        if counts[value] == 0:
            del counts[value]
            if counts:
                self._min_count[j], self._max_count[j] = min(counts), max(counts)

    # ------------------------------------------------------ split / merge
    def _split_if_needed(self, u: int, j: int, X: np.ndarray) -> int:
        """Move u to a new cluster if it no longer fits cluster j. Returns u's cluster."""
        if len(self.members[j]) == 1:
            return j
        user = self.user_models[u]
        pivot_theta, pivot_n = self.pivots[j]
        split = np.linalg.norm(user.theta - pivot_theta) > self.cluster_alpha * (
            confidence_term(user.n_obs) + confidence_term(pivot_n))
        if not split and self.use_frequency:
            t = self._n_updates
            freq, low, high = user.n_obs / t, self._min_count[j] / t, self._max_count[j] / t
            split = max(freq - low, high - freq) > 2.0 * self.freq_alpha * confidence_term(t)
        if not split:
            return j

        new = self._next_id
        self._next_id += 1
        self.members[j].discard(u)
        self._remove_count(j, user.n_obs)
        self._n_checked[j] -= int(self._checked[u])
        self._absorb(self.cluster_models[j], user, sign=-1.0)

        model = RidgeModel(self.dim, self.lambda_, dtype=self._float, keep_V=True)
        model.V, model.b, model.n_obs = user.V.copy(), user.b.copy(), user.n_obs
        model.refresh_from_V()
        self._attach_buckets(model, X)
        self.members[new], self.cluster_models[new], self.cluster_of[u] = {u}, model, new
        self.pivots[new] = (user.theta.copy(), user.n_obs)
        self._member_counts[new] = Counter({user.n_obs: 1})
        self._min_count[new] = self._max_count[new] = user.n_obs
        self._n_checked[new] = int(self._checked[u])
        self.n_splits += 1
        return new

    def _merge_if_possible(self, j: int, X: np.ndarray) -> None:
        """Merge checked clusters into j while one is close enough."""
        merged = True
        while merged:
            merged = False
            cj = self.cluster_models[j]
            for k in list(self.members):
                if k == j or self._n_checked.get(k, 0) != len(self.members[k]):
                    continue
                ck = self._cluster_model(k, X)
                close = np.linalg.norm(cj.theta - ck.theta) < 0.5 * self.cluster_alpha * (
                    confidence_term(cj.n_obs) + confidence_term(ck.n_obs))
                if close and self.use_frequency:
                    t = self._n_updates
                    p_j = cj.n_obs / (len(self.members[j]) * t)
                    p_k = ck.n_obs / (len(self.members[k]) * t)
                    close = abs(p_j - p_k) < self.freq_alpha * confidence_term(t)
                if close:
                    self._merge(j, k, X)
                    merged = True
                    break

    def _merge(self, j: int, k: int, X: np.ndarray) -> None:
        """Merge cluster k into cluster j."""
        self._absorb(self.cluster_models[j], self.cluster_models.pop(k))
        self._attach_buckets(self.cluster_models[j], X)
        for u in self.members[k]:
            self.cluster_of[u] = j
        self.members[j] |= self.members.pop(k)
        self._member_counts[j].update(self._member_counts.pop(k))
        self._min_count[j], self._max_count[j] = min(self._member_counts[j]), max(self._member_counts[j])
        del self._min_count[k], self._max_count[k]
        self._n_checked[j] += self._n_checked.pop(k)
        self.pivots.pop(k, None)
        self.n_merges += 1

    def diagnostics(self) -> Dict[str, Any]:
        n_buckets = sum(c.buckets.champions.size for c in self.cluster_models.values() if c.buckets is not None)
        return {"n_user_clusters": len(self.members), "n_buckets": n_buckets}
