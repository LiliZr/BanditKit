"""Linear bandit policies, with optional random projection, action buckets and user models."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._utils import argmax_random
from .._validation import ParamSpec
from ..spec import EnvSpec, Observation
from ._linear_core import ActionBuckets, RidgeModel, make_projection, weighted_norms
from .base import Policy

__all__ = ["LinUCB", "CBRAP", "ConfidenceBall1", "LinTS", "LinGreedy"]


class LinearPolicy(Policy):
    """Shared machinery: working features (projected or not) and the OFUL confidence radius.

    The radius is Theorem 2 of Abbasi-Yadkori et al. (2011) in the working dimension k
    (n_features, or n_components after projection), after n observations::

        beta(n) = R sqrt(2 ln(1/delta) + k ln(1 + n L^2 / (lambda k))) + sqrt(lambda) S

    The UCB of features x is ``x . theta + alpha * beta(n) * ||x||_{V^-1}``.
    """

    families = frozenset({"linear", "multi_user"})
    params = {
        "alpha": ParamSpec("float", default=1.0, low=0.0, doc="Multiplier of the confidence width (1: theoretical radius)."),
        "lambda_": ParamSpec("float", default=0.1, low=0.0, low_inclusive=False, doc="Ridge regularization lambda."),
        "delta": ParamSpec("float", default=0.01, low=0.0, high=1.0, low_inclusive=False, high_inclusive=False,
                           doc="Confidence level delta of the radius."),
        "subgaussian": ParamSpec("float", default=None, low=0.0, allow_none=True,
                                 doc="Sub-Gaussian constant R of the noise (None: from the environment, else 1)."),
        "theta_bound": ParamSpec("float", default=None, low=0.0, low_inclusive=False, allow_none=True,
                                 doc="Bound S on ||theta|| (None: from the environment, else 1)."),
        "feature_bound": ParamSpec("float", default=None, low=0.0, low_inclusive=False, allow_none=True,
                                   doc="Bound L on feature norms (None: largest norm of the working features)."),
        "n_components": ParamSpec("int", default=None, low=1, allow_none=True,
                                  doc="Dimension after random projection (None: no projection), as in scikit-learn."),
        "projection": ParamSpec("choice", default="gaussian", choices=("gaussian", "fjlt"),
                                doc="Random projection: Gaussian or Fast Johnson-Lindenstrauss transform."),
        "eps": ParamSpec("float", default=0.1, low=0.0, high=1.0, low_inclusive=False, high_inclusive=False,
                         doc="Distortion parameter of the FJLT."),
        "fjlt_p": ParamSpec("int", default=2, low=1, high=2, doc="Norm (1 or 2) preserved by the FJLT."),
        "dtype": ParamSpec("choice", default="float64", choices=("float64", "float32"), doc="Floating point precision."),
    }

    arbitrary_defaults = {"lambda_": "any lambda > 0 keeps the bound valid"}

    @property
    def techniques(self) -> Tuple[str, ...]:
        if self.n_components is None:
            return ()
        return ("FJLT",) if self.projection == "fjlt" else ("RP",)

    def _check_spec(self, spec: EnvSpec) -> None:
        if self.n_components is not None and self.n_components >= spec.n_features:
            raise ValueError(f"`n_components` ({self.n_components}) must be smaller than n_features ({spec.n_features}).")

    def _reset(self) -> None:
        spec = self.spec
        self._float = np.float64 if self.dtype == "float64" else np.float32
        self.dim = self.n_components or spec.n_features  # working dimension k
        self.projection_matrix = None
        if self.n_components is not None:
            self.projection_matrix = make_projection(
                spec.n_features, self.n_components, self.projection, self.rng, n_points=spec.n_actions,
                eps=self.eps, fjlt_p=self.fjlt_p).astype(self._float)
        self._raw_features = None   # last action_features seen, to detect a change of action set
        self._features = None       # their working (projected) version
        self._L = 1.0
        self._feature_cache: List[Tuple[np.ndarray, np.ndarray, float]] = []  # last two action sets
        # Constants R and S of the theoretical radius: given, read from the environment, or assumed.
        self._R = self._bound_constant("subgaussian", "R", self.subgaussian, spec.scalar_noise_std(self.weights))
        self._S = self._bound_constant("theta_bound", "S", self.theta_bound, spec.scalar_theta_norm(self.weights))

    def _bound_constant(self, param: str, symbol: str, given: Optional[float], known: Optional[float]) -> float:
        if given is not None:
            return given
        if known is not None:
            self._note("from_problem", f"{param}: {symbol}={known:.3g}")
            return known
        self._note("assumed", f"{param}: {symbol}=1 (unknown for this problem)")
        return 1.0

    def _working_features(self, obs: Observation) -> np.ndarray:
        """Features in the working space; recomputed only when the action set changes.

        The last two action sets are cached, so that alternating between a
        candidate set and single played actions (scikit-learn style loop) does
        not project the candidates again.
        """
        X = obs.action_features
        if X is None:
            raise ValueError(f"{type(self).__name__} needs action features in the observation.")
        if X is not self._raw_features:
            entry = next((e for e in self._feature_cache if e[0] is X), None)
            if entry is None:
                Z = X if self.projection_matrix is None else X @ self.projection_matrix.T
                Z = np.ascontiguousarray(Z, dtype=self._float)
                if self.feature_bound is not None and self.projection_matrix is None:
                    L = self.feature_bound
                else:
                    L = float(np.linalg.norm(Z, axis=1).max()) or 1.0
                entry = (X, Z, L)
            self._feature_cache = [entry] + [e for e in self._feature_cache if e is not entry][:1]
            self._raw_features, self._features, self._L = entry
        return self._features

    def _beta(self, n_obs: int) -> float:
        k, lam = self.dim, self.lambda_
        inner = 2.0 * math.log(1.0 / self.delta) + k * math.log(1.0 + n_obs * self._L ** 2 / (lam * k))
        return self._R * math.sqrt(inner) + math.sqrt(lam) * self._S


class BucketedPolicy(LinearPolicy):
    """Adds arm bucketing (label "AB") to policies built on :class:`RidgeModel`."""

    params = {
        "n_buckets": ParamSpec("int", default=None, low=1, allow_none=True, special=("sqrt",),
                               doc="Number of action buckets ('sqrt': round(sqrt(K)); None: no buckets)."),
        "bucket_rebuild_freq": ParamSpec("int", default=500, low=1,
                                         doc="A model rebuilds its buckets after this many of its own updates."),
        "bucket_by_ucb": ParamSpec("bool", default=False, doc="Rank actions by UCB instead of estimated mean when rebuilding."),
        "refresh_champions": ParamSpec("bool", default=False,
                                       doc="After each prediction, the chosen bucket's champion becomes its best UCB action."),
    }

    def _check_params(self) -> None:
        if self.n_buckets is None and (self.bucket_by_ucb or self.refresh_champions):
            raise ValueError("`bucket_by_ucb` and `refresh_champions` need `n_buckets`.")

    @property
    def techniques(self) -> Tuple[str, ...]:
        return super().techniques + (("AB",) if self.n_buckets is not None else ())

    def _check_spec(self, spec: EnvSpec) -> None:
        super()._check_spec(spec)
        if isinstance(self.n_buckets, int) and self.n_buckets > spec.n_actions:
            raise ValueError(f"`n_buckets` ({self.n_buckets}) cannot exceed n_actions ({spec.n_actions}).")

    def _reset(self) -> None:
        super()._reset()
        K = self.spec.n_actions
        self._n_buckets = max(1, round(math.sqrt(K))) if self.n_buckets == "sqrt" else self.n_buckets
        self._last_bucket = -1
        self._last_buckets: Optional[ActionBuckets] = None

    def last_bucket(self) -> int:
        return self._last_bucket

    def _new_model(self, X: np.ndarray, keep_V: bool = False) -> RidgeModel:
        model = RidgeModel(self.dim, self.lambda_, dtype=self._float, keep_V=keep_V)
        self._attach_buckets(model, X)
        return model

    def _attach_buckets(self, model: RidgeModel, X: np.ndarray) -> None:
        """(Re)build the buckets of ``model`` on the actions X from its current estimate."""
        if self._n_buckets is not None:
            K = X.shape[0]
            n_buckets = max(1, round(math.sqrt(K))) if self.n_buckets == "sqrt" else self.n_buckets
            model.buckets = ActionBuckets(K, n_buckets, self.refresh_champions)
            model.buckets.rebuild(X @ model.theta, self.rng)
            model.buckets_on = X
            model.steps_since_rebucket = 0

    def _ucb_action(self, X: np.ndarray, model: RidgeModel, theta: np.ndarray, width_scale: float) -> int:
        """UCB argmax of ``model`` (through its buckets if any)."""
        if model.buckets is not None:
            if model.buckets_on is not X:  # new action set: buckets of the old one are meaningless
                self._attach_buckets(model, X)
            action = model.buckets.select(X, theta, model.V_inv, width_scale, self.rng)
            self._last_bucket = model.buckets.last_bucket
            self._last_buckets = model.buckets  # read by bk.trace for the bucket figure
            return action
        return argmax_random(X @ theta + width_scale * weighted_norms(X, model.V_inv), self.rng)

    def _rebuild_buckets_if_due(self, model: RidgeModel, X: np.ndarray, width_scale: float) -> None:
        if (model.buckets is not None and model.buckets_on is X
                and model.steps_since_rebucket >= self.bucket_rebuild_freq):
            scores = X @ model.theta
            if self.bucket_by_ucb:
                scores = scores + width_scale * weighted_norms(X, model.V_inv)
            model.buckets.rebuild(scores, self.rng)
            model.steps_since_rebucket = 0


class LinUCB(BucketedPolicy):
    """LinUCB / OFUL, with optional random projection (RP), arm buckets (AB) and user models.

    With ``user_model="shared"`` one model serves all users. ``"individual"`` keeps
    one model per user (LinUCB_IND). ``"kmeans"`` keeps one model per user and
    shrinks it towards the centroid of its user cluster, with weight
    ``n_u / (n_u + size of the cluster)``; clusters come from k-means on the user
    estimates, every ``user_recluster_freq`` steps.
    """

    base_name = "LinUCB"
    params = {
        "user_model": ParamSpec("choice", default="shared", choices=("shared", "individual", "kmeans"),
                                doc="One model for all users, one per user, or one per user shrunk to its k-means cluster."),
        "n_user_clusters": ParamSpec("int", default=None, low=1, allow_none=True,
                                     doc="Number of k-means user clusters (user_model='kmeans' only)."),
        "user_recluster_freq": ParamSpec("int", default=2000, low=1, doc="Steps between two k-means of the users."),
    }

    def _check_params(self) -> None:
        super()._check_params()
        if (self.user_model == "kmeans") != (self.n_user_clusters is not None):
            raise ValueError("`n_user_clusters` is required with user_model='kmeans' and only applies to it.")

    @property
    def techniques(self) -> Tuple[str, ...]:
        t = super().techniques
        if self.user_model == "individual":
            return ("IND",) + t
        if self.user_model == "kmeans":
            return ("IND",) + t + ("KMeans",)
        return t

    def _check_spec(self, spec: EnvSpec) -> None:
        super()._check_spec(spec)
        if self.user_model == "kmeans" and self.n_user_clusters > spec.n_users:
            raise ValueError(f"`n_user_clusters` ({self.n_user_clusters}) cannot exceed n_users ({spec.n_users}).")

    def _reset(self) -> None:
        super()._reset()
        self.models: Dict[int, RidgeModel] = {}  # key: 0 (shared) or user_id
        self._n_updates = 0
        self._centroids: Optional[np.ndarray] = None
        self._user_cluster: Dict[int, int] = {}
        self._cluster_sizes: Optional[np.ndarray] = None

    def _model(self, obs: Observation, X: np.ndarray) -> RidgeModel:
        key = 0 if self.user_model == "shared" else obs.user_id
        if key not in self.models:
            self.models[key] = self._new_model(X)
        return self.models[key]

    def _estimate(self, user_id: int, model: RidgeModel) -> np.ndarray:
        """Model estimate, shrunk towards the user's cluster centroid with user_model='kmeans'."""
        if self.user_model != "kmeans" or self._centroids is None or user_id not in self._user_cluster:
            return model.theta
        cluster = self._user_cluster[user_id]
        n_user, n_cluster = max(1, model.n_obs), max(1, int(self._cluster_sizes[cluster]))
        weight = n_user / (n_user + n_cluster)
        return weight * model.theta + (1.0 - weight) * self._centroids[cluster]

    def _predict(self, obs: Observation) -> int:
        X = self._working_features(obs)
        model = self._model(obs, X)
        return self._ucb_action(X, model, self._estimate(obs.user_id, model), self.alpha * self._beta(model.n_obs))

    def _update(self, obs: Observation, action: int, reward: np.ndarray, scalar_reward: float) -> None:
        X = self._working_features(obs)
        model = self._model(obs, X)
        model.update(X[action], scalar_reward)
        self._rebuild_buckets_if_due(model, X, self.alpha * self._beta(model.n_obs))
        self._n_updates += 1
        if (self.user_model == "kmeans" and self._n_updates % self.user_recluster_freq == 0
                and len(self.models) >= self.n_user_clusters):
            self._cluster_users()

    def _cluster_users(self, n_iter: int = 10) -> None:
        """k-means (Lloyd) on the estimates of the users seen so far."""
        users = np.fromiter(self.models, dtype=np.int64)
        thetas = np.stack([self.models[u].theta for u in users]).astype(np.float64)
        k = self.n_user_clusters
        centroids = thetas[self.rng.choice(users.size, size=k, replace=False)]
        for _ in range(n_iter):
            assign = ((thetas[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
            sizes = np.bincount(assign, minlength=k)
            sums = np.zeros_like(centroids)
            np.add.at(sums, assign, thetas)
            new = np.where(sizes[:, None] > 0, sums / np.maximum(sizes, 1)[:, None], centroids)
            converged = np.allclose(new, centroids, atol=1e-6)
            centroids = new
            if converged:
                break
        self._centroids = centroids.astype(self._float)
        self._user_cluster = dict(zip(users.tolist(), assign.tolist()))
        self._cluster_sizes = np.bincount(assign, minlength=k)

    def diagnostics(self) -> Dict[str, Any]:
        if self.user_model == "shared":
            n_clusters = 1
        elif self.user_model == "kmeans" and self._cluster_sizes is not None:
            n_clusters = int((self._cluster_sizes > 0).sum())
        else:
            n_clusters = len(self.models)
        n_buckets = sum(m.buckets.champions.size for m in self.models.values() if m.buckets is not None)
        return {"n_user_clusters": n_clusters, "n_buckets": n_buckets}


class CBRAP(LinUCB):
    """CBRAP (Yu, Lyu and King, AAAI 2017): LinUCB after a Gaussian random projection.

    Projection entries are N(0, 1/m) with m = ``n_components`` (required). Radius of
    Lemma 1 of the AAAI paper, where the noise plus the projection error is
    (sqrt(4/m) + R)-sub-Gaussian::

        beta(n) = (sqrt(4 / m) + R) sqrt(m ln((1 + n L^2 / lambda) / delta)) + sqrt(lambda) S

    A later erratum by the first author (arXiv:1903.08600) revises this analysis.
    """

    base_name = "CBRAP"
    params = {"projection": ParamSpec("choice", default="gaussian", choices=("gaussian",), doc="Fixed to 'gaussian'.")}

    def _check_params(self) -> None:
        super()._check_params()
        if self.n_components is None:
            raise ValueError("CBRAP needs `n_components`.")

    @property
    def techniques(self) -> Tuple[str, ...]:
        return tuple(t for t in super().techniques if t != "RP")  # the projection is part of CBRAP

    def _beta(self, n_obs: int) -> float:
        m, lam = self.dim, self.lambda_
        inner = m * math.log((1.0 + n_obs * self._L ** 2 / lam) / self.delta)
        return (math.sqrt(4.0 / m) + self._R) * math.sqrt(inner) + math.sqrt(lam) * self._S


class SingleModelPolicy(LinearPolicy):
    """One ridge model shared by all users."""

    def _reset(self) -> None:
        super()._reset()
        self.model = RidgeModel(self.dim, self.lambda_, dtype=self._float)

    def _update(self, obs: Observation, action: int, reward: np.ndarray, scalar_reward: float) -> None:
        self.model.update(self._working_features(obs)[action], scalar_reward)


class ConfidenceBall1(SingleModelPolicy):
    """CONFIDENCEBALL_1 (Dani, Hayes and Kakade, 2008), for a finite action set.

    The confidence set is an L1 ball in the V^1/2 geometry, of radius sqrt(k) beta(n).
    Its UCB on x is ``x . theta + alpha * sqrt(k) * beta(n) * ||V^-1/2 x||_inf``.
    beta(n) is the OFUL radius (Abbasi-Yadkori et al., 2011), as in the thesis code,
    not the radius of Dani et al.: the L1 ball of radius sqrt(k) beta(n) contains
    the OFUL ellipsoid, so the confidence level is kept.
    """

    base_name = "ConfidenceBall1"

    def _predict(self, obs: Observation) -> int:
        X = self._working_features(obs)
        eigval, eigvec = np.linalg.eigh(self.model.V_inv)
        V_inv_sqrt = (eigvec * np.sqrt(np.clip(eigval, 0.0, None))) @ eigvec.T
        width = np.abs(X @ V_inv_sqrt).max(axis=1)
        return argmax_random(X @ self.model.theta + self.alpha * math.sqrt(self.dim) * self._beta(self.model.n_obs) * width,
                             self.rng)


class LinTS(SingleModelPolicy):
    """Linear Thompson Sampling (Agrawal and Goyal, ICML 2013), named as in MABWiser.

    Samples ``theta ~ N(theta_hat, (alpha v_t)^2 V^-1)`` and plays the best action
    for the sample, with ``v_t = R sqrt(9 k ln(t / delta))`` at round t (1-based):
    the scale of the final arXiv version (1209.3352), for an unknown horizon. The
    paper assumes features and theta of norm at most 1, and lambda = 1.
    """

    base_name = "LinTS"

    def _scale(self) -> float:
        t = self.model.n_obs + 1
        return self._R * math.sqrt(9.0 * self.dim * math.log(t / self.delta))

    def _predict(self, obs: Observation) -> int:
        X = self._working_features(obs)
        try:
            factor = np.linalg.cholesky(self.model.V_inv)
        except np.linalg.LinAlgError:  # V_inv numerically not positive definite (float32)
            eigval, eigvec = np.linalg.eigh(self.model.V_inv)
            factor = eigvec * np.sqrt(np.clip(eigval, 0.0, None))
        sample = self.model.theta + self.alpha * self._scale() * (factor @ self.rng.standard_normal(self.dim))
        return argmax_random(X @ sample, self.rng)


class LinGreedy(SingleModelPolicy):
    """Greedy on the ridge estimate, uniform exploration with probability epsilon (as MABWiser's LinGreedy)."""

    base_name = "LinGreedy"
    params = {"epsilon": ParamSpec("float", default=0.1, low=0.0, high=1.0, doc="Exploration probability.")}
    arbitrary_defaults = {"epsilon": "no theoretical value"}
    default_search_params = ("epsilon", "lambda_")

    def _predict(self, obs: Observation) -> int:
        X = self._working_features(obs)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(X.shape[0]))
        return argmax_random(X @ self.model.theta, self.rng)
