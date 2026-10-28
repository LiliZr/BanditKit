"""From raw interaction data to bandit environments, plus the documented .npz format.

Any preprocessing can be plugged in: what matters is that its output matches the
inputs of the environments (``bk.make_env`` documents the shapes). Two routes:

* call ``bk.make_env(...)`` (or an environment class) on your own arrays;
* save them in an ``.npz`` file with the keys below and use :func:`load_npz`.

============================================================  ==============================
Keys of the .npz file                                         Environment
============================================================  ==============================
``action_features`` (K, d), ``user_thetas`` (U, d) or         multi-user linear bandit
(U, D, d), optional ``user_stream`` (T,), ``noise_std``
``action_features`` (K, d), ``theta`` (d,) or (D, d),         linear bandit
optional ``noise_std``
``means`` (K,) or (K, D), optional ``noise_std``              Gaussian (multi-objective) MAB
``relevance`` (n_members, K), optional ``objectives``         group recommendation MAB
============================================================  ==============================

Files written by the Scaling-Linear-Bandits preprocessing (``item_features``,
``user_thetas``, ``user_stream``, ``noise_std``) are read as well.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from .._validation import check_array, check_float, check_int
from ..envs import GaussianMAB, GroupRecommendationMAB, LinearEnv, MultiUserLinearEnv
from ..envs.base import Environment

__all__ = ["LinearDataset", "load_npz", "ratings_to_linear", "filter_min_counts"]

RATING_COLUMNS = ("user_id", "item_id", "rating")


@dataclass
class LinearDataset:
    """A multi-user linear bandit built from data: the arrays of :class:`~banditkit.MultiUserLinearEnv`.

    Attributes
    ----------
    action_features : ndarray (n_actions, n_features)
        Item features, rows of norm 1.
    user_thetas : ndarray (n_users, reward_dim, n_features)
        Parameter of each user, one row per objective.
    user_stream : ndarray (n_interactions,)
        Users in the order of the interactions (chronological when timestamps exist).
    noise_std : ndarray (reward_dim,)
        Standard deviation of the residuals of the user fits, used as the noise level.
    objectives : tuple of str
        Name of each objective.
    item_ids, user_ids : ndarray
        Original identifiers of the items and users (row order of the arrays above).
    """

    action_features: np.ndarray
    user_thetas: np.ndarray
    user_stream: np.ndarray
    noise_std: np.ndarray
    objectives: Tuple[str, ...] = ("rating",)
    item_ids: Optional[np.ndarray] = None
    user_ids: Optional[np.ndarray] = None
    info: dict = field(default_factory=dict)

    def to_env(self, noise_std: Any = None) -> MultiUserLinearEnv:
        """Environment that replays the user stream. ``noise_std`` overrides the fitted noise level."""
        return MultiUserLinearEnv(self.action_features, self.user_thetas,
                                  noise_std=self.noise_std if noise_std is None else noise_std,
                                  user_stream=self.user_stream)

    def save(self, path: Union[str, Path]) -> Path:
        """Save as an .npz file readable by :func:`load_npz` and :meth:`load`."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, action_features=self.action_features, user_thetas=self.user_thetas,
                            user_stream=self.user_stream, noise_std=self.noise_std,
                            objectives=np.array(self.objectives),
                            item_ids=np.array([] if self.item_ids is None else self.item_ids),
                            user_ids=np.array([] if self.user_ids is None else self.user_ids))
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LinearDataset":
        with np.load(path, allow_pickle=False) as data:
            features = data["action_features"] if "action_features" in data else data["item_features"]
            thetas = data["user_thetas"]
            thetas = thetas[:, None, :] if thetas.ndim == 2 else thetas
            stream = data["user_stream"] if "user_stream" in data else np.arange(thetas.shape[0])
            noise = np.broadcast_to(data["noise_std"] if "noise_std" in data else 0.1, (thetas.shape[1],))
            objectives = tuple(data["objectives"].tolist()) if "objectives" in data else \
                tuple(f"objective_{j}" for j in range(thetas.shape[1]))
            item_ids = data["item_ids"] if "item_ids" in data and data["item_ids"].size else None
            user_ids = data["user_ids"] if "user_ids" in data and data["user_ids"].size else None
        return cls(features, thetas, stream, np.array(noise, dtype=float), objectives, item_ids, user_ids)


def load_npz(path: Union[str, Path], **env_kwargs: Any) -> Environment:
    """Environment described by an .npz file (keys in the module docstring).

    ``env_kwargs`` are passed to the environment (e.g. ``noise_std``).
    """
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        arrays = {k: data[k] for k in keys}
    noise = arrays.get("noise_std")
    if "user_thetas" in keys:
        dataset = LinearDataset.load(path)
        return dataset.to_env(**env_kwargs)
    if "theta" in keys:
        features = arrays.get("action_features", arrays.get("item_features"))
        if features is None:
            raise KeyError(f"{path}: a linear problem needs `action_features` next to `theta`.")
        return LinearEnv(features, arrays["theta"], noise_std=0.1 if noise is None else noise, **env_kwargs)
    if "means" in keys:
        return GaussianMAB(arrays["means"], noise_std=1.0 if noise is None else noise, **env_kwargs)
    if "relevance" in keys:
        objectives = tuple(arrays["objectives"].tolist()) if "objectives" in keys else ("relevance", "agreement")
        return GroupRecommendationMAB(arrays["relevance"], objectives=objectives)
    raise KeyError(f"{path}: no known layout. Found keys {sorted(keys)}; "
                   "expected `user_thetas`, `theta`, `means` or `relevance` (see banditkit.datasets).")


# ------------------------------------------------------------------ ratings
def _check_ratings(ratings) -> "Any":
    import pandas as pd

    if not isinstance(ratings, pd.DataFrame):
        raise TypeError("`ratings` must be a pandas DataFrame with columns user_id, item_id, rating [, timestamp].")
    missing = [c for c in RATING_COLUMNS if c not in ratings.columns]
    if missing:
        raise ValueError(f"`ratings` lacks the columns {missing}; it has {list(ratings.columns)}.")
    columns = list(RATING_COLUMNS) + (["timestamp"] if "timestamp" in ratings.columns else [])
    out = ratings[columns].dropna()
    check_array("ratings['rating']", out["rating"].to_numpy())
    if out.empty:
        raise ValueError("`ratings` is empty after dropping missing values.")
    return out


def filter_min_counts(ratings, min_user_ratings: int = 1, min_item_ratings: int = 1, max_rounds: int = 20):
    """Keep users and items with enough ratings, repeating until both thresholds hold (k-core)."""
    for _ in range(max_rounds):
        item_counts = ratings["item_id"].map(ratings["item_id"].value_counts())
        user_counts = ratings["user_id"].map(ratings["user_id"].value_counts())
        keep = (item_counts >= min_item_ratings) & (user_counts >= min_user_ratings)
        if keep.all():
            break
        ratings = ratings[keep]
    if ratings.empty:
        raise ValueError(f"No rating left with min_user_ratings={min_user_ratings}, min_item_ratings={min_item_ratings}.")
    return ratings


def _svd_item_embeddings(user_idx, item_idx, values, n_users: int, n_items: int, rank: int, seed: int) -> np.ndarray:
    """Item embeddings of rank ``rank`` from the truncated SVD of the (users x items) matrix of centered ratings."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds

    matrix = csr_matrix((values, (user_idx, item_idx)), shape=(n_users, n_items))
    rank = max(1, min(rank, min(n_users, n_items) - 1))
    v0 = np.random.default_rng(seed).uniform(-1, 1, size=min(n_users, n_items))
    _, sing, vt = svds(matrix, k=rank, v0=v0)
    return vt.T * np.sqrt(sing)  # (n_items, rank)


def ratings_to_linear(ratings, item_features=None, n_features: int = 100,
                      objectives: Sequence[str] = ("rating", "like"), like_threshold: float = 4.0,
                      min_user_ratings: int = 20, min_item_ratings: int = 20, lambda_: float = 1.0,
                      seed: int = 0) -> LinearDataset:
    """Example preprocessing of ratings into a multi-user linear bandit (as in the Scaling-Linear-Bandits paper).

    One reasonable pipeline among many: replace any step, or the whole function, as long
    as the result has the arrays of :class:`LinearDataset`.

    1. Keep users and items with enough ratings (:func:`filter_min_counts`).
    2. Item features: the given side features, rating statistics (mean, std,
       log count), then a truncated SVD of the centered rating matrix to reach
       ``n_features``; columns are standardized, a bias column is added, and rows
       are scaled to norm 1.
    3. One ridge regression per user on all their ratings gives ``user_thetas``.
       Objective ``"rating"``: the rating mapped to [0, 1]; ``"like"``: 1 if
       rating >= ``like_threshold``, else 0.
    4. The user stream follows the order of the ratings (by ``timestamp`` if present).

    Parameters
    ----------
    ratings : DataFrame
        Columns ``user_id``, ``item_id``, ``rating`` and optionally ``timestamp``.
    item_features : DataFrame, optional
        Numeric side features indexed by ``item_id`` (e.g. one-hot genres, year).
    n_features : int
        Dimension d of the item features (including the side features and statistics).
    objectives : sequence of {"rating", "like"}
        Objectives of the reward vector, in this order.
    like_threshold : float
        Rating from which an interaction counts as a like.
    min_user_ratings, min_item_ratings : int
        Minimum number of ratings to keep a user / an item.
    lambda_ : float
        Ridge regularization of the user fits.
    seed : int
        Seed of the SVD initialization.
    """
    import pandas as pd

    ratings = _check_ratings(ratings)
    n_features = check_int("n_features", n_features, low=2)
    lambda_ = check_float("lambda_", lambda_, low=0.0, low_inclusive=False)
    objectives = tuple(objectives)
    if not objectives or any(o not in ("rating", "like") for o in objectives):
        raise ValueError(f"`objectives` must be a non-empty sequence of 'rating' and 'like', got {objectives}.")
    ratings = filter_min_counts(ratings, check_int("min_user_ratings", min_user_ratings, low=1),
                                check_int("min_item_ratings", min_item_ratings, low=1))
    if "timestamp" in ratings.columns:
        ratings = ratings.sort_values("timestamp", kind="stable")
    user_codes, user_ids = pd.factorize(ratings["user_id"])
    item_codes, item_ids = pd.factorize(ratings["item_id"])
    rating = ratings["rating"].to_numpy(dtype=float)
    n_users, n_items = user_ids.size, item_ids.size

    # Item features: side features + rating statistics + SVD embeddings.
    blocks = []
    if item_features is not None:
        side = item_features.reindex(item_ids).fillna(0.0)
        blocks.append(side.select_dtypes("number").to_numpy(dtype=float))
    stats = pd.DataFrame({"item": item_codes, "rating": rating}).groupby("item")["rating"].agg(["mean", "std", "size"])
    blocks.append(np.c_[stats["mean"], stats["std"].fillna(0.0), np.log1p(stats["size"])])
    used = sum(b.shape[1] for b in blocks) + 1  # + bias column
    if used >= n_features:
        raise ValueError(f"`n_features` ({n_features}) must exceed the {used} side, statistic and bias columns.")
    user_means = np.bincount(user_codes, weights=rating, minlength=n_users) / np.bincount(user_codes, minlength=n_users)
    blocks.append(_svd_item_embeddings(user_codes, item_codes, rating - user_means[user_codes], n_users, n_items,
                                       n_features - used, seed))
    X = np.hstack(blocks)
    std = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)
    X = np.c_[X, np.ones(n_items)]
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    # Targets of each objective, then one ridge regression per user.
    low, high = rating.min(), rating.max()
    targets = {"rating": (rating - low) / (high - low if high > low else 1.0),
               "like": (rating >= like_threshold).astype(float)}
    Y = np.stack([targets[o] for o in objectives], axis=1)
    order = np.argsort(user_codes, kind="stable")
    bounds = np.searchsorted(user_codes[order], np.arange(n_users + 1))
    d = X.shape[1]
    thetas = np.empty((n_users, len(objectives), d))
    residuals = np.empty_like(Y)
    eye = lambda_ * np.eye(d)
    for u in range(n_users):
        rows = order[bounds[u]:bounds[u + 1]]
        Xu, Yu = X[item_codes[rows]], Y[rows]
        thetas[u] = np.linalg.solve(Xu.T @ Xu + eye, Xu.T @ Yu).T
        residuals[rows] = Yu - Xu @ thetas[u].T
    return LinearDataset(action_features=X, user_thetas=thetas, user_stream=user_codes.astype(np.int64),
                         noise_std=residuals.std(axis=0), objectives=objectives,
                         item_ids=np.asarray(item_ids), user_ids=np.asarray(user_ids),
                         info={"n_ratings": int(rating.size), "rating_range": (float(low), float(high))})
