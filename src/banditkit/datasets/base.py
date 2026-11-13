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

``item_features`` is accepted as another name for ``action_features``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Union

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
        Item features.
    user_thetas : ndarray (n_users, reward_dim, n_features)
        Parameter of each user, one row per objective.
    user_stream : ndarray (n_interactions,) or None
        Users in the order of the interactions, or None for uniform arrivals.
    noise_std : ndarray (reward_dim,)
        Standard deviation of the Gaussian noise of the environment.
    objectives : tuple of str
        Name of each objective.
    item_ids, user_ids : ndarray
        Original identifiers of the items and users (row order of the arrays above).
    """

    action_features: np.ndarray
    user_thetas: np.ndarray
    user_stream: Optional[np.ndarray]
    noise_std: np.ndarray
    objectives: Tuple[str, ...] = ("rating",)
    item_ids: Optional[np.ndarray] = None
    user_ids: Optional[np.ndarray] = None
    info: dict = field(default_factory=dict)

    def to_env(self, noise_std: Any = None, n_candidates: Optional[int] = 20) -> MultiUserLinearEnv:
        """The environment. ``noise_std`` overrides the stored noise level.

        ``n_candidates`` items are offered at each step (20 in Li et al., 2019 and
        Wang et al., 2023; None: every item).
        """
        n_candidates = None if n_candidates is None else min(n_candidates, self.action_features.shape[0])
        return MultiUserLinearEnv(self.action_features, self.user_thetas,
                                  noise_std=self.noise_std if noise_std is None else noise_std,
                                  user_stream=self.user_stream, n_candidates=n_candidates)

    def save(self, path: Union[str, Path]) -> Path:
        """Save as an .npz file readable by :func:`load_npz` and :meth:`load`."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, action_features=self.action_features, user_thetas=self.user_thetas,
                            user_stream=np.array([], dtype=np.int64) if self.user_stream is None else self.user_stream,
                            noise_std=self.noise_std,
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
            stream = data["user_stream"] if "user_stream" in data and data["user_stream"].size else None
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


def ratings_to_linear(ratings, n_users: int = 1000, n_items: int = 1000, n_features: int = 20,
                      feedback: str = "rating", like_threshold: float = 3.0, arrivals: str = "uniform",
                      noise_std: float = 0.1, seed: int = 0) -> LinearDataset:
    """Example preprocessing of ratings into a multi-user linear bandit.

    Based on the real-data experiments of SCLUB (Li, Chen, Li and Leung, IJCAI 2019),
    with the binary variant of Wang, Xie, Liu, Li and Lui (NeurIPS 2023):

    1. Keep the ``n_items`` most rated items and the ``n_users`` users who rate most.
    2. Build the users x items matrix: ratings (``feedback="rating"``, Li et al.)
       or 1 if rating > ``like_threshold`` else 0 (``feedback="binary"``, Wang et al.).
       Missing entries are 0.
    3. Truncated SVD of rank d - 1, d = ``n_features``: users U sqrt(S), items V sqrt(S).
    4. Every vector v becomes ``(v / (sqrt(2) ||v||), 1 / sqrt(2))`` (Eq. 3 of Li et al.),
       so that every mean reward lies in [0, 1].

    The environment offers 20 random items per step (``LinearDataset.to_env``), as in
    both papers. Choices the papers do not state: how S is split between users and
    items, the noise level, and the order of user arrivals.

    Parameters
    ----------
    ratings : DataFrame
        Columns ``user_id``, ``item_id``, ``rating`` and optionally ``timestamp``.
    n_users, n_items : int
        Number of most active users and most rated items kept (1000 in both papers).
    n_features : int
        Dimension d of the features (20 in Li et al.).
    feedback : {"rating", "binary"}
        Matrix factorized: the ratings, or ratings binarized at ``like_threshold``.
    like_threshold : float
        A rating above it counts as positive feedback (3 in Wang et al.).
    arrivals : {"uniform", "data"}
        Users arrive uniformly at random, or in the order of their ratings
        (by ``timestamp`` if present).
    noise_std : float
        Standard deviation of the Gaussian reward noise (not stated in the papers).
    seed : int
        Seed of the SVD initialization.
    """
    import pandas as pd
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds

    ratings = _check_ratings(ratings)
    n_users = check_int("n_users", n_users, low=2)
    n_items = check_int("n_items", n_items, low=2)
    n_features = check_int("n_features", n_features, low=2)
    if feedback not in ("rating", "binary"):
        raise ValueError(f"`feedback` must be 'rating' or 'binary', got {feedback!r}.")
    if arrivals not in ("uniform", "data"):
        raise ValueError(f"`arrivals` must be 'uniform' or 'data', got {arrivals!r}.")
    noise_std = check_float("noise_std", noise_std, low=0.0)

    items = ratings["item_id"].value_counts().index[:n_items]
    ratings = ratings[ratings["item_id"].isin(items)]
    users = ratings["user_id"].value_counts().index[:n_users]
    ratings = ratings[ratings["user_id"].isin(users)]
    if "timestamp" in ratings.columns:
        ratings = ratings.sort_values("timestamp", kind="stable")
    user_codes, user_ids = pd.factorize(ratings["user_id"])
    item_codes, item_ids = pd.factorize(ratings["item_id"])
    rank = n_features - 1
    if rank >= min(user_ids.size, item_ids.size):
        raise ValueError(f"`n_features` ({n_features}) must be at most min(n_users, n_items) = "
                         f"{min(user_ids.size, item_ids.size)} after filtering.")

    values = ratings["rating"].to_numpy(dtype=float)
    if feedback == "binary":
        values = (values > like_threshold).astype(float)
    matrix = csr_matrix((values, (user_codes, item_codes)), shape=(user_ids.size, item_ids.size))
    v0 = np.random.default_rng(seed).uniform(-1, 1, size=min(matrix.shape))
    u, sing, vt = svds(matrix, k=rank, v0=v0)
    root = np.sqrt(sing)

    def lift(v: np.ndarray) -> np.ndarray:  # Eq. 3 of Li et al. (2019)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return np.c_[v / (math.sqrt(2.0) * np.where(norms > 0, norms, 1.0)), np.full(len(v), 1.0 / math.sqrt(2.0))]

    features, thetas = lift(vt.T * root), lift(u * root)
    stream = user_codes.astype(np.int64) if arrivals == "data" else None
    return LinearDataset(action_features=features, user_thetas=thetas[:, None, :], user_stream=stream,
                         noise_std=np.array([noise_std]), objectives=(feedback,),
                         item_ids=np.asarray(item_ids), user_ids=np.asarray(user_ids),
                         info={"n_ratings": int(values.size), "density": values.size / np.prod(matrix.shape)})
