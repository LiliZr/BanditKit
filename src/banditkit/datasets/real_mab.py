"""Classical MABs on real data: each action returns rewards actually observed for it.

Real rewards follow no textbook distribution: discrete, skewed, bounded. These
EXAMPLES follow the MovieLens experiments of Gupta, Chaudhari, Joshi and Yagan
("Multi-Armed Bandits with Correlated Arms", IEEE Transactions on Information
Theory, 2021): a step returns one observed rating of the chosen arm. The
environment is an :class:`~banditkit.EmpiricalMAB`, so the true means are the
empirical means. :func:`rewards_to_mab` does the same on any data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from .._validation import check_choice, check_float, check_int
from ..envs import EmpiricalMAB

__all__ = ["movielens_mab", "rewards_to_mab"]


def _unique_names(pairs):
    """{name: values} from (name, values) pairs, adding a suffix to repeated names (two games with one title)."""
    out = {}
    for name, values in pairs:
        key, k = str(name), 2
        while key in out:
            key, k = f"{name} ({k})", k + 1
        out[key] = values
    return out


def rewards_to_mab(groups: Dict[str, np.ndarray], scale: Optional[str] = "quantile", quantile: float = 0.99,
                   ) -> EmpiricalMAB:
    """EmpiricalMAB from observed rewards per action (any real data).

    Parameters
    ----------
    groups : dict {action name: 1-D array of observed rewards}
    scale : {"quantile", "max", None}
        Map rewards to [0, 1]: divide by the ``quantile`` of all rewards and clip
        ("quantile", robust to outliers), divide by the maximum ("max"), or keep them (None).
        Rewards must be >= 0 for the two scalings.
    quantile : float
        Quantile used by ``scale="quantile"``.
    """
    scale = check_choice("scale", scale, ("quantile", "max", None))
    quantile = check_float("quantile", quantile, low=0.0, high=1.0, low_inclusive=False)
    names = list(groups)
    if len(names) < 2:
        raise ValueError(f"A bandit needs at least 2 actions, got {len(names)}.")
    values = [np.asarray(groups[n], dtype=np.float64).ravel() for n in names]
    if any(v.size == 0 for v in values):
        raise ValueError("Every action needs at least one observed reward.")
    if scale is not None:
        pooled = np.concatenate(values)
        if pooled.min() < 0:
            raise ValueError("`scale` needs rewards >= 0; shift them first or use scale=None.")
        top = np.quantile(pooled, quantile) if scale == "quantile" else pooled.max()
        top = top if top > 0 else 1.0
        values = [np.clip(v / top, 0.0, 1.0) for v in values]
    return EmpiricalMAB(values, action_names=names, reward_range=1.0 if scale is not None else None)


def movielens_mab(variant: str = "small", arms: str = "genres", n_arms: Optional[int] = None,
                  data_home: Union[str, Path, None] = None, seed: int = 0) -> EmpiricalMAB:
    """Example: MovieLens ratings as rewards, as in Gupta et al. (IEEE Trans. Inf. Theory, 2021).

    ``arms="genres"``: an arm is a genre. A movie with several genres is given one
    of them at random (seeded by ``seed``), as in the paper; a step returns one
    rating of a movie of that genre. ``arms="movies"``: the ``n_arms`` most rated
    movies (50 in the paper); a step returns one of their ratings. Ratings are
    divided by 5, so rewards lie in [0, 1].

    The paper splits the users in two halves and samples rewards from the second;
    here every rating is used.
    """
    from .movielens import fetch_movielens

    arms = check_choice("arms", arms, ("genres", "movies"))
    ratings, movies = fetch_movielens(variant, data_home)
    if arms == "genres":
        rng = np.random.default_rng(seed)
        movies = movies[movies["genres"].map(len) > 0]
        genre = {m: g[rng.integers(len(g))] for m, g in zip(movies["item_id"], movies["genres"])}
        rated = ratings.assign(genre=ratings["item_id"].map(genre)).dropna(subset=["genre"])
        groups = {g: r.to_numpy() for g, r in rated.groupby("genre")["rating"]}
        if n_arms is not None:
            keep = sorted(groups, key=lambda g: -len(groups[g]))[:check_int("n_arms", n_arms, low=2)]
            groups = {g: groups[g] for g in keep}
    else:
        top = ratings["item_id"].value_counts().index[:check_int("n_arms", 50 if n_arms is None else n_arms, low=2)]
        titles = movies.set_index("item_id")["title"]
        groups = _unique_names((titles.get(m, m), ratings.loc[ratings["item_id"] == m, "rating"].to_numpy())
                               for m in top)
    return EmpiricalMAB([v / 5.0 for v in groups.values()], action_names=list(groups), reward_range=1.0)
