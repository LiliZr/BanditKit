"""Classical MABs on real data: each action returns rewards actually observed for it.

Real rewards follow no textbook distribution: skewed, heavy-tailed, discrete,
zero-inflated... These EXAMPLES test policies whose guarantees assume Bernoulli
or Gaussian rewards (Thompson Sampling, KL-UCB) outside their comfort zone. The
environment is an :class:`~banditkit.EmpiricalMAB`: a step resamples one observed
reward of the chosen action, so the true means are the empirical means.

Rewards are mapped to [0, 1] (``scale``), as bounded-reward policies expect.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

from .._validation import check_choice, check_float, check_int
from ..envs import EmpiricalMAB

__all__ = ["fetch_taxis", "taxis_mab", "movielens_mab", "steam_mab", "rewards_to_mab", "TAXIS_URL"]

#: NYC taxi trips of March 2019 (6,433 trips; from the NYC TLC records, as packaged by seaborn).
TAXIS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"


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


def fetch_taxis(data_home: Union[str, Path, None] = None, download_if_missing: bool = True):
    """NYC taxi trips (pandas DataFrame), downloaded once into ``data_home``.

    Columns include pickup and dropoff times, distance, fare, tip, total, payment,
    pickup_zone, pickup_borough.
    """
    import pandas as pd

    from .movielens import get_data_home

    path = get_data_home(data_home) / "taxis.csv"
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"{path} not found and download_if_missing=False.")
        import urllib.request

        try:
            urllib.request.urlretrieve(TAXIS_URL, path)
        except OSError as exc:
            raise OSError(f"Download of {TAXIS_URL} failed ({exc}). Download it by hand into {path}.") from None
    return pd.read_csv(path)


def taxis_mab(reward: str = "fare", n_arms: int = 10, exclude: Sequence[str] = ("JFK Airport", "LaGuardia Airport"),
              min_trips: int = 50, scale: Optional[str] = "quantile", data_home: Union[str, Path, None] = None
              ) -> EmpiricalMAB:
    """Example: where should a taxi driver wait? Actions are pickup zones.

    A step returns one real trip from the chosen zone. Fares are right-skewed with
    a spike at the flat airport fare; tip rates are zero-inflated (cash tips are
    not recorded). The default leaves out the airports, whose fares dwarf the others.

    Parameters
    ----------
    reward : {"fare", "total", "tip_rate"}
        Fare, total paid, or tip / fare.
    n_arms : int
        The ``n_arms`` busiest zones (after ``exclude`` and ``min_trips``).
    exclude : sequence of str
        Zones left out.
    min_trips : int
        Minimum number of trips of a zone.
    scale : {"quantile", "max", None}
        See :func:`rewards_to_mab`.
    """
    reward = check_choice("reward", reward, ("fare", "total", "tip_rate"))
    n_arms = check_int("n_arms", n_arms, low=2)
    min_trips = check_int("min_trips", min_trips, low=1)
    df = fetch_taxis(data_home)
    df = df[~df["pickup_zone"].isin(list(exclude))].dropna(subset=["pickup_zone", "fare"])
    df = df[df["fare"] > 0]
    values = df["tip"] / df["fare"] if reward == "tip_rate" else df[reward]
    counts = df["pickup_zone"].value_counts()
    zones = counts[counts >= min_trips].index[:n_arms]
    if len(zones) < 2:
        raise ValueError(f"Fewer than 2 zones with {min_trips} trips or more.")
    return rewards_to_mab({z: values[df["pickup_zone"] == z].to_numpy() for z in zones}, scale=scale)


def movielens_mab(variant: str = "small", arms: str = "genres", n_arms: Optional[int] = None, min_ratings: int = 50,
                  data_home: Union[str, Path, None] = None) -> EmpiricalMAB:
    """Example: MovieLens ratings as rewards (discrete, skewed towards high stars).

    ``arms="genres"``: an action is a genre, a step returns one rating of a movie
    of that genre. ``arms="movies"``: the ``n_arms`` most rated movies. Ratings are
    divided by 5.
    """
    from .movielens import _genre_table, fetch_movielens

    arms = check_choice("arms", arms, ("genres", "movies"))
    min_ratings = check_int("min_ratings", min_ratings, low=1)
    ratings, movies = fetch_movielens(variant, data_home)
    if arms == "genres":
        joined = ratings.merge(_genre_table(movies), on="item_id")
        groups = {g: r.to_numpy() for g, r in joined.groupby("category")["rating"] if len(r) >= min_ratings}
        if n_arms is not None:
            keep = sorted(groups, key=lambda g: -len(groups[g]))[:check_int("n_arms", n_arms, low=2)]
            groups = {g: groups[g] for g in keep}
    else:
        counts = ratings["item_id"].value_counts()
        top = counts[counts >= min_ratings].index[:check_int("n_arms", n_arms or 20, low=2)]
        titles = movies.set_index("item_id")["title"]
        groups = _unique_names((titles.get(m, m), ratings.loc[ratings["item_id"] == m, "rating"].to_numpy())
                               for m in top)
    mab = rewards_to_mab(groups, scale=None)
    return EmpiricalMAB([s / 5.0 for s in mab._samples], action_names=mab.action_names, reward_range=1.0)


def steam_mab(path: Union[str, Path], n_arms: int = 20, min_reviews: int = 100, quantile: float = 0.99
              ) -> EmpiricalMAB:
    """Example: hours played per game on Steam (heavy-tailed), from the Kaggle files.

    Actions are the ``n_arms`` most reviewed games; a step returns the hours played
    by one reviewer, divided by the ``quantile`` of all hours and clipped to [0, 1].
    """
    import pandas as pd

    from .steam import load_steam

    n_arms = check_int("n_arms", n_arms, low=2)
    min_reviews = check_int("min_reviews", min_reviews, low=1)
    load_steam(path)  # checks the folder and its files
    rec = pd.read_csv(Path(path).expanduser() / "recommendations.csv", usecols=["app_id", "hours"]).dropna()
    counts = rec["app_id"].value_counts()
    games = counts[counts >= min_reviews].index[:n_arms]
    titles = {}
    games_csv = Path(path).expanduser() / "games.csv"
    if games_csv.exists():
        titles = pd.read_csv(games_csv, usecols=["app_id", "title"]).set_index("app_id")["title"].to_dict()
    groups = _unique_names((titles.get(g, g), rec.loc[rec["app_id"] == g, "hours"].to_numpy()) for g in games)
    return rewards_to_mab(groups, scale="quantile", quantile=quantile)
