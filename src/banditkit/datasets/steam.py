"""Steam ("Game Recommendations on Steam", Kaggle): parsing and an EXAMPLE preprocessing.

The dataset needs a Kaggle account, so it is not downloaded automatically:
download it from https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam
and extract it into a folder. Files used:

* ``recommendations.csv`` (required): app_id, is_recommended, user_id, date...
* ``games.csv`` (optional): app_id, win, mac, linux, positive_ratio, user_reviews, price_final, steam_deck...
* ``games_metadata.json`` (optional, needed for tags): app_id and tags (list of str).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from .base import LinearDataset, ratings_to_linear

__all__ = ["load_steam", "steam_linear"]

_GAME_NUMERIC = ["win", "mac", "linux", "steam_deck", "positive_ratio", "user_reviews", "price_final"]


def load_steam(path: Union[str, Path]):
    """Read the extracted Kaggle folder.

    Returns
    -------
    ratings : DataFrame
        Columns ``user_id``, ``item_id`` (= app_id), ``rating`` (1 if recommended, else 0),
        ``timestamp`` (from ``date``, if present).
    games : DataFrame or None
        Numeric game features indexed by item_id (log of user_reviews), if games.csv exists.
    tags : DataFrame or None
        Columns ``item_id`` and ``category`` (one row per tag), if games_metadata.json exists.
    """
    import pandas as pd

    root = Path(path).expanduser()
    if not (root / "recommendations.csv").exists():
        raise FileNotFoundError(f"{root / 'recommendations.csv'} not found; extract the Kaggle archive into {root}.")
    rec = pd.read_csv(root / "recommendations.csv")
    missing = {"user_id", "app_id", "is_recommended"} - set(rec.columns)
    if missing:
        raise ValueError(f"recommendations.csv lacks the columns {sorted(missing)}.")
    ratings = pd.DataFrame({"user_id": rec["user_id"], "item_id": rec["app_id"],
                            "rating": rec["is_recommended"].astype(float)})
    if "date" in rec.columns:
        ratings["timestamp"] = pd.to_datetime(rec["date"], errors="coerce").astype("int64")

    games = None
    if (root / "games.csv").exists():
        raw = pd.read_csv(root / "games.csv").set_index("app_id")
        games = raw[[c for c in _GAME_NUMERIC if c in raw.columns]].astype(float)
        if "user_reviews" in games:
            games["user_reviews"] = np.log1p(games["user_reviews"])
        games.index.name = "item_id"

    tags = None
    if (root / "games_metadata.json").exists():
        try:
            meta = pd.read_json(root / "games_metadata.json", lines=True)
        except ValueError:
            meta = pd.read_json(root / "games_metadata.json")
        if not {"app_id", "tags"} <= set(meta.columns):
            raise ValueError(f"games_metadata.json needs app_id and tags, found {list(meta.columns)}.")
        tags = (meta[["app_id", "tags"]].explode("tags").dropna()
                .rename(columns={"app_id": "item_id", "tags": "category"}).reset_index(drop=True))
    return ratings, games, tags


def steam_linear(path: Union[str, Path], n_users: int = 1000, n_items: int = 1000, n_features: int = 20,
                 arrivals: str = "uniform", noise_std: float = 0.1, seed: int = 0) -> LinearDataset:
    """Example preprocessing: Steam as a multi-user linear bandit (games are actions).

    Built with :func:`~banditkit.datasets.ratings_to_linear` and ``feedback="binary"``:
    the method of Li et al. (IJCAI 2019) on the binary matrix of Wang et al.
    (NeurIPS 2023), where a recommended game is 1 and any other entry is 0.
    These papers use MovieLens and Yelp; applying the method to Steam is our choice.
    """
    ratings, _, _ = load_steam(path)
    return ratings_to_linear(ratings, n_users=n_users, n_items=n_items, n_features=n_features, feedback="binary",
                             like_threshold=0.5, arrivals=arrivals, noise_std=noise_std, seed=seed)
