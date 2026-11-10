"""Single-user linear bandits of Izri, Groz and Maniu (IEEE BigData 2025).

"Implementing Efficient Linear Bandits Via Sketches and Random Projections",
IEEE International Conference on Big Data, 2025, pp. 956-965,
doi:10.1109/BigData66926.2025.11401692. Code and processed data:
https://github.com/LiliZr/Stochastic-linear-bandits (downloaded once, a few MB for Steam).

Each instance is one user, drawn with a seed as in the paper's runs (one user per
run). Features are high-dimensional (one-hot tags or one-hot users), which is the
setting of sketches and random projections.

* Steam: the features of a game are its metadata, its rating class and its tags
  (one-hot). The user's parameter is the least-squares fit (no intercept) of
  "recommended" on the games they reviewed. The actions are the ``n_actions`` most
  reviewed games the user did not review. Mean rewards are min-max normalized over
  the actions (``LinearEnv(normalize=True)``).
* MovieLens: the actions are the movies the user rated. The features of a movie are
  its year, genres and rating statistics, plus the users who rated it (one-hot).
  The mean reward is the user's rating / 5 (no linear model).

Rows are scaled to norm 1, as in the paper. ``paper_code=True`` rebuilds the Steam
instances exactly as the repository code does; see :func:`steam_single_user`.
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import List, Union

import numpy as np

from .._validation import check_choice, check_float, check_int
from ..envs import LinearEnv

__all__ = ["BIGDATA2025_URL", "fetch_bigdata2025", "steam_single_user", "movielens_single_user",
           "bigdata2025_instances"]

#: Folder of the processed data in the paper's repository.
BIGDATA2025_URL = "https://raw.githubusercontent.com/LiliZr/Stochastic-linear-bandits/main/data/"
_FILES = {"steam": ("steam/data/steam_actions.zip", "steam/data/steam_links.zip"),
          "movielens": ("movielens_25m/data/movielens.zip",)}


def fetch_bigdata2025(name: str, data_home: Union[str, Path, None] = None, download_if_missing: bool = True):
    """Processed data of the paper, as pandas DataFrames, downloaded once into ``data_home``.

    Returns (game features, reviews) for ``"steam"`` and (ratings with movie features,) for ``"movielens"``.
    """
    import pandas as pd

    from .movielens import get_data_home

    name = check_choice("name", name, tuple(_FILES))
    folder = get_data_home(data_home) / "bigdata2025"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for rel in _FILES[name]:
        path = folder / Path(rel).name
        if not path.exists():
            if not download_if_missing:
                raise FileNotFoundError(f"{path} not found and download_if_missing=False.")
            url = BIGDATA2025_URL + rel
            tmp = path.with_suffix(".part")
            try:
                with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
                    shutil.copyfileobj(response, out)
            except OSError as exc:
                raise OSError(f"Download of {url} failed ({exc}). Download it by hand into {folder}.") from None
            tmp.rename(path)
        paths.append(path)
    if name == "steam":
        return pd.read_csv(paths[0]), pd.read_csv(paths[1])
    return (pd.read_csv(paths[0], index_col=0),)


def _unit_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


def steam_single_user(seed: int = 12, n_actions: int = 2000, noise_std: float = 0.1, paper_code: bool = False,
                      data_home: Union[str, Path, None] = None) -> LinearEnv:
    """Steam instance of the paper: one user (drawn with ``seed``), the most reviewed games they did not review.

    Parameters
    ----------
    seed : int
        Draws the user, with ``numpy.random.RandomState(seed)`` as in the paper's code.
    n_actions : int
        Size of the catalog (2000 in the paper).
    noise_std : float
        Standard deviation of the Gaussian reward noise (0.1 in the paper).
    paper_code : bool
        Rebuild the instance exactly as the repository code does. That code keeps the
        CSV index column as a feature (it dominates the norm of every game vector) and
        pairs the features of the reviewed games, in catalog order, with their labels,
        in review order. The default drops the index column and aligns each game with
        its own label.
    """
    games, reviews = fetch_bigdata2025("steam", data_home)
    n_actions = check_int("n_actions", n_actions, low=1)
    noise_std = check_float("noise_std", noise_std, low=0.0)
    if not paper_code:
        games = games.drop(columns=[c for c in games.columns if c.startswith("Unnamed")])
    rng = np.random.RandomState(seed)
    user = rng.choice(reviews["user_id"].unique())
    reviewed = reviews[reviews["user_id"] == user]
    if paper_code:
        rows = games[games["app_id"].isin(reviewed["app_id"].values)]
        labels = reviewed["is_recommended"].to_numpy(dtype=float)
    else:
        rows = reviewed[["app_id", "is_recommended"]].merge(games, on="app_id", how="inner")
        labels = rows.pop("is_recommended").to_numpy(dtype=float)
    X_user = _unit_rows(rows.drop(columns="app_id").to_numpy(dtype=float))
    theta, *_ = np.linalg.lstsq(X_user, labels, rcond=None)
    catalog = games[~games["app_id"].isin(reviewed["app_id"].values)].iloc[:n_actions]
    X = _unit_rows(catalog.drop(columns="app_id").to_numpy(dtype=float))
    return LinearEnv(X, theta, noise_std=noise_std, normalize=True)


def movielens_single_user(seed: int = 12, noise_std: float = 0.1, data_home: Union[str, Path, None] = None
                          ) -> LinearEnv:
    """MovieLens instance of the paper: the movies one user rated (user drawn with ``seed``).

    The mean reward of a movie is the user's rating / 5; the environment adds
    Gaussian noise of standard deviation ``noise_std`` (0.1 in the paper).
    """
    (ratings,) = fetch_bigdata2025("movielens", data_home)
    noise_std = check_float("noise_std", noise_std, low=0.0)
    rng = np.random.RandomState(seed)
    user = rng.choice(ratings["userId"].unique())
    raters = ratings.groupby("movieId")["userId"].apply(list)
    seen = ratings[ratings["userId"] == user]
    columns = sorted(ratings["userId"].unique())
    position = {u: j for j, u in enumerate(columns)}
    one_hot = np.zeros((len(seen), len(columns)))
    for i, movie in enumerate(seen["movieId"].to_numpy()):
        one_hot[i, [position[u] for u in raters[movie]]] = 1.0
    movie_features = seen.drop(columns=["userId", "rating", "movieId", "timestamp", "movie_index"])
    X = _unit_rows(np.hstack([movie_features.to_numpy(dtype=float), one_hot]))
    return LinearEnv.from_means(X, seen["rating"].to_numpy(dtype=float) / 5.0, noise_std=noise_std)


def bigdata2025_instances(name: str, n_instances: int = 10, first_seed: int = 0, **kwargs) -> List[LinearEnv]:
    """``n_instances`` users (seeds ``first_seed, first_seed + 1, ...``), for ``bk.Experiment``.

    ``kwargs`` go to :func:`steam_single_user` or :func:`movielens_single_user`.
    """
    name = check_choice("name", name, tuple(_FILES))
    n_instances = check_int("n_instances", n_instances, low=1)
    build = steam_single_user if name == "steam" else movielens_single_user
    return [build(seed=first_seed + i, **kwargs) for i in range(n_instances)]
