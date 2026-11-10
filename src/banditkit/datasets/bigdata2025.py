"""Single-user linear bandits of Izri, Groz and Maniu (IEEE BigData 2025), rebuilt from the raw public data.

"Implementing Efficient Linear Bandits Via Sketches and Random Projections",
IEEE International Conference on Big Data, 2025, pp. 956-965,
doi:10.1109/BigData66926.2025.11401692. Code: https://github.com/LiliZr/Stochastic-linear-bandits.

The instances follow the processing of the paper's notebooks and loaders, from the raw files:

* Steam: Kaggle "Game Recommendations on Steam" (``games.csv``, ``games_metadata.json``,
  ``recommendations.csv``). Kaggle needs an account: download it by hand
  (:data:`STEAM_KAGGLE_URL`, or
  ``kaggle datasets download -d antonkozyriev/game-recommendations-on-steam -p ~/banditkit_data``).
* MovieLens 25M (``ratings.csv``, ``movies.csv``), downloaded once from GroupLens.

Each instance is one user, drawn with a seed as in the paper's runs (one user per run). The
same seed gives the same user as in the paper: rows are kept in the order the paper's code
produced them (pandas older than 2.2 grouped the rows of an inner merge by key; recent
versions keep the left order, which changes the user drawn by a seed). Features are
high-dimensional (one-hot tags, one-hot users), the setting of sketches and random projections.

The processed tables are cached in ``data_home/bigdata2025`` (a few MB): only the first call
reads the raw files. Delete the cache files to rebuild them.
"""
from __future__ import annotations

import json
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .._validation import check_choice, check_float, check_int
from ..envs import LinearEnv

__all__ = ["STEAM_KAGGLE_URL", "steam_single_user", "movielens_single_user", "bigdata2025_instances",
           "load_steam_tables", "load_movielens_tables"]

#: Page of the raw Steam data (Kaggle, account needed).
STEAM_KAGGLE_URL = "https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam"
_STEAM_FILES = ("games.csv", "games_metadata.json", "recommendations.csv")
PathLike = Union[str, Path]


# ------------------------------------------------------------------ helpers
def _group_rows(frame, key: str):
    """Rows grouped by ``key``, keys in order of first appearance, rows of a key in their order.

    This is the row order of ``pd.merge(frame, right, on=key)`` (inner, ``right`` with
    unique keys) in pandas < 2.2, the version the paper's notebooks ran with.
    """
    import pandas as pd

    codes = pd.factorize(frame[key])[0]
    return frame.iloc[np.argsort(codes, kind="stable")].reset_index(drop=True)


def _unit_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


def _one_hot(labels, classes) -> np.ndarray:
    """``sklearn.preprocessing.LabelBinarizer`` output (one column if there are two classes)."""
    labels = np.asarray(labels)
    if len(classes) == 2:
        return (labels == classes[1]).astype(float)[:, None]
    return (labels[:, None] == np.asarray(classes)[None, :]).astype(float)


def _cache_dir(data_home: Optional[PathLike]) -> Path:
    from .movielens import get_data_home

    folder = get_data_home(data_home) / "bigdata2025"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _steam_reader(path: Optional[PathLike], data_home: Optional[PathLike]):
    """Function name -> open file of the raw Steam data (a folder or the Kaggle zip)."""
    from .movielens import get_data_home

    candidates = [Path(path).expanduser()] if path is not None else [
        get_data_home(data_home) / "steam", get_data_home(data_home) / "game-recommendations-on-steam.zip"]
    for root in candidates:
        if root.is_dir() and all((root / f).exists() for f in _STEAM_FILES):
            return lambda name: open(root / name, "rb")
        if root.is_file() and zipfile.is_zipfile(root):
            archive = zipfile.ZipFile(root)
            members = {Path(m).name: m for m in archive.namelist()}
            if all(f in members for f in _STEAM_FILES):
                return lambda name: archive.open(members[name])
    where = " or ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Raw Steam data not found in {where}. Download the Kaggle dataset ({STEAM_KAGGLE_URL}), for instance "
        "with `kaggle datasets download -d antonkozyriev/game-recommendations-on-steam -p ~/banditkit_data`, "
        f"or pass its folder or zip as `path`. Files needed: {', '.join(_STEAM_FILES)}.")


# ------------------------------------------------------------------- Steam
def load_steam_tables(path: Optional[PathLike] = None, min_reviews: int = 200,
                      data_home: Optional[PathLike] = None) -> Dict[str, np.ndarray]:
    """Steam tables of the paper, from the raw Kaggle files (cached after the first call).

    Processing of the paper's notebook (``data/steam/steam_processing.ipynb``): drop the
    games without tags and their reviews; keep the users with at least ``min_reviews``
    reviews; a game is described by its release year, platforms, positive ratio, number
    of reviews, prices, discount, Steam Deck flag, rating class (one-hot), tags (one-hot)
    and the kept users who reviewed it (one-hot). Games are sorted by number of reviews.

    Parameters
    ----------
    path : str or Path, optional
        Folder or zip of the Kaggle dataset (default: ``data_home/steam`` or
        ``data_home/game-recommendations-on-steam.zip``).
    min_reviews : int
        Users with fewer reviews are dropped (200 in the paper).
    data_home : str or Path, optional
        Folder of the cache (see :func:`banditkit.datasets.get_data_home`).

    Returns
    -------
    dict of arrays
        ``app_id``, ``first_row`` (index column of the paper's CSV), ``user_reviews``,
        ``numeric``, ``tags``, ``users`` (one row per game, in order of first review),
        ``review_app``, ``review_user``, ``review_label`` (reviews in the order of the
        paper's files), ``feature_names`` and ``counts`` (users, games, reviews).
    """
    min_reviews = check_int("min_reviews", min_reviews, low=1)
    cache = _cache_dir(data_home) / f"steam_min{min_reviews}_v1.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    tables = _build_steam_tables(_steam_reader(path, data_home), min_reviews)
    np.savez_compressed(cache, **tables)
    return tables


def _build_steam_tables(open_file, min_reviews: int) -> Dict[str, np.ndarray]:
    import pandas as pd

    # Metadata: tags of each game; games with an empty tag list are dropped.
    tags_of: Dict[int, list] = {}
    no_tags = set()
    with open_file("games_metadata.json") as f:
        for line in f:
            if not line.strip():
                continue
            game = json.loads(line)
            if game["tags"]:
                tags_of.setdefault(int(game["app_id"]), []).extend(game["tags"])
            else:
                no_tags.add(int(game["app_id"]))

    with open_file("games.csv") as f:
        games = pd.read_csv(f)
    rating_classes = sorted(games["rating"].unique())   # the paper fits the one-hot encoder on every game
    games = games[games["app_id"].isin(list(tags_of))].reset_index(drop=True)
    numeric_cols = [c for c in games.columns if c not in ("app_id", "title", "rating")]
    games["date_release"] = [int(str(x)[:4]) for x in games["date_release"]]
    for col in ("win", "mac", "linux", "steam_deck"):
        if col in games:
            games[col] = games[col].astype(int)
    tag_classes = sorted({t for g in games["app_id"] for t in tags_of[g]})
    tag_pos = {t: j for j, t in enumerate(tag_classes)}

    with open_file("recommendations.csv") as f:
        reviews = pd.read_csv(f, usecols=["app_id", "is_recommended", "user_id"])
    reviews = reviews[~reviews["app_id"].isin(list(no_tags))]
    counts = reviews["user_id"].value_counts()
    kept = counts.index[counts.to_numpy() >= min_reviews]
    reviews = _group_rows(reviews[reviews["user_id"].isin(kept)], "user_id")
    reviews = _group_rows(reviews[reviews["app_id"].isin(games["app_id"])], "app_id")
    reviews["is_recommended"] = reviews["is_recommended"].astype(int)

    # One row per reviewed game, at its first review (the paper's drop_duplicates). The paper then
    # sorts the games by number of reviews; steam_single_user does it (see _catalog_order).
    first = reviews.drop_duplicates("app_id")[["app_id"]]
    app_id = first["app_id"].to_numpy()
    by_id = games.set_index("app_id").loc[app_id]

    numeric = np.hstack([by_id[numeric_cols].to_numpy(dtype=float), _one_hot(by_id["rating"], rating_classes)])
    tags = np.zeros((len(app_id), len(tag_classes)), dtype=np.uint8)
    for i, g in enumerate(app_id):
        tags[i, [tag_pos[t] for t in tags_of[g]]] = 1
    user_classes = np.sort(reviews["user_id"].unique())
    row_of = pd.Series(np.arange(len(app_id)), index=app_id)
    users = np.zeros((len(app_id), len(user_classes)), dtype=np.uint8)
    users[row_of.loc[reviews["app_id"]].to_numpy(), np.searchsorted(user_classes, reviews["user_id"])] = 1

    names = ([str(c) for c in numeric_cols] + [f"rating={c}" for c in rating_classes]
             + [f"tag={t}" for t in tag_classes] + [f"user={u}" for u in user_classes])
    return {
        "app_id": app_id, "first_row": first.index.to_numpy(), "user_reviews": by_id["user_reviews"].to_numpy(),
        "numeric": numeric, "tags": tags, "users": users, "review_app": reviews["app_id"].to_numpy(), "review_user": reviews["user_id"].to_numpy(),
        "review_label": reviews["is_recommended"].to_numpy(), "feature_names": np.array(names),
        "counts": np.array([len(user_classes), len(app_id), len(reviews)]),
    }


@lru_cache(maxsize=2)
def _steam_cached(path: Optional[str], min_reviews: int, data_home: Optional[str]):
    return load_steam_tables(path, min_reviews, data_home)


def steam_single_user(seed: int = 12, n_actions: int = 2000, noise_std: float = 0.1, paper_code: bool = False,
                      path: Optional[PathLike] = None, min_reviews: int = 200,
                      data_home: Optional[PathLike] = None) -> LinearEnv:
    """Steam instance of the paper: one user (drawn with ``seed``), the most reviewed games they did not review.

    The user's parameter is the least-squares fit, without intercept, of "recommended"
    on the games they reviewed (rows scaled to norm 1). The actions are the ``n_actions``
    most reviewed games the user did not review (rows scaled to norm 1). Mean rewards are
    min-max normalized over the actions (``LinearEnv(normalize=True)``), as in the paper.

    Parameters
    ----------
    seed : int
        Draws the user, with ``numpy.random.RandomState(seed)`` as in the paper's code.
    n_actions : int
        Size of the catalog (2000 in the paper).
    noise_std : float
        Standard deviation of the Gaussian reward noise (0.1 in the paper).
    paper_code : bool
        Reproduce two errors of the paper's code. (1) The index column of the CSV file
        the code writes is read back as a feature: it holds row numbers (up to about
        500,000) and dominates the norm of every game vector. (2) The user's games, taken
        in catalog order, are paired with labels in review order, so the fit uses wrong
        labels. The default drops the index column and pairs each game with its own label.
        Games with the same number of reviews are ordered by numpy's quicksort with
        ``paper_code=True`` (the paper's call; this order can change with the numpy version
        and the CPU), and by first review otherwise (the same on every machine).
    path : str or Path, optional
        Folder or zip of the raw Kaggle data; see :func:`load_steam_tables`.
    min_reviews : int
        Users kept: at least this many reviews (200 in the paper).
    data_home : str or Path, optional
        Folder of the cache.
    """
    n_actions = check_int("n_actions", n_actions, low=1)
    noise_std = check_float("noise_std", noise_std, low=0.0)
    t = _steam_cached(None if path is None else str(path), int(min_reviews),
                      None if data_home is None else str(data_home))
    rng = np.random.RandomState(seed)
    user = rng.choice(np.asarray(_unique_in_order(t["review_user"])))
    mine = t["review_user"] == user
    ids, labels = t["review_app"][mine], t["review_label"][mine].astype(float)

    def features(rows: np.ndarray) -> np.ndarray:
        parts = [t["numeric"][rows], t["tags"][rows], t["users"][rows]]
        if paper_code:
            parts.insert(0, t["first_row"][rows, None].astype(float))
        return _unit_rows(np.hstack(parts).astype(float))

    order = _catalog_order(t["user_reviews"], paper_code)  # games sorted by number of reviews
    reviewed = np.isin(t["app_id"][order], ids)
    if paper_code:
        rows = order[np.flatnonzero(reviewed)]           # catalog order, labels stay in review order
    else:
        rows = _rows_of(t["app_id"], ids)                # the row of each reviewed game, in review order
    theta, *_ = np.linalg.lstsq(features(rows), labels, rcond=None)
    catalog = order[np.flatnonzero(~reviewed)[:n_actions]]
    return LinearEnv(features(catalog), theta, noise_std=noise_std, normalize=True)


def _catalog_order(user_reviews: np.ndarray, paper_code: bool) -> np.ndarray:
    """Games by decreasing number of reviews: the paper's pandas call, or a stable sort."""
    import pandas as pd

    kind = "quicksort" if paper_code else "stable"
    return pd.Series(user_reviews).sort_values(ascending=False, kind=kind).index.to_numpy()


def _unique_in_order(values: np.ndarray) -> np.ndarray:
    _, first = np.unique(values, return_index=True)
    return values[np.sort(first)]


# --------------------------------------------------------------- MovieLens
def load_movielens_tables(min_ratings: int = 1500, max_ratings: int = 3000, path: Optional[PathLike] = None,
                          data_home: Optional[PathLike] = None) -> Dict[str, np.ndarray]:
    """MovieLens 25M tables of the paper (cached after the first call).

    Processing of the paper's notebook (``data/movielens_25m/movies.ipynb``): keep the
    users with more than ``min_ratings`` and fewer than ``max_ratings`` ratings; a movie
    is described by its year (-1 if the title has none), its genres (one-hot) and the
    mean, median, min, max, standard deviation and number of its ratings by these users.

    Parameters
    ----------
    min_ratings, max_ratings : int
        Users kept: strictly between these numbers of ratings (1500 and 3000 in the paper).
    path : str or Path, optional
        Folder with ``ratings.csv`` and ``movies.csv`` (default: MovieLens 25M, downloaded
        once into ``data_home``).
    data_home : str or Path, optional
        Folder of the download and the cache.

    Returns
    -------
    dict of arrays
        ``movie_id`` and ``movie_features`` (one row per rated movie), ``user``, ``movie``,
        ``rating`` (ratings in the order of the paper's file) and ``feature_names``.
    """
    import pandas as pd

    from .movielens import _movielens_root

    min_ratings = check_int("min_ratings", min_ratings, low=0)
    max_ratings = check_int("max_ratings", max_ratings, low=min_ratings + 1)
    cache = _cache_dir(data_home) / f"movielens_{min_ratings}_{max_ratings}_v1.npz"
    if path is None and cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    root = Path(path).expanduser() if path is not None else _movielens_root("25m", data_home)
    ratings = pd.read_csv(root / "ratings.csv", usecols=["userId", "movieId", "rating"])
    movies = pd.read_csv(root / "movies.csv")

    counts = ratings["userId"].value_counts()
    kept = counts.index[(counts.to_numpy() > min_ratings) & (counts.to_numpy() < max_ratings)]
    ratings = _group_rows(ratings[ratings["userId"].isin(kept)], "userId")

    def year(title: str) -> int:  # the paper's get_year
        idx = title.rfind("(") + 1
        try:
            return int(title[idx:idx + 4])
        except ValueError:
            return -1

    genres = movies["genres"].str.split("|")
    genre_classes = sorted({g for gs in genres for g in gs})
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "median", "min", "max", "std", "size"])
    movies = movies[movies["movieId"].isin(stats.index)]
    genre_hot = np.array([[g in set(gs) for g in genre_classes] for gs in genres.loc[movies.index]], dtype=float)
    movie_features = np.hstack([
        np.array([year(t) for t in movies["title"]], dtype=float)[:, None], genre_hot,
        stats.loc[movies["movieId"]].fillna(0.0).to_numpy(dtype=float)])
    ratings = _group_rows(ratings[ratings["movieId"].isin(movies["movieId"])], "movieId")
    names = (["year"] + [f"genre={g}" for g in genre_classes]
             + [f"movie_{s}_ratings" for s in ("mean", "median", "min", "max", "std", "number")])
    tables = {"movie_id": movies["movieId"].to_numpy(), "movie_features": movie_features,
              "user": ratings["userId"].to_numpy(), "movie": ratings["movieId"].to_numpy(),
              "rating": ratings["rating"].to_numpy(dtype=float), "feature_names": np.array(names)}
    if path is None:
        np.savez_compressed(cache, **tables)
    return tables


@lru_cache(maxsize=2)
def _movielens_cached(min_ratings: int, max_ratings: int, path: Optional[str], data_home: Optional[str]):
    return load_movielens_tables(min_ratings, max_ratings, path, data_home)


def movielens_single_user(seed: int = 12, noise_std: float = 0.1, path: Optional[PathLike] = None,
                          min_ratings: int = 1500, max_ratings: int = 3000,
                          data_home: Optional[PathLike] = None) -> LinearEnv:
    """MovieLens instance of the paper: the movies one user rated (user drawn with ``seed``).

    The features of a movie are those of :func:`load_movielens_tables`, followed by the
    kept users who rated it (one-hot); rows are scaled to norm 1. The mean reward of a
    movie is the user's rating / 5; the environment adds Gaussian noise of standard
    deviation ``noise_std`` (0.1 in the paper).

    The rating statistics of a movie include the user's own rating, as in the paper. For a
    movie few kept users rated, they almost give the reward away.
    """
    noise_std = check_float("noise_std", noise_std, low=0.0)
    t = _movielens_cached(int(min_ratings), int(max_ratings), None if path is None else str(path),
                          None if data_home is None else str(data_home))
    rng = np.random.RandomState(seed)
    user = rng.choice(np.asarray(_unique_in_order(t["user"])))
    mine = np.flatnonzero(t["user"] == user)
    mine = mine[np.argsort(t["movie"][mine], kind="stable")]     # the paper's rows: sorted by movie
    movies = t["movie"][mine]
    users = np.sort(np.unique(t["user"]))
    seen = np.isin(t["movie"], movies)
    one_hot = np.zeros((len(movies), len(users)))
    one_hot[np.searchsorted(movies, t["movie"][seen]), np.searchsorted(users, t["user"][seen])] = 1.0
    X = _unit_rows(np.hstack([t["movie_features"][_rows_of(t["movie_id"], movies)], one_hot]))
    return LinearEnv.from_means(X, t["rating"][mine] / 5.0, noise_std=noise_std)


def _rows_of(ids: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Positions in ``ids`` of each value of ``wanted`` (all present)."""
    order = np.argsort(ids)
    return order[np.searchsorted(ids, wanted, sorter=order)]


# --------------------------------------------------------------- instances
def bigdata2025_instances(name: str, n_instances: int = 10, first_seed: int = 0, **kwargs) -> List[LinearEnv]:
    """``n_instances`` users (seeds ``first_seed, first_seed + 1, ...``), for ``bk.Experiment``.

    The paper runs 20 users, seeds 0 to 19. ``kwargs`` go to :func:`steam_single_user` or
    :func:`movielens_single_user` (``path`` of the raw data, ``paper_code``...).
    """
    name = check_choice("name", name, ("steam", "movielens"))
    n_instances = check_int("n_instances", n_instances, low=1)
    build = steam_single_user if name == "steam" else movielens_single_user
    return [build(seed=first_seed + i, **kwargs) for i in range(n_instances)]
