"""MovieLens (GroupLens): download, parsing, and an EXAMPLE preprocessing into a multi-user linear bandit."""
from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Union

from .base import LinearDataset, ratings_to_linear

__all__ = ["fetch_movielens", "movielens_linear", "MOVIELENS_VARIANTS"]

#: Variant name -> (archive URL, folder inside the archive).
MOVIELENS_VARIANTS = {
    "small": ("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip", "ml-latest-small"),
    "1m": ("https://files.grouplens.org/datasets/movielens/ml-1m.zip", "ml-1m"),
    "20m": ("https://files.grouplens.org/datasets/movielens/ml-20m.zip", "ml-20m"),
    "25m": ("https://files.grouplens.org/datasets/movielens/ml-25m.zip", "ml-25m"),
    "32m": ("https://files.grouplens.org/datasets/movielens/ml-32m.zip", "ml-32m"),
}


def get_data_home(data_home: Union[str, Path, None] = None) -> Path:
    """Folder of downloaded datasets: ``data_home``, else $BANDITKIT_DATA, else ~/banditkit_data (as scikit-learn)."""
    path = Path(data_home or os.environ.get("BANDITKIT_DATA", Path.home() / "banditkit_data")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_and_extract(url: str, folder: str, data_home: Path) -> Path:
    archive = data_home / (folder + ".zip")
    tmp = archive.with_suffix(".part")
    try:
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
    except OSError as exc:
        raise ConnectionError(f"Could not download {url} ({exc}). Download it by hand and extract it "
                              f"into {data_home}.") from None
    tmp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(data_home)
    archive.unlink()
    return data_home / folder


def fetch_movielens(variant: str = "small", data_home: Union[str, Path, None] = None,
                    download_if_missing: bool = True):
    """Ratings and movies of a MovieLens variant, downloaded on first use.

    Parameters
    ----------
    variant : {"small", "1m", "20m", "25m", "32m"}
        ml-latest-small (100k ratings), ml-1m, ml-20m, ml-25m or ml-32m.
    data_home : str or Path, optional
        Download folder (see :func:`get_data_home`). An already extracted folder
        (``ml-1m``, ``ml-25m``...) placed there is used as is.
    download_if_missing : bool
        Download the archive from files.grouplens.org when absent.

    Returns
    -------
    ratings : DataFrame
        Columns ``user_id``, ``item_id``, ``rating``, ``timestamp``.
    movies : DataFrame
        Columns ``item_id``, ``title``, ``year`` (NaN if absent from the title), ``genres`` (list of str).
    """
    import pandas as pd

    if variant not in MOVIELENS_VARIANTS:
        raise ValueError(f"Unknown MovieLens variant {variant!r}; choose among {sorted(MOVIELENS_VARIANTS)}.")
    url, folder = MOVIELENS_VARIANTS[variant]
    home = get_data_home(data_home)
    root = home / folder
    if not root.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"{root} not found and download_if_missing=False.")
        root = _download_and_extract(url, folder, home)

    if (root / "ratings.csv").exists():
        ratings = pd.read_csv(root / "ratings.csv").rename(columns={"userId": "user_id", "movieId": "item_id"})
        movies = pd.read_csv(root / "movies.csv").rename(columns={"movieId": "item_id"})
    else:  # ml-1m: "::"-separated .dat files, latin-1 encoded
        ratings = pd.read_csv(root / "ratings.dat", sep="::", engine="python", encoding="latin-1",
                              names=["user_id", "item_id", "rating", "timestamp"])
        movies = pd.read_csv(root / "movies.dat", sep="::", engine="python", encoding="latin-1",
                             names=["item_id", "title", "genres"])
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False)
    movies = movies.assign(year=pd.to_numeric(year, errors="coerce"),
                           genres=movies["genres"].str.split("|").apply(
                               lambda g: [x for x in g if x != "(no genres listed)"]))
    return ratings[["user_id", "item_id", "rating", "timestamp"]], movies[["item_id", "title", "year", "genres"]]


def _genre_table(movies):
    """(item_id, category) pairs of the movie genres."""
    pairs = movies[["item_id", "genres"]].explode("genres").dropna()
    return pairs.rename(columns={"genres": "category"}).reset_index(drop=True)


def movielens_linear(variant: str = "small", n_users: int = 1000, n_items: int = 1000, n_features: int = 20,
                     feedback: str = "rating", like_threshold: float = 3.0, arrivals: str = "uniform",
                     noise_std: float = 0.1, data_home: Union[str, Path, None] = None, seed: int = 0
                     ) -> LinearDataset:
    """Example preprocessing: MovieLens as a multi-user linear bandit (movies are actions).

    Built with :func:`~banditkit.datasets.ratings_to_linear`, which follows Li et al.
    (IJCAI 2019) and, with ``feedback="binary"``, Wang et al. (NeurIPS 2023). Both
    papers use MovieLens 20M (``variant="20m"``); the default variant is the small
    one, for a quick download. ``.to_env()`` gives the environment.
    """
    ratings, _ = fetch_movielens(variant, data_home)
    return ratings_to_linear(ratings, n_users=n_users, n_items=n_items, n_features=n_features, feedback=feedback,
                             like_threshold=like_threshold, arrivals=arrivals, noise_std=noise_std, seed=seed)
