"""Dataset loaders on small fake files in the real formats (no download)."""
import json

import numpy as np
import pandas as pd
import pytest

import banditkit as bk
from banditkit import datasets

GENRES = ["Action", "Comedy", "Drama", "Romance", "Thriller"]


def fake_ratings(n_users=60, n_items=80, n_ratings=4000, seed=0):
    rng = np.random.default_rng(seed)
    pairs = rng.choice(n_users * n_items, size=n_ratings, replace=False)
    return pd.DataFrame({"user_id": pairs // n_items + 1, "item_id": pairs % n_items + 1,
                         "rating": rng.integers(1, 11, size=n_ratings) / 2.0,
                         "timestamp": rng.integers(10 ** 9, 2 * 10 ** 9, size=n_ratings)})


def fake_movies(n_items=80, seed=0):
    rng = np.random.default_rng(seed)
    genres = ["|".join(rng.choice(GENRES, size=rng.integers(1, 3), replace=False)) for _ in range(n_items)]
    genres[0] = "(no genres listed)"
    return pd.DataFrame({"item_id": np.arange(1, n_items + 1),
                         "title": [f"Café {i} ({1980 + i % 30})" for i in range(n_items)], "genres": genres})


@pytest.fixture
def movielens_home(tmp_path):
    ratings, movies = fake_ratings(), fake_movies()
    small = tmp_path / "ml-latest-small"
    small.mkdir()
    ratings.rename(columns={"user_id": "userId", "item_id": "movieId"}).to_csv(small / "ratings.csv", index=False)
    movies.rename(columns={"item_id": "movieId"}).to_csv(small / "movies.csv", index=False)
    one_m = tmp_path / "ml-1m"
    one_m.mkdir()
    with open(one_m / "ratings.dat", "w", encoding="latin-1") as f:
        for r in ratings.itertuples():
            f.write(f"{r.user_id}::{r.item_id}::{int(r.rating * 2) // 2 or 1}::{r.timestamp}\n")
    with open(one_m / "movies.dat", "w", encoding="latin-1") as f:
        for m in movies.itertuples():
            f.write(f"{m.item_id}::{m.title}::{m.genres}\n")
    return tmp_path


def test_fetch_movielens_parses_csv_and_dat(movielens_home):
    for variant in ("small", "1m"):
        ratings, movies = datasets.fetch_movielens(variant, data_home=movielens_home, download_if_missing=False)
        assert list(ratings.columns) == ["user_id", "item_id", "rating", "timestamp"]
        assert movies.loc[0, "genres"] == [] and movies.loc[1, "year"] == 1981
        assert movies.loc[1, "title"].startswith("Café")


def test_missing_variant_without_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        datasets.fetch_movielens("25m", data_home=tmp_path, download_if_missing=False)


def test_movielens_linear(movielens_home, tmp_path):
    data = datasets.movielens_linear("small", n_features=20, min_user_ratings=5, min_item_ratings=5,
                                     data_home=movielens_home)
    K, d = data.action_features.shape
    assert d == 20 and data.user_thetas.shape == (data.user_ids.size, 2, 20)
    np.testing.assert_allclose(np.linalg.norm(data.action_features, axis=1), 1.0)
    assert data.user_stream.size == data.info["n_ratings"]
    env = data.to_env()
    res = bk.Experiment(env, [bk.LinUCB(), bk.Random()], horizon=300, seed=0).run()
    assert res.final("regret")["LinUCB"].shape == (1,)
    # Round trip through the documented .npz format.
    path = data.save(tmp_path / "ml.npz")
    again = datasets.load_npz(path)
    assert isinstance(again, bk.MultiUserLinearEnv) and again.spec.n_actions == K


def test_load_npz_reads_legacy_scaling_files(tmp_path):
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "old.npz", item_features=rng.normal(size=(30, 4)), user_thetas=rng.normal(size=(5, 2, 4)),
             user_stream=rng.integers(5, size=100), noise_std=0.05, n_users=5, n_items=30, d=4, D_objectives=2)
    env = datasets.load_npz(tmp_path / "old.npz")
    assert env.spec.n_users == 5 and env.spec.reward_dim == 2 and env.max_episode_steps == 100


def test_load_npz_mab_layouts(tmp_path):
    np.savez(tmp_path / "mab.npz", means=[[0.1, 0.2], [0.5, 0.4]], noise_std=0.1)
    assert datasets.load_npz(tmp_path / "mab.npz").spec.reward_dim == 2
    np.savez(tmp_path / "group.npz", relevance=np.full((4, 3), 0.5))
    assert isinstance(datasets.load_npz(tmp_path / "group.npz"), bk.GroupRecommendationMAB)
    np.savez(tmp_path / "bad.npz", foo=[1])
    with pytest.raises(KeyError, match="no known layout"):
        datasets.load_npz(tmp_path / "bad.npz")


def test_ratings_to_linear_validates_inputs():
    with pytest.raises(ValueError, match="columns"):
        datasets.ratings_to_linear(pd.DataFrame({"user": [1], "item_id": [1], "rating": [1]}))
    with pytest.raises(ValueError, match="No rating left"):
        datasets.ratings_to_linear(fake_ratings(), min_user_ratings=10 ** 6)
    with pytest.raises(ValueError, match="objectives"):
        datasets.ratings_to_linear(fake_ratings(), objectives=("clicks",))


@pytest.fixture
def steam_home(tmp_path):
    ratings = fake_ratings(seed=1)
    rec = pd.DataFrame({"app_id": ratings["item_id"], "helpful": 0, "funny": 0,
                        "date": pd.to_datetime(ratings["timestamp"], unit="s").dt.strftime("%Y-%m-%d"),
                        "is_recommended": ratings["rating"] > 2.5, "hours": 1.0, "user_id": ratings["user_id"],
                        "review_id": np.arange(len(ratings))})
    rec.to_csv(tmp_path / "recommendations.csv", index=False)
    games = pd.DataFrame({"app_id": np.arange(1, 81), "title": "g", "win": True, "mac": False, "linux": False,
                          "positive_ratio": 80, "user_reviews": 100, "price_final": 9.99, "steam_deck": True})
    games.to_csv(tmp_path / "games.csv", index=False)
    with open(tmp_path / "games_metadata.json", "w") as f:
        for app in range(1, 81):
            f.write(json.dumps({"app_id": app, "description": "", "tags": [GENRES[app % 5], GENRES[(app + 1) % 5]]}) + "\n")
    return tmp_path


def test_steam(steam_home):
    data = datasets.steam_linear(steam_home, n_features=20, n_tags=5, min_user_ratings=5, min_item_ratings=5)
    assert data.objectives == ("like",) and data.action_features.shape[1] == 20


def test_rewards_to_mab_scales_real_rewards():
    mab = datasets.rewards_to_mab({"a": [0, 1, 2, 100], "b": [3, 4, 5]}, scale="max")
    assert mab.action_names == ["a", "b"] and mab.spec.n_actions == 2
    obs = mab.reset(seed=0)[0]
    assert np.allclose(mab.mean_rewards(obs)[:, 0], [103 / 400, 12 / 300])
    with pytest.raises(ValueError, match="at least 2"):
        datasets.rewards_to_mab({"a": [1.0]})


def test_taxis_mab_from_cached_file(tmp_path):
    rng = np.random.default_rng(0)
    zones = ["A", "B", "C", "JFK Airport"]
    df = pd.DataFrame({"pickup_zone": rng.choice(zones, size=400), "fare": rng.gamma(2.0, 5.0, size=400),
                       "tip": rng.uniform(0, 3, size=400)})
    df["total"] = df["fare"] + df["tip"]
    df.to_csv(tmp_path / "taxis.csv", index=False)
    mab = datasets.taxis_mab(n_arms=3, min_trips=10, data_home=tmp_path)
    assert sorted(mab.action_names) == ["A", "B", "C"]  # airports left out by default
    assert mab.spec.reward_range <= 1.0
    tips = datasets.taxis_mab(reward="tip_rate", n_arms=2, min_trips=10, data_home=tmp_path)
    assert tips.spec.n_actions == 2
    with pytest.raises(FileNotFoundError):
        datasets.fetch_taxis(data_home=tmp_path / "empty", download_if_missing=False)


def test_movielens_and_steam_mabs(movielens_home, steam_home):
    genres = datasets.movielens_mab("small", min_ratings=5, data_home=movielens_home)
    assert set(genres.action_names) <= set(GENRES) and genres.spec.reward_range <= 1.0
    movies = datasets.movielens_mab("small", arms="movies", n_arms=5, min_ratings=5, data_home=movielens_home)
    assert movies.spec.n_actions == 5
    games = datasets.steam_mab(steam_home, n_arms=4, min_reviews=5)
    assert games.spec.n_actions == 4
