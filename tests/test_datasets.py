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


def test_movielens_linear_follows_li_et_al(movielens_home, tmp_path):
    data = datasets.movielens_linear("small", n_users=50, n_items=60, n_features=10, data_home=movielens_home)
    K, d = data.action_features.shape
    assert (K, d) == (60, 10) and data.user_thetas.shape == (50, 1, 10)
    # Eq. 3 of Li et al. (2019): unit norm, last coordinate 1 / sqrt(2), means in [0, 1].
    for v in (data.action_features, data.user_thetas[:, 0]):
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0)
        np.testing.assert_allclose(v[:, -1], 2 ** -0.5)
    means = data.user_thetas[:, 0] @ data.action_features.T
    assert means.min() >= -1e-12 and means.max() <= 1 + 1e-12
    assert data.user_stream is None  # uniform arrivals
    env = data.to_env()
    assert env.spec.n_actions == 20  # 20 items offered per step
    res = bk.Experiment(env, [bk.LinUCB(), bk.Random()], horizon=300, seed=0).run()
    assert res.final("regret")["LinUCB"].shape == (1,)
    # Round trip through the documented .npz format.
    path = data.save(tmp_path / "ml.npz")
    again = datasets.load_npz(path)
    assert isinstance(again, bk.MultiUserLinearEnv) and again.spec.n_users == 50
    binary = datasets.movielens_linear("small", n_users=50, n_items=60, n_features=10, feedback="binary",
                                       arrivals="data", data_home=movielens_home)
    assert binary.objectives == ("binary",) and binary.user_stream.size == binary.info["n_ratings"]


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
    with pytest.raises(ValueError, match="n_features"):
        datasets.ratings_to_linear(fake_ratings(), n_users=5, n_features=20)
    with pytest.raises(ValueError, match="feedback"):
        datasets.ratings_to_linear(fake_ratings(), feedback="clicks")


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
    data = datasets.steam_linear(steam_home, n_users=40, n_items=50, n_features=10)
    assert data.objectives == ("binary",) and data.action_features.shape == (50, 10)


def test_rewards_to_mab_scales_real_rewards():
    mab = datasets.rewards_to_mab({"a": [0, 1, 2, 100], "b": [3, 4, 5]}, scale="max")
    assert mab.action_names == ["a", "b"] and mab.spec.n_actions == 2
    obs = mab.reset(seed=0)[0]
    assert np.allclose(mab.mean_rewards(obs)[:, 0], [103 / 400, 12 / 300])
    with pytest.raises(ValueError, match="at least 2"):
        datasets.rewards_to_mab({"a": [1.0]})


def test_movielens_mab_follows_gupta_et_al(movielens_home):
    genres = datasets.movielens_mab("small", data_home=movielens_home)
    assert set(genres.action_names) <= set(GENRES) and genres.spec.reward_range == 1.0
    # One genre per movie: the genres share out the ratings of movies with a genre.
    _, movies = datasets.fetch_movielens("small", data_home=movielens_home)
    rated = fake_ratings()
    n_with_genre = rated["item_id"].isin(movies.loc[movies["genres"].map(len) > 0, "item_id"]).sum()
    assert sum(s.size for s in genres._samples) == n_with_genre
    movies_mab = datasets.movielens_mab("small", arms="movies", n_arms=5, data_home=movielens_home)
    assert movies_mab.spec.n_actions == 5
    assert datasets.movielens_mab("small", arms="movies", data_home=movielens_home).spec.n_actions == 50


def test_multi_user_env_offers_random_candidates():
    rng = np.random.default_rng(0)
    env = bk.MultiUserLinearEnv(rng.normal(size=(30, 4)), rng.normal(size=(5, 4)), n_candidates=6)
    obs, _ = env.reset(seed=0)
    assert env.spec.n_actions == 6 and obs.action_features.shape == (6, 4)
    best = env.best_mean_reward(np.ones(1), obs)
    assert np.isclose(best, (obs.action_features @ env.user_thetas[obs.user_id, 0]).max())
    res = bk.Experiment(env, [bk.LinUCB(n_buckets=2), bk.SCLUB(bonus="oful")], horizon=200, seed=0).run()
    assert np.all(np.isfinite(res.final("regret")["LinUCB_AB"]))


@pytest.fixture
def bigdata_raw(tmp_path):
    """Tiny raw files in the formats of Kaggle Steam and MovieLens 25M."""
    rng = np.random.default_rng(0)
    steam = tmp_path / "steam"
    steam.mkdir()
    n_games, users = 60, [901, 17, 555, 42, 3, 77]
    app_ids = rng.choice(np.arange(10, 10 ** 6), n_games, replace=False)
    pd.DataFrame({"app_id": app_ids, "title": "g", "date_release": "2015-03-01", "win": True, "mac": False,
                  "linux": rng.random(n_games) < 0.5, "rating": rng.choice(["Mixed", "Positive", "Very Positive"],
                                                                           n_games),
                  "positive_ratio": rng.integers(0, 101, n_games), "user_reviews": rng.integers(10, 10 ** 5, n_games),
                  "price_final": rng.integers(0, 60, n_games) + 0.99, "price_original": 59.99,
                  "discount": rng.integers(0, 90, n_games) * 1.0, "steam_deck": rng.random(n_games) < 0.5}
                 ).to_csv(steam / "games.csv", index=False)
    with open(steam / "games_metadata.json", "w") as f:
        for i, g in enumerate(app_ids):
            tags = [] if i == 0 else list(rng.choice(["Indie", "RPG", "Puzzle", "Co-op"], 2, replace=False))
            f.write(json.dumps({"app_id": int(g), "description": "", "tags": tags}) + "\n")
    rows = [(int(g), bool(rng.random() < 0.6), u) for u in users for g in rng.choice(app_ids, 40, replace=False)]
    rec = pd.DataFrame(rows, columns=["app_id", "is_recommended", "user_id"]).sample(frac=1, random_state=0)
    rec.assign(helpful=0, funny=0, date="2022-01-01", hours=1.0, review_id=np.arange(len(rec))).to_csv(
        steam / "recommendations.csv", index=False)
    ml = tmp_path / "ml-25m"
    ml.mkdir()
    pd.DataFrame({"movieId": np.arange(1, 31), "title": [f"M{i} ({1990 + i})" for i in range(30)],
                  "genres": rng.choice(["Action|Drama", "Comedy", "(no genres listed)"], 30)}
                 ).to_csv(ml / "movies.csv", index=False)
    rows = [(u, m, rng.integers(1, 11) / 2, 0) for u in range(1, 7) for m in np.sort(rng.choice(30, 12, False) + 1)]
    pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(ml / "ratings.csv", index=False)
    return tmp_path


def test_bigdata2025_steam_from_raw_files(bigdata_raw):
    home, path = bigdata_raw / "home", bigdata_raw / "steam"
    env = datasets.steam_single_user(seed=0, n_actions=10, path=path, min_reviews=5, data_home=home)
    t = datasets.load_steam_tables(path, min_reviews=5, data_home=home)
    assert env.spec.n_actions == 10 and env.spec.n_features == t["numeric"].shape[1] + 4 + 6  # 4 tags, 6 users
    np.testing.assert_allclose(np.linalg.norm(env.action_features, axis=1), 1.0)
    means = env.mean_rewards(env.reset(seed=0)[0])[:, 0]
    assert means.min() == 0.0 and means.max() == 1.0  # min-max normalized, as in the paper
    # Each reviewed game is fitted with its own label, read here from the raw file.
    user = np.random.RandomState(0).choice(pd.unique(t["review_user"]))
    raw = pd.read_csv(path / "recommendations.csv")
    mine = raw[(raw.user_id == user) & raw.app_id.isin(t["app_id"])]
    X = np.hstack([t["numeric"], t["tags"], t["users"]]).astype(float)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    row = {a: i for i, a in enumerate(t["app_id"])}
    theta = np.linalg.lstsq(X[[row[a] for a in mine.app_id]], mine.is_recommended.astype(float), rcond=None)[0]
    np.testing.assert_allclose(env.theta[0], theta, rtol=1e-6, atol=1e-8 * np.abs(theta).max())
    paper = datasets.steam_single_user(seed=0, n_actions=10, paper_code=True, path=path, min_reviews=5,
                                       data_home=home)
    assert paper.spec.n_features == env.spec.n_features + 1  # the paper's code keeps the CSV index column
    # The Kaggle zip works as well as the extracted folder.
    import zipfile
    with zipfile.ZipFile(bigdata_raw / "game-recommendations-on-steam.zip", "w") as z:
        for f in path.iterdir():
            z.write(f, f.name)
    zipped = datasets.steam_single_user(seed=0, n_actions=10, min_reviews=5, data_home=bigdata_raw / "zip_home",
                                        path=bigdata_raw / "game-recommendations-on-steam.zip")
    np.testing.assert_allclose(zipped.theta, env.theta)
    # The cache makes the raw files unnecessary afterwards.
    again = datasets.steam_single_user(seed=0, n_actions=10, min_reviews=5, data_home=home)
    np.testing.assert_allclose(again.theta, env.theta)
    with pytest.raises(FileNotFoundError, match="kaggle"):
        datasets.steam_single_user(min_reviews=7, data_home=bigdata_raw / "empty")


def test_bigdata2025_movielens_from_raw_files(bigdata_raw):
    kwargs = dict(path=bigdata_raw / "ml-25m", min_ratings=5, max_ratings=20, data_home=bigdata_raw / "home")
    env = datasets.movielens_single_user(seed=3, **kwargs)
    t = datasets.load_movielens_tables(5, 20, bigdata_raw / "ml-25m", bigdata_raw / "home")
    user = np.random.RandomState(3).choice(pd.unique(t["user"]))
    raw = pd.read_csv(bigdata_raw / "ml-25m" / "ratings.csv")
    mine = raw[raw.userId == user].sort_values("movieId")
    assert env.spec.n_actions == 12 and env.spec.n_features == 1 + 4 + 6 + 6  # year, 4 genres, statistics, users
    np.testing.assert_allclose(env.mean_rewards(env.reset(seed=0)[0])[:, 0], mine.rating / 5)
    envs = datasets.bigdata2025_instances("movielens", n_instances=2, **kwargs)
    res = bk.Experiment(envs, [bk.LinUCB(), bk.Random()], horizon=20, seed=0, verbose=0).run()
    assert res.n_instances == 2 and np.isfinite(res.final()["LinUCB"]).all()
