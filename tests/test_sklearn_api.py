"""Policies used like scikit-learn estimators: fit, partial_fit, predict on data."""
import warnings

import numpy as np
import pytest

import banditkit as bk

pytestmark = pytest.mark.filterwarnings("ignore::banditkit.ParameterWarning")


def logged_linear(n=200, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    theta = rng.normal(size=d)
    return X, X @ theta + 0.1 * rng.normal(size=n), theta


def test_mab_fit_learns_the_logged_means():
    rng = np.random.default_rng(0)
    arms = rng.integers(0, 3, size=300)
    rewards = np.array([0.2, 0.5, 0.8])[arms] + 0.1 * rng.normal(size=300)
    model = bk.UCB1().fit(arms, rewards)
    assert model.n_samples_seen_ == 300 and model.spec.n_actions == 3
    np.testing.assert_allclose(model.means, [rewards[arms == a].mean() for a in range(3)])
    action = model.predict()
    model.partial_fit([action], [1.0])
    assert model.n_samples_seen_ == 301


def test_mab_online_loop_from_scratch():
    rng = np.random.default_rng(1)
    means = np.array([0.2, 0.5, 0.8])
    model = bk.UCB1(reward_range=1.0).fit([], [], n_actions=3)
    for _ in range(2000):
        a = model.predict()
        model.partial_fit([a], [float(rng.random() < means[a])])
    assert model.counts[2] > 0.8 * 2000


def test_linear_fit_is_a_ridge_regression():
    X, y, _ = logged_linear()
    model = bk.LinUCB(lambda_=0.5).fit(X, y)
    expected = np.linalg.solve(0.5 * np.eye(5) + X.T @ X, X.T @ y)
    np.testing.assert_allclose(model.models[0].theta, expected, atol=1e-8)
    assert model.n_features_in_ == 5
    candidates = np.random.default_rng(2).normal(size=(20, 5))
    assert 0 <= model.predict(candidates) < 20


def test_linear_indices_in_the_candidate_set():
    X, y, _ = logged_linear()
    A = X[:50]
    rows = np.arange(200) % 50
    by_index = bk.LinUCB().fit(rows, y, action_features=A)
    by_features = bk.LinUCB().fit(A[rows], y)
    np.testing.assert_allclose(by_index.models[0].theta, by_features.models[0].theta)
    assert 0 <= by_index.predict() < 50  # default candidates: action_features


def test_linear_online_loop_finds_the_best_action():
    _, _, theta = logged_linear()
    rng = np.random.default_rng(3)
    A = rng.normal(size=(30, 5))
    model = bk.LinUCB(subgaussian=0.1).fit(np.empty((0, 5)), [], action_features=A, seed=0)
    for _ in range(1500):
        a = model.predict()
        model.partial_fit(A[[a]], [A[a] @ theta + 0.1 * rng.normal()])
    best = int(np.argmax(A @ theta))
    assert np.mean([model.predict() == best for _ in range(20)]) > 0.5
    # Projected candidates stay cached while single played actions come and go.
    projected = bk.LinUCB(n_components=3).fit(np.empty((0, 5)), [], action_features=A, seed=0)
    a = projected.predict()
    projected.partial_fit(A[[a]], [1.0])
    assert any(entry[0] is A for entry in projected._feature_cache)


def test_buckets_are_built_on_the_candidates():
    X, y, _ = logged_linear()
    with pytest.raises(ValueError, match="action_features"):
        bk.LinUCB(n_buckets=3).fit(X, y)
    A = X[:40]
    model = bk.LinUCB(n_buckets="sqrt").fit(X, y, action_features=A)
    model.predict()
    groups = model.models[0].buckets.groups
    assert len(groups) == 6 and sum(g.size for g in groups) == 40


def test_multi_user_fit_and_predict():
    X, y, _ = logged_linear(n=120)
    users = np.arange(120) % 4
    model = bk.CLUB().fit(X, y, users=users)
    assert model.spec.n_users == 4
    A = X[:10]
    assert 0 <= model.predict(A, user=2) < 10
    with pytest.raises(ValueError, match="user"):
        model.predict(A)


def test_generic_policies_resolve_from_the_data():
    X, y, _ = logged_linear()
    assert isinstance(bk.UCB().fit(X, y).inner, bk.LinUCB)
    assert isinstance(bk.UCB().fit([0, 1, 2], [0.1, 0.5, 0.2]).inner, bk.UCB1)


def test_shape_and_family_errors_name_the_fix():
    X, y, _ = logged_linear()
    with pytest.raises(ValueError, match="indices of the played arms"):
        bk.UCB1().fit(X, y)
    with pytest.raises(ValueError, match="features of the played actions"):
        bk.LinUCB().fit([0, 1], [1.0, 0.0])
    with pytest.raises(ValueError, match="same number of samples"):
        bk.LinUCB().fit(X, y[:10])
    with pytest.raises(ValueError, match="n_actions"):
        bk.UCB1().fit([], [])
    with pytest.raises(RuntimeError, match="fit"):
        bk.UCB1().predict()
    with pytest.raises(ValueError, match="candidate actions"):
        bk.LinUCB().fit(X, y).predict()
    with pytest.raises(ValueError, match="without candidate actions"):
        bk.UCB1().fit([0, 1], [0.0, 1.0]).predict(X)
    with pytest.raises(TypeError, match="first call"):
        bk.UCB1().fit([0, 1], [0.0, 1.0]).partial_fit([0], [1.0], n_actions=5)


def test_arbitrary_or_assumed_values_raise_a_warning():
    X, y, _ = logged_linear()
    with pytest.warns(bk.ParameterWarning, match="lambda_"):
        bk.LinUCB().fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", bk.ParameterWarning)
        bk.LinUCB(lambda_=1.0, subgaussian=0.1, theta_bound=2.0).fit(X, y)


def test_scikit_learn_clone():
    base = pytest.importorskip("sklearn.base")
    for policy in (bk.LinUCB(alpha=0.5, n_buckets="sqrt"), bk.UCB(alpha=0.3), bk.KLUCB(c=0.0)):
        copy = base.clone(policy)
        assert type(copy) is type(policy) and copy.get_params() == policy.get_params()
