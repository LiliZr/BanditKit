import math
from collections import Counter

import numpy as np
import pytest

import banditkit as bk
from banditkit._utils import pareto_front, sherman_morrison_
from banditkit.policies._linear_core import ActionBuckets, RidgeModel, make_projection


# ----------------------------------------------------------------- utilities
def test_sherman_morrison_matches_inverse():
    rng = np.random.default_rng(0)
    A = np.eye(4) * 2.0
    inv = np.linalg.inv(A)
    for _ in range(10):
        x = rng.normal(size=4)
        A += np.outer(x, x)
        sherman_morrison_(inv, x)
    np.testing.assert_allclose(inv, np.linalg.inv(A), atol=1e-10)


def test_ridge_model_matches_closed_form():
    rng = np.random.default_rng(1)
    Z, r = rng.normal(size=(50, 5)), rng.normal(size=50)
    m = RidgeModel(5, lambda_=0.5)
    for z, y in zip(Z, r):
        m.update(z, y)
    expected = np.linalg.solve(0.5 * np.eye(5) + Z.T @ Z, Z.T @ r)
    np.testing.assert_allclose(m.theta, expected, atol=1e-9)


def test_pareto_front():
    pts = np.array([[1, 0], [0, 1], [0.5, 0.5], [0.2, 0.2], [1, 0]])
    assert set(pareto_front(pts)) == {0, 1, 2, 4}
    assert set(pareto_front(np.array([[np.inf, np.inf], [np.inf, np.inf], [1, 1]]))) == {0, 1}


def test_pareto_gaps():
    env = bk.make_env(means=[[1.0, 0.0], [0.0, 1.0], [0.2, 0.1]])
    obs, _ = env.reset(0)
    gaps = env.pareto_gaps(obs)
    np.testing.assert_allclose(gaps, [0.0, 0.0, 0.0])  # arm 2 is not dominated by a margin on both axes
    env = bk.make_env(means=[[1.0, 1.0], [0.7, 0.9]])
    obs, _ = env.reset(0)
    np.testing.assert_allclose(env.pareto_gaps(obs), [0.0, 0.1])


def test_clipped_gaussian_mean_matches_monte_carlo():
    env = bk.GaussianMAB([[0.9], [0.1]], noise_std=0.3, clip=(0.0, 1.0))
    obs, _ = env.reset(0)
    draws = np.array([env.step(0)[1][0] for _ in range(40000)])
    assert abs(draws.mean() - env.mean_rewards(obs)[0, 0]) < 0.005


def test_projection_preserves_norms_roughly():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 128))
    for kind in ("gaussian", "fjlt"):
        P = make_projection(128, 64, kind, rng, n_points=200)
        ratio = np.linalg.norm(X @ P.T, axis=1) / np.linalg.norm(X, axis=1)
        assert 0.75 < np.median(ratio) < 1.25, kind


def test_buckets_partition_all_arms():
    rng = np.random.default_rng(3)
    b = ActionBuckets(n_actions=103, n_buckets=10)
    scores = rng.normal(size=103)
    b.rebuild(scores, rng)
    allarms = np.sort(np.concatenate(b.groups))
    np.testing.assert_array_equal(allarms, np.arange(103))
    for g, champ in zip(b.groups, b.champions):
        assert scores[champ] == scores[g].max()


# ------------------------------------------------------------------ MAB indices
def run_policy(env, policy, steps, seed=0):
    """Gymnasium-style loop: predict, step, update."""
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        action = policy.predict(obs)
        next_obs, reward, _, _, _ = env.step(action)
        policy.update(obs, action, reward)
        obs = next_obs


def _played(policy, rewards):
    policy.reset(bk.EnvSpec(n_actions=len(rewards), horizon=1000), seed=0)
    for action, rs in enumerate(rewards):
        for r in rs:
            policy.update(bk.Observation(t=0), action, [r])
    return policy


def test_ucb1_index():
    p = _played(bk.UCB1(alpha=1.0), [[0.0, 1.0], [0.5, 0.5, 0.5]])
    p.predict(bk.Observation(t=5))
    expected = np.sqrt(2 * math.log(5) / np.array([2, 3]))
    np.testing.assert_allclose(p.diagnostics()["bonus"], expected)


def test_ucbv_index():
    p = _played(bk.UCBV(), [[0.0, 1.0], [0.5, 0.5, 0.5]])
    p.predict(bk.Observation(t=5))
    e = 1.2 * math.log(6)  # E = zeta ln t at round t = 6 (5 pulls done)
    var = np.array([0.25, 0.0])
    n = np.array([2, 3])
    np.testing.assert_allclose(p.diagnostics()["bonus"], np.sqrt(2 * var * e / n) + 3 * e / n)


def test_ucbmin_is_minimum_of_bonuses():
    rewards = [[0.0, 1.0, 0.2], [0.5, 0.52, 0.48, 0.5]]
    pmin = _played(bk.UCBMinV2(), rewards)
    p1 = _played(bk.UCB1(), rewards)
    pv = _played(bk.UCBV2(), rewards)
    for p in (pmin, p1, pv):
        p.predict(bk.Observation(t=7))
    # UCB-min evaluates both bonuses at the current round (8 here, 7 pulls done); UCB1 uses the 7 pulls.
    b1 = p1.diagnostics()["bonus"] * math.sqrt(math.log(8) / math.log(7))
    bv = pv.diagnostics()["bonus"]
    np.testing.assert_allclose(pmin.diagnostics()["bonus"], np.minimum(b1, bv))
    np.testing.assert_array_equal(pmin.diagnostics()["variance_bonus_used"], bv < b1)


@pytest.mark.parametrize("policy", [bk.UCB1(), bk.UCB1Tuned(), bk.UCBV(), bk.UCBV2(), bk.UCBMinV1(),
                                    bk.UCBMinV2(), bk.EUCBV(), bk.KLUCB(), bk.ThompsonSampling(),
                                    bk.UCB1Normal()])
def test_mab_policies_beat_random(policy):
    env = bk.BernoulliMAB([0.2, 0.3, 0.8])
    res = bk.Experiment(env, [policy, bk.Random()], horizon=3000, n_runs=2, seed=0).run()
    f = res.final("regret")
    labels = res.labels
    assert f[labels[0]].mean() < 0.5 * f[labels[1]].mean()


def test_eucbv_eliminates_bad_arms():
    p = bk.EUCBV()
    env = bk.BernoulliMAB([0.1, 0.9])
    p.reset(env.spec.with_horizon(20000), seed=0)
    run_policy(env, p, steps=4000, seed=0)
    assert p.active.tolist() == [False, True]


def test_pareto_ucb_finds_front():
    env = bk.make_env(means=[[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.2, 0.2]], noise_std=0.05)
    res = bk.Experiment(env, bk.ParetoUCB1(), horizon=4000, seed=0, track=("pareto_regret",)).run()
    counts = res.arrays["ParetoUCB1"]["action_counts"][0]
    assert counts[3] < 0.05 * counts.sum()
    assert res.final("pareto_regret")["ParetoUCB1"][0] < 0.05 * 4000 * 0.3


# --------------------------------------------------------------------- sketches
def _run_sketch(policy, X, rewards):
    policy.reset(bk.EnvSpec(n_actions=X.shape[0], n_features=X.shape[1]), seed=0)
    obs = bk.Observation(t=0, action_features=X)
    for a, r in rewards:
        policy.update(obs, a, [r])
    return policy


@pytest.mark.parametrize("cls", [bk.SOFUL, bk.CBSCFD])
def test_sketch_estimate_matches_its_design_matrix(cls):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 12))
    pulls = [(int(rng.integers(30)), float(rng.normal())) for _ in range(40)]
    p = _run_sketch(cls(sketch_size=4, lambda_=0.7), X, pulls)
    V = p.regularization * np.eye(12) + p.sketch.T @ p.sketch
    ar = sum(r * X[a] for a, r in pulls)
    np.testing.assert_allclose(p.theta, np.linalg.solve(V, ar), atol=1e-8)
    p._working_features(bk.Observation(t=0, action_features=X))
    widths = np.sqrt(np.einsum("ij,jk,ik->i", X, np.linalg.inv(V), X))
    np.testing.assert_allclose(p._widths(X), widths, atol=1e-8)


def test_frequent_directions_error_bound():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 10))
    pulls = [(i % 60, 0.0) for i in range(60)]
    p = _run_sketch(bk.SOFUL(sketch_size=4), X, pulls)
    A = X[[a for a, _ in pulls]]
    gap = np.linalg.eigvalsh(A.T @ A - p.sketch.T @ p.sketch)
    assert gap.min() > -1e-8                      # sketch never over-estimates
    assert gap.max() <= p.sketch_error + 1e-8     # error bounded by the total shrinkage


# ------------------------------------------------------------------- clustering
def _users_env(U=60, K=40, d=8, C=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(K, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    centers = rng.normal(size=(C, d))
    return bk.make_env(action_features=X, user_thetas=centers[rng.integers(C, size=U)], noise_std=0.1)


def test_sclub_bookkeeping_invariants():
    env = _users_env()
    p = bk.SCLUB(cluster_alpha=0.3, n_components=4, n_buckets=4)
    p.reset(env.spec.with_horizon(3000), seed=1)
    run_policy(env, p, steps=3000, seed=1)
    assert sorted(u for m in p.members.values() for u in m) == list(range(env.spec.n_users))
    for j, members in p.members.items():
        counts = [p.user_models[u].n_obs if u in p.user_models else 0 for u in members]
        assert Counter(counts) == +p._member_counts[j]
        assert p._min_count[j] == min(counts) and p._max_count[j] == max(counts)
        assert p._n_checked[j] == sum(p._checked[u] for u in members)
        c = p.cluster_models.get(j)
        if c is not None:
            assert c.n_obs == sum(counts)
            eye = p.lambda_ * np.eye(p.dim)
            V = eye + sum((p.user_models[u].V - eye) for u in members if u in p.user_models)
            np.testing.assert_allclose(c.V, V, atol=1e-8)


def test_club_components_match_graph():
    env = _users_env()
    p = bk.CLUB(cluster_alpha=0.2)
    p.reset(env.spec.with_horizon(2000), seed=2)
    run_policy(env, p, steps=2000, seed=2)
    for cid, members in p.components.items():
        assert len(p._connected_components(members)) == 1
        assert all(p.component_of[u] == cid for u in members)
    assert sum(len(m) for m in p.components.values()) == env.spec.n_users


def test_clustering_policies_learn():
    env = _users_env(U=30, C=2)
    res = bk.Experiment(env, [bk.SCLUB(), bk.CLUB(), bk.LinUCB(user_model="kmeans", n_user_clusters=2,
                                                                  user_recluster_freq=300), bk.Random()],
                        horizon=3000, n_runs=1, seed=0).run()
    f = res.final("regret")
    rnd = f["Random"][0]
    for label in res.labels[:-1]:
        assert f[label][0] < 0.6 * rnd, label


# ------------------------------------------------- values of the papers
def test_ucb1_normal_uses_the_round_index():
    rewards = [[0.1, 0.3, 0.2] * 6, [0.5, 0.7, 0.6] * 6]  # 36 pulls: ceil(8 ln 37) = 29 > 18
    p = _played(bk.UCB1Normal(), rewards)
    assert p.predict(bk.Observation(t=36)) in (0, 1)  # both under-sampled: forced exploration
    q = _played(bk.UCB1Normal(), [[0.1, 0.3] * 20, [0.5, 0.7] * 20])  # 80 pulls, 40 each >= ceil(8 ln 81) = 36
    q.predict(bk.Observation(t=80))
    s2 = np.array([np.var([0.1, 0.3] * 20, ddof=1), np.var([0.5, 0.7] * 20, ddof=1)])
    np.testing.assert_allclose(q.diagnostics()["bonus"], np.sqrt(16 * s2 * math.log(80) / 40))


def test_lints_scale_of_the_paper():
    env = bk.LinearEnv(np.eye(4), np.array([0.5, 0.1, 0.0, 0.0]), noise_std=0.2)
    p = bk.LinTS(lambda_=1.0)
    p.reset(env.spec, seed=0)
    assert math.isclose(p._scale(), 0.2 * math.sqrt(9 * 4 * math.log(1 / 0.01)))


def test_gaussian_ts_default_scale_is_the_reward_amplitude():
    env = bk.GaussianMAB([0.2, 0.5], noise_std=0.05)
    p = bk.GaussianThompsonSampling(reward_range=4.0)
    p.reset(env.spec)
    assert p._sigma == 4.0
    q = bk.GaussianThompsonSampling(sigma=0.05)
    q.reset(env.spec)
    assert q._sigma == 0.05 and not any("reward_range" in text for _, text in q.parameter_notes())


def test_soful_shrinks_at_every_step():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    env = bk.LinearEnv(X, rng.normal(size=8), noise_std=0.1)
    obs, _ = env.reset(seed=0)
    p = bk.SOFUL(sketch_size=3)
    p.reset(env.spec, seed=0)
    for step in range(12):
        p.update(obs, step % 50, 0.0)
        assert p.sketch.shape[0] <= 2  # at most m - 1 non-zero rows
        if step < 2:
            assert p.sketch_error == 0.0  # fewer than m rows: exact OFUL
    assert p.sketch_error > 0.0
    # The Woodbury inverse matches (lambda I + S^T S)^-1.
    V = p.lambda_ * np.eye(8) + p.sketch.T @ p.sketch
    widths = np.sqrt(np.einsum("ij,ji->i", X, np.linalg.solve(V, X.T)))
    np.testing.assert_allclose(p._widths(X), widths, rtol=1e-8)


def test_sclub_theoretical_threshold():
    X = np.eye(4) * 2.0  # E[x x^T] = I: lambda_x = 1
    env = bk.MultiUserLinearEnv(X, np.ones((3, 4)) * 0.1, noise_std=0.3)
    p = bk.SCLUB(bonus="oful")
    p.reset(env.spec, seed=0)
    obs, _ = env.reset(seed=0)
    p.predict(obs)
    assert math.isclose(p._alpha_theta, 4 * 0.3 * math.sqrt(4 / 1.0))
