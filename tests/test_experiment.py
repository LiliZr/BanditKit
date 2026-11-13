import numpy as np
import pytest

import banditkit as bk


def small_lin_env():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    return bk.make_env(action_features=X / np.linalg.norm(X, axis=1, keepdims=True), theta=rng.normal(size=5))


def run(env, policies, **kw):
    kw.setdefault("horizon", 400)
    kw.setdefault("n_runs", 2)
    kw.setdefault("seed", 7)
    return bk.Experiment(env, policies, **kw).run()


def test_same_seed_same_results():
    env = small_lin_env()
    a = run(env, [bk.LinUCB(), bk.ThompsonSampling()])
    b = run(env, [bk.LinUCB(), bk.ThompsonSampling()])
    for label in a.labels:
        np.testing.assert_array_equal(a.curve(label), b.curve(label))


def test_parallel_matches_sequential():
    env = small_lin_env()
    seq = run(env, [bk.LinUCB(), bk.LinUCB(n_components=3)])
    par = run(env, [bk.LinUCB(), bk.LinUCB(n_components=3)], n_jobs=2)
    for label in seq.labels:
        np.testing.assert_array_equal(seq.curve(label), par.curve(label))


def test_common_random_numbers_across_policies():
    # Same arm sequence => same rewards: two copies of a deterministic policy coincide.
    env = small_lin_env()
    res = run(env, {"a": bk.LinUCB(), "b": bk.LinUCB()})
    np.testing.assert_array_equal(res.curve("a"), res.curve("b"))


def test_recorded_steps_and_regret_shapes():
    res = run(bk.make_env(means=[0.1, 0.9]), bk.UCB1(), horizon=1000, log_interval=100,
              track=("pseudo_regret", "actions", "estimates"))
    assert res.steps.tolist() == list(range(100, 1001, 100))
    assert res.curve("UCB1").shape == (2, 10)
    assert res.arrays["UCB1"]["actions"].shape == (2, 1000)
    assert res.curves["UCB1"]["mean"].shape == (2, 10, 2)
    assert np.all(np.diff(res.curve("UCB1", "pseudo_regret"), axis=1) >= -1e-12)


def test_regret_definition():
    # Deterministic rewards: regret = sum over rounds of (best mean - reward).
    env = bk.make_env(means=[0.2, 0.7], noise_std=0.0)
    res = run(env, bk.Random(), horizon=500, n_runs=1, track=("actions",))
    actions = res.arrays["Random"]["actions"][0]
    expected = np.sum(0.7 - np.where(actions == 1, 0.7, 0.2))
    assert res.final("regret")["Random"][0] == pytest.approx(expected)


def test_scalarization_weights():
    env = bk.make_env(means=[[1.0, 0.0], [0.0, 1.0]], noise_std=0.0)
    res = run(env, bk.UCB1(), horizon=300, n_runs=1, weights=[0.2, 0.8])
    counts = res.arrays["UCB1"]["action_counts"][0]
    assert counts[1] > counts[0]


def test_timeout_stops_runs():
    res = run(small_lin_env(), bk.ConfidenceBall1(), horizon=200000, n_runs=1, timeout=0.05)
    assert res.arrays["ConfidenceBall1"]["n_steps"][0] < 200000
    assert np.isnan(res.curve("ConfidenceBall1")[0, -1])
    assert not np.isnan(res.final("regret")["ConfidenceBall1"][0])


def test_batched_updates():
    res = run(small_lin_env(), bk.LinUCB(), train_freq=10)
    assert res.final("regret")["LinUCB"].shape == (2,)


def test_checkpoint_resume(tmp_path):
    env = small_lin_env()
    calls = []
    a = run(env, bk.LinUCB(), checkpoint_dir=tmp_path, callback=calls.append)
    assert len(calls) == 2
    calls.clear()
    b = run(env, bk.LinUCB(), checkpoint_dir=tmp_path, callback=calls.append)
    assert calls == []  # everything loaded from the checkpoints
    np.testing.assert_array_equal(a.curve("LinUCB"), b.curve("LinUCB"))


def test_save_load_merge_and_latex(tmp_path):
    env = small_lin_env()
    a = run(env, [bk.LinUCB(), bk.Random()], seed=1)
    path = a.save(tmp_path / "res")
    b = bk.Results.load(path)
    np.testing.assert_array_equal(a.curve("LinUCB"), b.curve("LinUCB"))
    assert b.meta["horizon"] == 400
    m = bk.Results.merge(a, run(env, [bk.LinUCB(), bk.Random()], seed=2))
    assert m.curve("LinUCB").shape[0] == 4
    tex = a.to_latex()
    assert "\\begin{tabular}" in tex and "LinUCB" in tex
    df = a.to_dataframe(metrics=["regret"])
    assert set(df.columns) == {"policy", "run", "step", "metric", "value"}


def test_memory_measurement():
    res = run(small_lin_env(), bk.LinUCB(), measure_memory=True)
    assert np.all(res.curve("LinUCB", "memory_mb") > 0)


def test_partial_fit_warm_start():
    env = small_lin_env()
    obs, _ = env.reset(seed=0)
    actions = np.arange(10) % env.spec.n_actions
    rewards = [env.mean_reward(a, obs) for a in actions]
    policy = bk.LinUCB()
    policy.reset(env.spec)
    policy.partial_fit(actions, rewards, action_features=obs.action_features)
    assert policy.models[0].n_obs == 10 and policy.n_samples_seen_ == 10


def test_trace_records_every_step():
    tr = bk.trace(bk.BernoulliMAB([0.2, 0.8]), bk.UCB1(), horizon=200, every=10)
    assert tr.actions.shape == (200,) and tr.rewards.shape == (200, 1)
    assert tr.snapshots["bonus"].shape == (tr.snapshot_steps.size, 2)
    assert np.all(np.diff(tr.regret) > -1.0)


def test_group_recommendation_means_are_exact():
    rng = np.random.default_rng(0)
    R = rng.uniform(size=(6, 3))
    env = bk.GroupRecommendationMAB(R)
    obs, _ = env.reset(seed=1)
    draws = np.array([env.step(1)[1] for _ in range(60000)])
    np.testing.assert_allclose(draws.mean(axis=0), env.mean_rewards(obs)[1], atol=0.01)
    gaps = [abs(R[u, 1] - R[v, 1]) for u in range(6) for v in range(6) if u != v]
    assert env.mean_rewards(obs)[1, 1] == pytest.approx(1 - np.mean(gaps))


def test_pareto_share_and_unfairness():
    # Arms 0 and 1 are Pareto optimal, arm 2 is dominated by both.
    env = bk.make_env(means=[[0.9, 0.1], [0.1, 0.9], [0.05, 0.05]], noise_std=0.1)
    res = bk.Experiment(env, [bk.Random()], horizon=300, n_runs=2, seed=0, log_interval=300,
                        track=["pareto_share", "unfairness", "actions"]).run()
    actions = res.arrays["Random"]["actions"]
    counts = np.stack([np.bincount(a, minlength=3) for a in actions])
    assert np.allclose(res.final("pareto_share")["Random"], counts[:, :2].sum(axis=1) / 300)
    assert np.allclose(res.final("unfairness")["Random"], counts[:, :2].var(axis=1))


def test_instances_and_compare():
    envs = bk.sample_instances("linear", 3, n_actions=20, n_features=4, noise_std=0.1, min_gap=0.05, seed=0)
    assert len(envs) == 3 and all(np.allclose(e.spec.theta_norm, 1.0) for e in envs)
    res = bk.Experiment(envs, [bk.LinUCB(), bk.Random()], horizon=200, n_runs=2, seed=0, verbose=0).run()
    assert res.n_runs == 6 and res.n_instances == 3
    assert list(res.arrays["LinUCB"]["instance"]) == [0, 0, 1, 1, 2, 2]
    assert res.instance_means()["LinUCB"].shape == (3,)
    cmp = res.compare("LinUCB", "Random")
    assert cmp["pairs"] == "instances" and cmp["a_lower_share"] == 1.0
    parts = res.by_instance()
    assert set(parts) == {0, 1, 2} and parts[1].n_runs == 2
    for kind in ("bernoulli", "gaussian"):
        mabs = bk.sample_instances(kind, 4, n_actions=5, min_gap=0.1, seed=1)
        assert all(np.diff(np.sort(e.mean_rewards(e.reset(seed=0)[0])[:, 0]))[-1] >= 0.1 for e in mabs)
    with pytest.raises(ValueError, match="same kind"):
        bk.Experiment([envs[0], bk.make_env(means=[0.1, 0.2])], [bk.Random()], horizon=10)
    with pytest.raises(ValueError, match="n_actions for MABs"):
        bk.Experiment([bk.make_env(means=[0.1, 0.2, 0.3]), bk.make_env(means=[0.1, 0.2])], [bk.Random()], horizon=10)
    # Linear instances may offer different numbers of actions (e.g. the movies each user rated).
    X = np.random.default_rng(0).normal(size=(30, 4))
    uneven = [bk.make_env(action_features=X[:k], theta=np.ones(4) / 2, noise_std=0.1) for k in (30, 20)]
    res = bk.Experiment(uneven, [bk.LinUCB()], horizon=50, seed=0, verbose=0).run()
    assert res.arrays["LinUCB"]["action_counts"].shape == (2, 30)  # padded with zeros
    assert res.arrays["LinUCB"]["action_counts"][1, 20:].sum() == 0 and res.meta["n_actions_range"] == [20, 30]
    with pytest.raises(ValueError, match="same number of actions"):
        bk.Experiment(uneven, [bk.LinUCB()], horizon=10, track=["estimates"])


def test_param_search_picks_the_best_combination():
    env = bk.make_env(means=[0.3, 0.5, 0.7], noise_std=0.1)
    res = bk.Experiment(env, [bk.UCB1(), bk.GaussianThompsonSampling()], horizon=300, n_runs=2, seed=0, verbose=0,
                        param_grid={"alpha": [0.01, 100.0]}, search_runs=2).run()
    table = res.search_table()
    ucb = table[table.policy == "UCB1"]
    assert len(ucb) == 2 and ucb.loc[ucb["mean"].idxmin(), "best"]
    assert res.best_params["UCB1"] == {"alpha": float(ucb.loc[ucb["mean"].idxmin(), "alpha"])}
    assert res.best_params["GaussianThompsonSampling"] == {}  # no alpha: nothing to search
    assert res.meta["policies"]["UCB1"]["params"]["alpha"] == res.best_params["UCB1"]["alpha"]
    with pytest.raises(ValueError, match="unknown policies"):
        bk.Experiment(env, [bk.UCB1()], horizon=10, param_grid={"Nope": {"alpha": [1.0]}})


def test_parameter_notes():
    lin = bk.make_env(action_features=np.eye(3), theta=[0.3, 0.4, 0.0], noise_std=0.2)
    p = bk.LinUCB()
    p.reset(lin.spec, seed=0)
    notes = dict((text.split(":")[0].split("=")[0], kind) for kind, text in p.parameter_notes())
    assert notes == {"subgaussian": "from_problem", "theta_bound": "from_problem", "lambda_": "arbitrary"}
    assert p._R == pytest.approx(0.2) and p._S == pytest.approx(0.5)
    q = bk.MABEpsilonGreedy()
    q.reset(bk.make_env(means=[0.1, 0.2]).spec, seed=0)
    assert ("arbitrary" in dict(q.parameter_notes())) and any("epsilon" in t for _, t in q.parameter_notes())


def test_launch_warning_adapts_to_the_settings(capsys, monkeypatch):
    from banditkit.experiment import WARNING_TITLE

    env = bk.make_env(means=[0.2, 0.5, 0.7], noise_std=0.1)
    monkeypatch.setenv("NO_COLOR", "1")
    res = bk.Experiment(env, [bk.UCB1(), bk.Random()], horizon=5, seed=0, log_interval=2).run()
    err = capsys.readouterr().err
    assert WARNING_TITLE in err and "\033[" not in err  # NO_COLOR: two rules instead of a colored band
    assert " 1) Default hyperparameters. UCB1 keeps" in err and "param_grid=True" in err
    assert "Not stored" in err and 'track "pseudo_regret"' in err and "One run" in err and "log_interval=2" in err
    assert "UCB1: default alpha=1" in err
    keys = [n["key"] for n in res.meta["notices"]]  # kept with the results (the dashboard shows them)
    assert keys == ["defaults", "assumed", "not_stored", "runs", "log_interval"]  # Gaussian rewards: b unknown
    # Values set, pseudo-regret and actions stored, enough runs: only what is left.
    bk.Experiment(env, [bk.UCB1(alpha=0.2), bk.Random()], horizon=5, n_runs=6, seed=0, track=["pseudo_regret", "actions", "estimates"],
                  measure_memory=True).run()
    err = capsys.readouterr().err
    assert "Default hyperparameters" not in err and "Not stored" not in err and "Slower timings" in err
    exp = bk.Experiment(env, [bk.UCB1()], horizon=5, seed=0, param_grid={"alpha": [0.1, 1.0]}, search_runs=1)
    assert [n["key"] for n in exp.notices()][0] == "search"
    exp = bk.Experiment(env, [bk.UCB1(alpha=0.1)], horizon=5, n_runs=2, seed=0, n_jobs=2, timeout=1.0, train_freq=3)
    assert {"runs", "n_jobs", "timeout", "train_freq"} <= {n["key"] for n in exp.notices()}
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    bk.Experiment(env, [bk.UCB1()], horizon=5, seed=0).run()
    assert "\033[1;30;103m " + WARNING_TITLE in capsys.readouterr().err
    bk.Experiment(env, [bk.UCB1()], horizon=5, seed=0, verbose=0).run()
    assert capsys.readouterr().err == ""
