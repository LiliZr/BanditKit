"""Parameters, shapes and problem families are checked early, with clear messages."""
import numpy as np
import pytest

import banditkit as bk


def mab_env():
    return bk.make_env(means=[0.1, 0.5, 0.9], noise_std=0.1)


def linear_env(n_actions=20, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    return bk.make_env(action_features=rng.normal(size=(n_actions, n_features)), theta=rng.normal(size=n_features))


# ------------------------------------------------------------- hyperparameters
def test_unknown_parameter_suggests_closest():
    with pytest.raises(TypeError, match="Did you mean `alpha`"):
        bk.UCB1(alpah=1.0)


@pytest.mark.parametrize("kwargs", [dict(alpha=-1.0), dict(reward_range=0.0)])
def test_out_of_range_values(kwargs):
    with pytest.raises(ValueError):
        bk.UCB1(**kwargs)


def test_wrong_types():
    with pytest.raises(TypeError):
        bk.LinUCB(n_components=2.5)
    with pytest.raises(TypeError):
        bk.LinUCB(bucket_by_ucb="yes")
    with pytest.raises(ValueError):
        bk.LinUCB(projection="pca")


def test_cross_parameter_checks():
    with pytest.raises(ValueError, match="n_user_clusters"):
        bk.LinUCB(user_model="kmeans")
    with pytest.raises(ValueError, match="n_buckets"):
        bk.LinUCB(refresh_champions=True)
    with pytest.raises(ValueError, match="either"):
        bk.UCB1(sigma=1.0, reward_range=1.0)


def test_generated_signature_and_docstring():
    import inspect

    assert "n_components" in inspect.signature(bk.LinUCB).parameters
    assert "lambda_ : float" in bk.LinUCB.__doc__


# ------------------------------------------------------------ problem checks
def test_family_mismatch():
    with pytest.raises(ValueError, match="linear"):
        bk.Experiment(mab_env(), bk.LinUCB(), horizon=10)
    with pytest.raises(ValueError, match="MAB"):
        bk.Experiment(linear_env(), bk.UCB1(), horizon=10)


def test_problem_dependent_checks():
    with pytest.raises(ValueError, match="n_components"):
        bk.Experiment(linear_env(n_features=6), bk.LinUCB(n_components=6), horizon=10)
    with pytest.raises(ValueError, match="n_buckets"):
        bk.Experiment(linear_env(n_actions=20), bk.LinUCB(n_buckets=21), horizon=10)
    with pytest.raises(ValueError, match="sketch_size"):
        bk.Experiment(linear_env(n_features=6), bk.SOFUL(sketch_size=6), horizon=10)


def test_generic_policy_rejects_parameter_of_other_family():
    with pytest.raises(ValueError, match="applies to linear"):
        bk.Experiment(mab_env(), bk.UCB(n_components=3), horizon=10)


def test_generic_policy_resolution_and_labels():
    exp = bk.Experiment(mab_env(), [bk.UCB(), bk.ThompsonSampling()], horizon=10)
    assert list(exp.policies) == ["UCB1", "GaussianThompsonSampling"]
    exp = bk.Experiment(linear_env(), [bk.UCB(), bk.UCB(n_components=3)], horizon=10)
    assert list(exp.policies) == ["LinUCB", "LinUCB_RP"]


def test_horizon_needed():
    policy = bk.UCBV2()
    with pytest.raises(ValueError, match="horizon"):
        policy.reset(bk.EnvSpec(n_actions=3))
    policy.reset(bk.EnvSpec(n_actions=3, horizon=100))


# ---------------------------------------------------------------- environments
def test_environment_inputs_are_checked():
    with pytest.raises(ValueError, match="columns"):
        bk.make_env(action_features=np.ones((5, 3)), theta=np.ones(4))
    with pytest.raises(ValueError, match="finite"):
        bk.make_env(means=[0.1, np.nan])
    with pytest.raises(ValueError):
        bk.make_env(means=[0.1, 1.2], noise="bernoulli")
    with pytest.raises(TypeError, match="exactly one"):
        bk.make_env(action_features=np.ones((5, 3)), theta=np.ones(3), user_thetas=np.ones((2, 3)))
    with pytest.raises(ValueError, match="user_stream"):
        bk.make_env(action_features=np.ones((5, 3)), user_thetas=np.ones((2, 3)), user_stream=[0, 1, 2])
    with pytest.raises(ValueError, match="relevance"):
        bk.GroupRecommendationMAB(np.full((3, 4), 1.5))


def test_make_env_detects_the_family():
    assert bk.make_env(means=[0.1, 0.2]).spec.family == "mab"
    env = bk.make_env(means=[[0.1, 0.2], [0.3, 0.1]])
    assert env.spec.family == "mab" and env.spec.reward_dim == 2
    assert linear_env().spec.family == "linear"
    env = bk.make_env(action_features=np.eye(3), user_thetas=np.ones((4, 2, 3)))
    assert env.spec.family == "multi_user" and env.spec.reward_dim == 2 and env.spec.n_users == 4


def test_gymnasium_style_interface():
    env = bk.make_env(action_features=np.eye(3), user_thetas=np.ones((2, 3)), user_stream=[0, 1, 0])
    obs, info = env.reset(seed=0)
    assert obs.t == 0 and info == {}
    for t in range(3):
        obs, reward, terminated, truncated, _ = env.step(0)
        assert reward.shape == (1,) and not terminated
    assert truncated  # the stream is exhausted
    with pytest.raises(IndexError):
        env.step(5)


def test_horizon_beyond_user_stream():
    env = bk.make_env(action_features=np.eye(3), user_thetas=np.ones((2, 3)), user_stream=[0, 1, 0])
    with pytest.raises(ValueError, match="exceeds"):
        bk.Experiment(env, bk.LinUCB(), horizon=4)


def test_weights_validation():
    env = bk.make_env(means=[[0.1, 0.2], [0.3, 0.1]])
    with pytest.raises(ValueError, match="length"):
        bk.Experiment(env, bk.UCB1(), horizon=5, weights=[1.0])
    with pytest.raises(ValueError, match="non-negative"):
        bk.Experiment(env, bk.UCB1(), horizon=5, weights=[1.0, -1.0])
    with pytest.raises(ValueError, match="positive"):
        bk.Experiment(env, bk.UCB1(), horizon=5, weights=[0.0, 0.0])


def test_reward_shape_checked_in_update():
    policy = bk.UCB1()
    policy.reset(bk.EnvSpec(n_actions=3, reward_dim=2))
    with pytest.raises(ValueError, match="2 component"):
        policy.update(bk.Observation(t=0), 0, [1.0, 2.0, 3.0])


def test_predict_before_reset():
    with pytest.raises(RuntimeError, match="reset"):
        bk.UCB1().predict(bk.Observation(t=0))


def test_experiment_arguments():
    with pytest.raises(ValueError, match="track"):
        bk.Experiment(mab_env(), bk.UCB1(), horizon=5, track=("regrets",))
    with pytest.raises(ValueError, match="n_jobs"):
        bk.Experiment(mab_env(), bk.UCB1(), horizon=5, n_jobs=0)
    with pytest.raises(ValueError, match="pareto_regret"):
        bk.Experiment(mab_env(), bk.UCB1(), horizon=5, track=("pareto_regret",))
