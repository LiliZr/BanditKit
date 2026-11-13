"""Every figure builds without error on small runs (smoke tests)."""
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

import banditkit as bk
from banditkit import plot
from banditkit.plot import interactive as iplot

pytest.importorskip("plotly")


@pytest.fixture(scope="module")
def linear_results():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    env = bk.make_env(action_features=X / np.linalg.norm(X, axis=1, keepdims=True), theta=rng.normal(size=6))
    return bk.Experiment(env, [bk.LinUCB(), bk.LinUCB(n_components=3), bk.LinTS()], horizon=300, n_runs=3,
                         seed=0, measure_memory=True).run()


@pytest.mark.parametrize("band", ["std", "ci95", "quantiles", None])
def test_curves_bands(linear_results, band):
    fig, ax = plot.curves(linear_results, band=band, log_y=True, labels={"LinUCB": "OFUL"})
    assert ax.get_legend() is not None


def test_publication_figures(linear_results):
    plot.panel(linear_results)
    plot.tradeoff(linear_results)
    for kind in ("box", "violin", "strip"):
        plot.dispersion(linear_results, kind=kind)
    plot.sensitivity({0.5: linear_results, 1.0: linear_results}, "alpha")
    plot.performance_profile({"a": linear_results, "b": linear_results})
    plot.curves(linear_results, log_x=True)


def test_pareto_panel():
    mo = bk.make_env(means=[[0.9, 0.1], [0.1, 0.9], [0.4, 0.4], [0.5, 0.45]], noise_std=0.1)
    res = bk.Experiment(mo, [bk.ParetoUCB1(), bk.UCB1()], horizon=300, n_runs=2, seed=0,
                        track=["pareto_share", "pareto_regret", "unfairness"]).run()
    share = res.final("pareto_share")
    assert all(np.all((v >= 0) & (v <= 1)) for v in share.values())
    fig, axes = plot.pareto_panel(res)
    assert len(axes) == 3
    with pytest.raises(ValueError, match="track"):
        plot.pareto_panel(bk.Experiment(mo, [bk.UCB1()], horizon=50).run())


def test_interactive_figures():
    mab = bk.BernoulliMAB([0.2, 0.5, 0.7])
    ucb = bk.trace(mab, bk.UCB1(), 300, every=10)
    ts = bk.trace(mab, bk.ThompsonSampling(), 300, every=10)
    for fig in (iplot.allocation(ucb), iplot.estimates(ucb), iplot.why_this_arm(ucb), iplot.beliefs(ts),
                iplot.prob_best(ts), iplot.regret_by_arm(ucb)):
        assert fig.data
    het = bk.make_env(means=[0.3, 0.5, 0.6], noise_std=[0.01, 0.3, 0.05])
    assert iplot.ucbmin_choice(bk.trace(het, bk.UCBMinV1(), 300, every=10)).data
    mo = bk.make_env(means=[[0.9, 0.1], [0.1, 0.9], [0.4, 0.4]], noise_std=0.1)
    assert iplot.scalarization(mo, weights=[0.5, 0.5]).data
    assert iplot.scalarization(bk.trace(mo, bk.ParetoUCB1(), 50)).data
    with pytest.raises(ValueError, match="two objectives"):
        iplot.scalarization(np.ones((3, 3)))
    lin2d = bk.make_env(action_features=np.eye(2), theta=[0.5, 0.2])
    assert iplot.ellipsoid(bk.trace(lin2d, bk.LinUCB(), 100, every=5)).frames
    X = np.random.default_rng(0).normal(size=(30, 4))
    users = bk.make_env(action_features=X, user_thetas=np.random.default_rng(1).normal(size=(10, 4)))
    assert iplot.clusters([bk.trace(users, bk.SCLUB(), 300, every=10)], n_true_groups=2).data
    lin = bk.make_env(action_features=X, theta=np.ones(4))
    fig = iplot.bucket_choice(bk.trace(lin, bk.LinUCB(n_buckets=3), 100, every=10))
    assert fig.frames and len(fig.frames[0].data) == 7
    users_ab = bk.trace(users, bk.LinUCB(user_model="individual", n_buckets=3), 100, every=10)
    assert iplot.bucket_choice(users_ab).frames


def test_figure_needs_matching_policy():
    tr = bk.trace(bk.BernoulliMAB([0.2, 0.5]), bk.Random(), 50)
    with pytest.raises(ValueError, match="does not record"):
        iplot.why_this_arm(tr)


def test_search_landscape_and_legend_table():
    env = bk.make_env(action_features=np.eye(4), theta=[0.1, 0.5, 0.2, 0.0], noise_std=0.1)
    res = bk.Experiment(env, [bk.LinUCB(), bk.LinTS()], horizon=100, n_runs=2, seed=0, verbose=0,
                        param_grid={"alpha": [0.1, 1.0], "lambda_": [0.01, 0.1]}, search_runs=1).run()
    fig, ax = plot.search_landscape(res, "LinUCB")
    assert ax.get_xscale() == "log"
    fig, ax = plot.curves(res)
    assert ax.get_legend().get_title().get_text().startswith("best")
    with pytest.raises(ValueError, match="No parameter search"):
        plot.search_landscape(bk.Experiment(env, [bk.LinUCB()], horizon=10, verbose=0).run(), "LinUCB")
