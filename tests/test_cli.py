"""Results folders, the dashboard and the command line."""
import json

import numpy as np
import pytest

import banditkit as bk
from banditkit import cli, report

pytestmark = pytest.mark.filterwarnings("ignore::banditkit.ParameterWarning")

CONFIG = {
    "description": "Tiny test problem.",
    "env": {"kind": "bernoulli", "n_instances": 2, "n_actions": 3, "seed": 0},
    "policies": [{"policy": "UCB1"}, {"policy": "KLUCB", "c": 0.0, "label": "KL-UCB c=0"}, {"policy": "Random"}],
    "experiment": {"horizon": 200, "n_runs": 2, "seed": 0, "log_interval": 10, "verbose": 0},
}


def test_run_config_writes_a_results_folder(tmp_path):
    folder = cli.run_config(CONFIG, tmp_path / "tiny", command="python -m banditkit run tiny/config.json")
    assert {p.name for p in folder.iterdir()} == {"config.json", "results.npz", "summary.json", "explain.json"}
    explain = json.loads((folder / "explain.json").read_text())  # "Inside the policies" charts of the dashboard
    assert {(x["policy"], x["kind"]) for x in explain} >= {("UCB1", "why"), ("UCB1", "allocation"), ("KL-UCB c=0", "why")}
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["name"] == "tiny" and summary["problem"]["n_runs"] == 4 and summary["problem"]["n_instances"] == 2
    assert [p["label"] for p in summary["policies"]] == ["UCB1", "KL-UCB c=0", "Random"]
    assert summary["policies"][1]["params"] == {"c": 0.0}  # only the changed hyperparameters
    curve = summary["curves"]["regret"]["UCB1"]
    assert len(curve["mean"]) == len(summary["steps"]) and all(lo <= m for lo, m in zip(curve["low"], curve["mean"]))
    final = summary["final"]["regret"]["Random"]
    assert final["low"] <= final["mean"] <= final["high"] and len(final["values"]) == 4
    # The saved configuration reruns the same experiment.
    again = cli.run_config(json.loads((folder / "config.json").read_text()), tmp_path / "again")
    np.testing.assert_allclose(bk.Results.load(again / "results.npz").final()["UCB1"],
                               bk.Results.load(folder / "results.npz").final()["UCB1"])


def test_dashboard_embeds_every_folder(tmp_path):
    for name in ("b-run", "a-run"):
        cli.run_config(CONFIG, tmp_path / "results" / name)
    (tmp_path / "results" / "a-run" / "summary.json").unlink()  # rebuilt from results.npz
    assert [p.name for p in report.find_runs(tmp_path / "results")] == ["a-run", "b-run"]
    page = report.build_dashboard(results_dir=tmp_path / "results", out=tmp_path / "dash.html").read_text()
    data = json.loads(page.split('<script id="bk-data" type="application/json">')[1].split("</script>")[0])
    assert [s["name"] for s in data] == ["a-run", "b-run"]
    assert "cdn.jsdelivr.net/npm/plotly.js-dist-min" in page
    # The fonts travel inside the page: nothing to fetch, so they also show offline and in sandboxed viewers.
    assert page.count("@font-face") == 4 and 'font-family: "Archivo"' in page and "{{FONTS}}" not in page
    assert "fonts.googleapis.com" not in page
    only = report.build_dashboard([tmp_path / "results" / "b-run"], out=tmp_path / "one.html").read_text()
    assert '"name":"b-run"' in only and '"name":"a-run"' not in only


def test_command_line(tmp_path, capsys):
    config = tmp_path / "small.json"
    config.write_text(json.dumps(CONFIG))
    dash = tmp_path / "dashboard.html"
    assert cli.main(["run", str(config), "--results-dir", str(tmp_path / "results"), "--horizon", "100",
                     "--generate-dashboard", "--dashboard", str(dash)]) == 0
    assert (tmp_path / "results" / "small" / "summary.json").exists() and dash.exists()
    saved = json.loads((tmp_path / "results" / "small" / "config.json").read_text())
    assert saved["experiment"]["horizon"] == 100
    assert cli.main(["dashboard", str(tmp_path / "results" / "small"), "--out", str(tmp_path / "d2.html")]) == 0
    assert "Dashboard:" in capsys.readouterr().out


def test_build_env_kinds():
    envs = cli.build_env({"kind": "multi_user", "n_instances": 2, "n_actions": 30, "n_features": 4, "n_users": 9,
                          "n_clusters": 3, "n_candidates": 5, "seed": 1})
    assert len(envs) == 2 and envs[0].spec.n_users == 9 and envs[0].spec.n_actions == 5
    thetas = envs[0].user_thetas[:, 0]
    np.testing.assert_allclose(thetas[0], thetas[3])  # users 0 and 3 share cluster 0
    env = cli.build_env({"kind": "make_env", "means": [0.1, 0.5], "noise_std": 0.1})
    assert isinstance(env, bk.GaussianMAB)
    with pytest.raises(ValueError, match="Unknown env kind"):
        cli.build_env({"kind": "nope"})
    with pytest.raises(ValueError, match="Unknown dataset function"):
        cli.build_env({"kind": "dataset", "function": "nope"})
    with pytest.raises(ValueError, match="Unknown policy"):
        cli.build_policies([{"policy": "NotAPolicy"}])


def test_linear_env_from_means():
    X = np.eye(3)
    env = bk.LinearEnv.from_means(X, [0.2, 0.9, 0.4], noise_std=0.0)
    obs, _ = env.reset(seed=0)
    assert env.theta is None and env.spec.theta_norm is None
    np.testing.assert_allclose(env.mean_rewards(obs)[:, 0], [0.2, 0.9, 0.4])
    res = bk.Experiment(env, [bk.LinUCB()], horizon=50, seed=0, verbose=0).run()
    assert np.isfinite(res.final()["LinUCB"]).all()


def test_run_several_configs(tmp_path):
    for name in ("one", "two"):
        (tmp_path / f"{name}.json").write_text(json.dumps(CONFIG))
    args = ["run", str(tmp_path / "one.json"), str(tmp_path / "two.json"), "--results-dir", str(tmp_path / "res"),
            "--horizon", "50"]
    assert cli.main(args) == 0
    assert sorted(p.name for p in (tmp_path / "res").iterdir()) == ["one", "two"]
    # Rerun what results/ holds: each folder keeps its name.
    assert cli.main(["run", *map(str, sorted((tmp_path / "res").glob("*/config.json"))), "--results-dir",
                     str(tmp_path / "res"), "--horizon", "50"]) == 0
    assert sorted(p.name for p in (tmp_path / "res").iterdir()) == ["one", "two"]
    with pytest.raises(SystemExit):
        cli.main([*args, "--name", "x"])


def test_summary_for_the_dashboard(tmp_path):
    env = bk.sample_instances("multi_user", n_instances=1, n_actions=30, n_features=4, n_users=6, n_clusters=2, seed=0)
    res = bk.Experiment(env, {"LinUCB": bk.LinUCB(user_model="individual"), "CLUB": bk.CLUB(), "LinUCB searched": bk.LinUCB()}, horizon=60,
                        n_runs=3, seed=0, verbose=0, track=["clusters"],
                        param_grid={"LinUCB searched": {"alpha": [0.1, 1.0], "lambda_": [0.1, 1.0]}}, search_runs=1).run()
    s = report.summarize(res, "x")
    assert s["schema"] == 2 and s["curves"]["regret"]["CLUB"]["n"] == 3 and len(s["curves"]["regret"]["CLUB"]["sd"]) == len(s["steps"])
    assert s["final"]["regret"]["CLUB"]["sd"] >= 0 and len(s["final"]["regret"]["CLUB"]["values"]) == 3
    assert "n_user_clusters" in s["curves"] and s["metric_groups"]["n_user_clusters"] == "policy state"
    assert "n_buckets" not in s["curves"]  # always 0 without buckets: not worth a chart
    assert list(s["search"]) == ["LinUCB searched"] and len(s["search"]["LinUCB searched"]["rows"]) == 4
    assert [n["key"] for n in s["notices"]][:2] == ["defaults", "search"]


def test_explain_charts_by_problem(tmp_path):
    from banditkit.explain import explain

    lin = bk.sample_instances("linear", 1, n_actions=40, n_features=5, seed=0)[0]
    items = explain(lin, {"LinUCB": bk.LinUCB(), "LinUCB_AB": bk.LinUCB(n_buckets="sqrt")}, horizon=300)
    kinds = {(x["policy"], x["kind"]) for x in items}
    assert ("LinUCB", "ellipsoid") in kinds and ("LinUCB_AB", "buckets") in kinds
    users = bk.sample_instances("multi_user", 1, n_actions=30, n_features=4, n_users=6, n_clusters=2, seed=0)[0]
    items = explain(users, {"CLUB": bk.CLUB(), "LinUCB_IND": bk.LinUCB(user_model="individual")}, horizon=200,
                    n_true_groups=2)
    assert [(x["policy"], x["kind"]) for x in items] == [(None, "clusters")]
    assert all("title" not in x["figure"]["layout"] or not x["figure"]["layout"]["title"] for x in items)
