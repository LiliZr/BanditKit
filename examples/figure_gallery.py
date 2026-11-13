"""Build an HTML gallery of every banditkit figure.

Usage: python examples/figure_gallery.py OUTPUT.html [N_JOBS]

Every comparison averages 20 runs or more. Linear comparisons pool several
synthetic instances; classical MABs use a seeded synthetic Bernoulli problem.
Everything is synthetic and seeded, so the gallery runs anywhere without downloads.
Hyperparameters keep their theoretical defaults, except in the search example.
Takes a few minutes.
"""
import base64
import io
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
import plotly.offline

import banditkit as bk
from banditkit import plot
from banditkit.plot import interactive as iplot
from banditkit.plot.style import rc_params


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def div(fig, name: str) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=name, auto_play=False,
                       # No "download as PNG" button: the hosted gallery cannot save files.
                       config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["toImage"]})


def linear_instances(n, K, d, noise_std, seed):
    return bk.sample_instances("linear", n, n_actions=K, n_features=d, noise_std=noise_std, seed=seed)


def compute(n_jobs: int):
    pub, inter = {}, {}
    run = dict(n_jobs=n_jobs, verbose=1)

    # Linear bandits: 10 instances (new arms and theta each) x 2 runs = 20 runs per policy.
    lin = linear_instances(10, K=1000, d=20, noise_std=0.1, seed=0)
    lin_policies = [bk.LinUCB(), bk.LinUCB(n_components=10), bk.LinUCB(n_buckets="sqrt"), bk.SOFUL(sketch_size=10),
                    bk.CBSCFD(sketch_size=10), bk.LinTS()]
    res = bk.Experiment(lin, lin_policies, horizon=3000, n_runs=2, seed=0, **run).run()
    pub["curves"] = png(plot.curves(res, band="ci95")[0])
    pub["panel"] = png(plot.panel(res, metrics=("regret", "reward", "cpu_time"))[0])
    pub["tradeoff"] = png(plot.tradeoff(res, log_x=True)[0])
    pub["dispersion"] = png(plot.dispersion(res, kind="strip")[0])
    pub["profile"] = png(plot.performance_profile(res.by_instance())[0])

    # Parameter search (alpha x lambda_, default grid) next to the theoretical bounds.
    small = linear_instances(5, K=500, d=10, noise_std=0.1, seed=1)
    policies = {"LinUCB": bk.LinUCB(), "LinUCB searched": bk.LinUCB(), "LinTS": bk.LinTS(), "LinTS searched": bk.LinTS()}
    grid = {"alpha": bk.DEFAULT_PARAM_GRID["alpha"], "lambda_": bk.DEFAULT_PARAM_GRID["lambda_"]}
    res_s = bk.Experiment(small, policies, horizon=2000, n_runs=4, seed=1, search_runs=2,
                          param_grid={"LinUCB searched": grid, "LinTS searched": grid}, **run).run()
    dashed = {label: {"linestyle": "--", "filled": False} for label in ("LinUCB searched", "LinTS searched")}
    pub["search"] = png(plot.curves(res_s, band="ci95", overrides=dashed)[0])
    pub["landscape"] = png(plot.search_landscape(res_s, "LinUCB searched")[0])

    # Buckets as the catalog grows: 5 instances x 2 runs per size.
    by_K = {K: bk.Experiment(linear_instances(5, K=K, d=20, noise_std=0.1, seed=K), [bk.LinUCB(), bk.LinUCB(n_buckets="sqrt")],
                             horizon=2000, n_runs=2, seed=2, **run).run() for K in (1_000, 3_000, 10_000, 30_000, 100_000)}
    with plt.rc_context(rc_params()):
        fig_k, axes_k = plt.subplots(1, 2, figsize=(9.5, 3.4), layout="constrained")
    for ax, metric in zip(axes_k, ("cpu_time", "regret")):
        plot.sensitivity(by_K, "n_actions", metric=metric, log_x=True, ax=ax)
        ax.set_xlabel("Number of arms K")
        if metric == "cpu_time":
            ax.set_yscale("log")
        if ax.get_legend():
            ax.get_legend().remove()
    handles, _ = axes_k[0].get_legend_handles_labels()
    fig_k.legend(handles, ["LinUCB", "LinUCB with √K buckets"], loc="outside lower center", ncols=2, frameon=False)
    pub["scaling"] = png(fig_k)

    # Classical MAB: one seeded Bernoulli problem with 10 arms, 20 runs.
    mab = bk.sample_instances("bernoulli", n_instances=1, n_actions=10, min_gap=0.05, seed=3)[0]
    mab_policies = [bk.UCB1(), bk.UCBV(), bk.UCBMinV1(), bk.KLUCB(), bk.BernoulliThompsonSampling(),
                    bk.GaussianThompsonSampling(), bk.Random()]
    res_t = bk.Experiment(mab, mab_policies, horizon=10_000, n_runs=20, seed=3, **run).run()
    pub["real"] = png(plot.curves(res_t, band="ci95")[0])

    # Multi-objective: a front with a concave part (arm 2), as in Drugan and Nowe (2013), fig. 1. 20 runs.
    mo_means = np.array([[0.95, 0.20], [0.85, 0.50], [0.55, 0.55], [0.20, 0.95], [0.45, 0.80],
                         [0.40, 0.30], [0.70, 0.25], [0.30, 0.60]])
    mo = bk.make_env(means=mo_means, noise_std=0.2)
    res_mo = bk.Experiment(mo, [bk.ParetoUCB1(), bk.UCB1(), bk.Random()], horizon=10_000, n_runs=20, seed=0,
                           weights=[0.5, 0.5], track=["pareto_share", "pareto_regret", "unfairness"], **run).run()
    pub["pareto"] = png(plot.pareto_panel(res_mo)[0])
    inter["scalarization"] = div(iplot.scalarization(mo, weights=[0.5, 0.5]), "scalarization")

    # Presentation figures on the same Bernoulli problem. Averaged figures use 20 runs; the others show one decision process.
    ucb_runs = [bk.trace(mab, bk.UCB1(), 10_000, seed=s, every=50) for s in range(20)]
    ts_run = bk.trace(mab, bk.BernoulliThompsonSampling(), 10_000, seed=0, every=50)
    inter["allocation"] = div(iplot.allocation(ucb_runs), "allocation")
    inter["regret_by_arm"] = div(iplot.regret_by_arm(ucb_runs), "regret_by_arm")
    inter["estimates"] = div(iplot.estimates(ucb_runs[0]), "estimates")
    inter["why"] = div(iplot.why_this_arm(ucb_runs[0]), "why")
    inter["beliefs"] = div(iplot.beliefs(ts_run), "beliefs")
    inter["prob_best"] = div(iplot.prob_best(ts_run), "prob_best")
    inter["ucbmin"] = div(iplot.ucbmin_choice(bk.trace(mab, bk.UCBMinV1(), 10_000, every=50)), "ucbmin")

    rng = np.random.default_rng(0)
    ang = np.linspace(0, 2 * np.pi, 25, endpoint=False)
    arms2d = np.c_[np.cos(ang), np.sin(ang)] * rng.uniform(0.6, 1.0, size=(25, 1))
    lin2d = bk.make_env(action_features=arms2d, theta=[0.5, 0.3], noise_std=0.2)
    inter["ellipsoid"] = div(iplot.ellipsoid(bk.trace(lin2d, bk.LinUCB(), 600, every=5)), "ellipsoid")

    # 60 users from 4 groups, 10 runs per policy.
    def unit_rows(n, d):
        X = rng.normal(size=(n, d))
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    groups = unit_rows(4, 10)
    users = bk.make_env(action_features=unit_rows(200, 10), user_thetas=groups[rng.integers(4, size=60)], noise_std=0.1)
    cluster_runs = [bk.trace(users, p, 8000, seed=s, every=100) for p in (bk.SCLUB(), bk.CLUB()) for s in range(10)]
    inter["clusters"] = div(iplot.clusters(cluster_runs, n_true_groups=4), "clusters")

    # Buckets: 10,000 arms in dimension 50, noisy rewards, so exploration lasts and the choices spread.
    r = np.random.default_rng(2)
    X50, th50 = r.normal(size=(10_000, 50)), r.normal(size=50)
    hard = bk.make_env(action_features=X50 / np.linalg.norm(X50, axis=1, keepdims=True),
                       theta=th50 / np.linalg.norm(th50), noise_std=0.5)
    tr_b = bk.trace(hard, bk.LinUCB(n_buckets="sqrt", bucket_rebuild_freq=100), 3000, every=10)
    inter["buckets"] = div(iplot.bucket_choice(tr_b, max_frames=60), "buckets")
    return pub, inter


PUB = [
    ("curves", "Regret cumulé", "Bandits linéaires : 10 instances × 2 runs, valeurs théoriques. Bande = IC 95 %."),
    ("panel", "Planche", "Mêmes runs. Couleur = famille, forme = technique, marqueur vide = variante."),
    ("tradeoff", "Regret contre temps", "Mêmes runs. Ligne = politiques non dominées."),
    ("dispersion", "Dispersion entre runs", "Un point = une run."),
    ("profile", "Profils de performance", "Sur les 10 instances."),
    ("search", "Valeurs théoriques ou recherchées", "5 instances × 4 runs. Pointillés : alpha et lambda cherchés sur d'autres seeds."),
    ("landscape", "Résultat de la recherche", "LinUCB : regret final selon alpha, une courbe par lambda."),
    ("scaling", "Buckets et taille du catalogue", "5 instances × 2 runs par taille."),
    ("real", "MAB classique", "Problème de Bernoulli à 10 bras (seed 3), 20 runs."),
    ("pareto", "Mesures multi-objectifs", "Comme Drugan et Nowé (2013, fig. 3). 20 runs."),
]
INTER = [
    ("allocation", "Allocation", "Bernoulli, UCB1, moyenne sur 20 runs."),
    ("regret_by_arm", "Regret par bras", "Bernoulli, UCB1, moyenne sur 20 runs."),
    ("estimates", "Estimations", "Bernoulli, UCB1, une run."),
    ("why", "Pourquoi ce bras ?", "Une run. Curseur ou Play."),
    ("beliefs", "Croyances de Thompson Sampling", "Bernoulli, une run."),
    ("prob_best", "Probabilité d'être le meilleur", "Bernoulli, une run."),
    ("ucbmin", "Bonus choisi par UCB-min", "Bernoulli, une run."),
    ("ellipsoid", "Ellipsoïde de confiance", "Dimension 2, une run."),
    ("scalarization", "Scalarisation linéaire", "Comme Drugan et Nowé (2013, fig. 1)."),
    ("clusters", "Clusters d'utilisateurs", "60 utilisateurs, 4 vrais groupes, 10 runs par politique."),
    ("buckets", "Buckets", "10 000 bras en dimension 50, une run."),
]


def page(pub, inter) -> str:
    """The HTML page: one column of figure plates, light in both themes like printed figures."""
    # The bundle holds a literal U+FFFD in a regex; the escape keeps the same meaning in an ASCII-safe form.
    plotly_js = plotly.offline.get_plotlyjs().replace("�", "\\uFFFD")
    cards_pub = "\n".join(
        f'<figure class="plate" id="{k}"><figcaption><h3>{t}</h3><p>{d}</p></figcaption>'
        f'<img src="{pub[k]}" alt="{t}" loading="lazy"></figure>' for k, t, d in PUB)
    cards_int = "\n".join(
        f'<figure class="plate" id="fig-{k}"><figcaption><h3>{t}</h3><p>{d}</p></figcaption>'
        f'<div class="plot">{inter[k]}</div></figure>' for k, t, d in INTER)
    nav_pub = " ".join(f'<a href="#{k}">{t}</a>' for k, t, _ in PUB)
    nav_int = " ".join(f'<a href="#fig-{k}">{t}</a>' for k, t, _ in INTER)
    return f"""<title>banditkit figure gallery</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>
:root {{
  --bg: #f4f5f7; --fg: #15171c; --muted: #555b66; --line: #d9dce2; --plate: #ffffff; --accent: #2a78d6;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #121418; --fg: #eceef2; --muted: #a3a9b4; --line: #2b2f37; --plate: #ffffff; --accent: #5598e7; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg: #121418; --fg: #eceef2; --muted: #a3a9b4; --line: #2b2f37; --plate: #ffffff; --accent: #5598e7; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 15px; line-height: 1.55;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 32px 64px; }}
main {{ max-width: 980px; margin: 0 auto; display: grid; gap: 40px; }}
header {{ display: grid; gap: 8px; max-width: 70ch; }}
.eyebrow {{ font-family: var(--font-mono); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }}
h1 {{ font-size: 30px; line-height: 1.2; margin: 0; font-weight: 600; text-wrap: balance; }}
h2 {{ font-size: 21px; margin: 0; font-weight: 600; }}
h3 {{ font-size: 16px; margin: 0; font-weight: 600; color: #15171c; }}
p {{ margin: 0; }}
.lead {{ color: var(--muted); }}
section {{ display: grid; gap: 20px; }}
section > header p {{ color: var(--muted); }}
.plates {{ display: grid; gap: 24px; }}
.plate {{ margin: 0; background: var(--plate); border: 1px solid var(--line); border-radius: 6px; padding: 18px; display: grid; gap: 12px; min-width: 0; }}
.plate figcaption {{ display: grid; gap: 4px; }}
.plate figcaption p {{ color: #555b66; font-size: 14px; }}
.plate img {{ width: 100%; height: auto; display: block; }}
.plot {{ min-width: 0; overflow-x: auto; }}
nav {{ display: flex; flex-wrap: wrap; gap: 8px 16px; font-size: 14px; }}
nav a {{ color: var(--accent); text-decoration: none; }}
nav a:hover, nav a:focus-visible {{ text-decoration: underline; }}
code {{ font-family: var(--font-mono); font-size: 13px; }}
</style>
<script>{plotly_js}</script>
<main>
  <header>
    <span class="eyebrow">banditkit 0.1 · aperçus à valider</span>
    <h1>Galerie des figures</h1>
    <p class="lead">Valeurs théoriques par défaut. 20 runs par comparaison. Données synthétiques à seed fixe.</p>
  </header>
  <section>
    <header><h2>Figures de publication</h2><p><code>banditkit.plot</code></p></header>
    <nav>{nav_pub}</nav>
    <div class="plates">{cards_pub}</div>
  </section>
  <section>
    <header><h2>Figures de présentation</h2><p><code>banditkit.plot.interactive</code></p></header>
    <nav>{nav_int}</nav>
    <div class="plates">{cards_int}</div>
  </section>
</main>
"""


def main() -> None:
    warnings.filterwarnings("ignore")
    output = sys.argv[1] if len(sys.argv) > 1 else "gallery.html"
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    pub, inter = compute(n_jobs)
    html = page(pub, inter)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    print("ok", len(html) / 1e6, "MB")


if __name__ == "__main__":
    main()
