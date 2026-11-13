# banditkit

Bandit algorithms in Python: multi-armed (MAB), multi-objective, linear, efficient linear
(random projections, sketches) and multi-user linear (clustering of users). Experiments
are reproducible (fixed seeds) and their results open in an interactive dashboard.

> [!WARNING]
> **Default hyperparameters are the values of the papers** (`alpha=1`: theoretical confidence bounds).
> They are safe in theory but often explore too much in practice.
> To fit them to your problem, run a grid search (`param_grid=True`) or set them (`bk.LinUCB(alpha=0.1)`).
> Every experiment starts with a numbered warning that lists what to know about its settings.

## Install

```bash
git clone https://github.com/LiliZr/BanditKit.git
cd BanditKit
pip install -e ".[all]"    # numpy, pandas, scipy, matplotlib, plotly ("[dev]" adds pytest)
```

Python 3.10 or later. Only numpy is required; the extras add datasets and figures.

## See the results

Open [`dashboard.html`](dashboard.html) in a browser (no install, no server). It shows the
folders of [`results/`](results):

* pick a folder in the list, or add folders from disk with **Open a results folder…**;
* choose the metric, cumulative or per-step values, the band (95% interval or ± 1 sd),
  log axes, a reference policy (ratios to it), and what to do with outliers (show, crop the axis, hide);
* hide, rename or recolor policies (in the table or the legend), change the palette, zoom and pan;
* export each chart as PNG (chosen size and DPI), SVG, PDF (vector) or CSV.

Charts load plotly.js from the internet; `python -m banditkit dashboard --offline` embeds it.

Each folder of `results/` is the output of the configuration of [`configs/`](configs) with
the same name. The problems are synthetic, drawn from fixed seeds:

| Folder | Problem | Policies |
| --- | --- | --- |
| `mab-bernoulli` | 10 Bernoulli MABs, 10 arms | UCB1, UCB1-Tuned, UCB-V, UCB-min, KL-UCB, Thompson Sampling, Random |
| `linear-synthetic` | 10 linear bandits, 1000 actions in R^20 | LinUCB, ConfidenceBall1, LinUCB with random projection, CBRAP, SOFUL, CBSCFD, LinTS |
| `linear-search` | 5 linear bandits, 200 actions in R^10 | LinUCB and LinTS: defaults against values found by a grid search |
| `multi-user-synthetic` | 100 users in 5 clusters, 20 actions offered per step | LinUCB (shared, per user), CLUB, SCLUB |
| `pareto-two-objectives` | two-objective MAB with a concave Pareto front | Pareto UCB1, UCB1, Random |

## Run experiments

### Command line

```bash
python -m banditkit run configs/mab-bernoulli.json                    # -> results/mab-bernoulli/
python -m banditkit run configs/*.json --generate-dashboard           # every config, then dashboard.html
python -m banditkit run configs/mab-bernoulli.json --name mab-50 --n-runs 50   # a variant, in a new folder
python -m banditkit dashboard results/mab-bernoulli results/mab-50 --out mab.html
```

`run` also takes `--n-jobs`, `--horizon`, `--results-dir` and `--dashboard PATH`. A results
folder holds `config.json` (what was run), `results.npz` (every run) and `summary.json`
(what the dashboard reads). A configuration names the environment, the policies and the
settings of the experiment:

```json
{
  "description": "One or two sentences shown in the dashboard.",
  "env": {"kind": "linear", "n_instances": 10, "n_actions": 1000, "n_features": 20, "seed": 0},
  "policies": [{"policy": "LinUCB"}, {"policy": "LinUCB", "alpha": 0.1, "label": "LinUCB alpha=0.1"}],
  "experiment": {"horizon": 3000, "n_runs": 2, "seed": 0, "track": ["pseudo_regret"]}
}
```

`env.kind`: `bernoulli`, `gaussian`, `linear`, `multi_user` (synthetic instances), `make_env`
(arrays given in the file), `npz` (a file of your own) or `dataset` (a function of
`banditkit.datasets`).

### Python

```python
import numpy as np
import banditkit as bk

env = bk.make_env(means=[0.2, 0.5, 0.7], noise_std=0.1)               # MAB
res = bk.Experiment(env, [bk.UCB1(), bk.KLUCB(), bk.ThompsonSampling()], horizon=5000, n_runs=20, seed=0).run()

X = np.random.default_rng(0).normal(size=(1000, 50))
X /= np.linalg.norm(X, axis=1, keepdims=True)
env = bk.make_env(action_features=X, theta=np.ones(50) / 50 ** 0.5)  # linear bandit
res = bk.Experiment(env, [bk.LinUCB(), bk.CBRAP(n_components=10), bk.SOFUL(sketch_size=10)],
                    horizon=2000, n_runs=10, param_grid=True, seed=0).run()   # grid search of alpha, lambda_

print(res.summary())                                   # final means and standard deviations
from banditkit.report import save_run
save_run(res, "results/my-run")                        # a folder the dashboard reads
```

`res.to_latex()`, `res.compare(a, b)` and `res.save()` / `bk.Results.load()` work on any
results. `banditkit.plot` draws the dashboard charts with matplotlib for papers
(`curves`, `dispersion`, `tradeoff`, `search_landscape`, `pareto_panel`, `performance_profile`).

A policy also works alone on data, with the names of scikit-learn:

```python
model = bk.LinUCB(alpha=0.1).fit(X_played, rewards)    # X_played: (n, d), rewards: (n,)
action = model.predict(A)                              # row of the candidates A: (k, d)
model.partial_fit(A[[action]], [reward])
```

## Important parameters

**Experiment** (`bk.Experiment(...)`, or the `experiment` part of a configuration):

| Parameter | Meaning |
| --- | --- |
| `horizon` | steps T per run |
| `n_runs` | runs per instance; a list of environments gives several instances, pooled in the results |
| `seed` | master seed: the same seed gives the same results |
| `param_grid` | `None` (default): hyperparameters as given. `True`: grid search per policy. A dict: your grid |
| `search_runs`, `search_metric` | runs per combination during the search (on other seeds), and its criterion |
| `track` | extra measures: `pseudo_regret`, `pareto_regret`, `pareto_share`, `unfairness`, `clusters`, `estimates`, `actions`, `env_metrics` |
| `measure_memory` | peak memory (slows Python code about fivefold) |
| `log_interval` | curves are stored every `log_interval` steps (default: about 2000 points) |
| `n_jobs` | parallel processes (`-1`: all CPUs) |
| `weights` | scalarization of a multi-objective reward (default: uniform) |
| `timeout`, `train_freq`, `checkpoint_dir`, `verbose` | CPU-time limit per run, batched updates, saved runs, messages |

Stored by default: cumulative regret, cumulative reward, CPU time and wall time, at every
logged step. The rest needs `track` or `measure_memory`, and the warning at launch lists
what is not stored.

**Policies** (`help(bk.LinUCB)` lists every parameter with its range and default):

| Parameter | Meaning |
| --- | --- |
| `alpha` | scale of the exploration bonus; 1 is the bound of the paper |
| `lambda_` | ridge regularization of linear policies (0.1; no theoretical value) |
| `delta` | confidence level of the bounds |
| `n_components`, `projection` | random projection to k dimensions (`"gaussian"` or `"fjlt"`) |
| `sketch_size` | size of the Frequent Directions sketch (SOFUL, CBSCFD) |
| `user_model` | multi-user LinUCB: `"shared"`, `"individual"` or `"kmeans"` |
| `subgaussian`, `theta_bound`, `reward_range` | noise level R, norm S of theta, reward range b; read from the environment when it knows them |

Some theoretical values explore much more than usual tuned values: LinTS (sampling scale
R sqrt(9 d ln(t/delta))), Gaussian Thompson Sampling (variance b² / (n + 1)) and SCLUB
(split threshold 4 R sqrt(d / lambda_x)). CLUB has no theoretical value for `alpha` and
`cluster_alpha`; its paper tunes them on the data.

## Datasets

* **Synthetic**: `bk.sample_instances("bernoulli" | "gaussian" | "linear" | "multi_user", n_instances, ...)`.
* **MovieLens** ([GroupLens](https://grouplens.org/datasets/movielens/)): downloaded on first use
  into `~/banditkit_data` (or `$BANDITKIT_DATA`). Variants `small`, `1m`, `20m`, `25m`, `32m`.
* **Steam** ([Kaggle](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)):
  needs a Kaggle account, so download it once by hand:

  ```bash
  kaggle datasets download -d antonkozyriev/game-recommendations-on-steam -p ~/banditkit_data/steam --unzip
  ```

The dataset functions are examples of preprocessing taken from published experiments:

| Function | Environment | Preprocessing |
| --- | --- | --- |
| `datasets.movielens_linear(variant)` | multi-user linear bandit | 1000 most active users and 1000 most rated items, SVD of rank d - 1, vectors lifted so that rewards lie in [0, 1], 20 random items per step (Li et al., 2019) |
| `datasets.steam_linear(path)` | multi-user linear bandit | same steps on binary feedback: recommended = 1 (Wang et al., 2023) |
| `datasets.ratings_to_linear(df)` | multi-user linear bandit | the same steps on any table `user_id, item_id, rating` |
| `datasets.movielens_mab(variant, arms="genres" or "movies")` | MAB | a step returns one real rating / 5 of a movie of the arm (Gupta et al., 2021) |
| `datasets.rewards_to_mab(groups)` | MAB | the same on any observed rewards |

Your own data: environments take plain arrays.

| Problem | Arrays | Environment |
| --- | --- | --- |
| MAB, multi-objective MAB | `means` (K,) or (K, D) | `bk.make_env(means=..., noise_std=0.1)` |
| MAB from observed rewards | `samples`: K arrays | `bk.make_env(samples=...)` |
| Linear | `action_features` (K, d), `theta` (d,) | `bk.make_env(action_features=..., theta=...)` |
| Multi-user linear | `action_features` (K, d), `user_thetas` (U, d), optional `user_stream` (T,) | `bk.make_env(action_features=..., user_thetas=...)` |
| Group recommendation | `relevance` (members, K) in [0, 1] | `bk.GroupRecommendationMAB(relevance)` |

The same arrays saved in an `.npz` file load with `datasets.load_npz(path)` (keys in
`banditkit/datasets/base.py`).

## Policies

| Family | Policies |
| --- | --- |
| Any (resolved from the problem) | `UCB`, `ThompsonSampling`, `EpsilonGreedy`, `Random` |
| MAB, multi-objective MAB (scalarized) | `UCB1`, `UCB1Tuned`, `UCB1Normal`, `MABEpsilonGreedy` [Auer et al.]; `UCBV` [Audibert et al.]; `UCBV2` [Fan et al.]; `EUCBV` [Mukherjee et al.]; `UCBMinV1`, `UCBMinV2` [Izri et al., 2026]; `KLUCB` [Garivier and Cappé]; `BernoulliThompsonSampling` [Thompson; Agrawal and Goyal, 2012]; `GaussianThompsonSampling` [Agrawal and Goyal, 2013b]; `Softmax` [Sutton and Barto] |
| Multi-objective MAB (Pareto) | `ParetoUCB1` [Drugan and Nowé] |
| Linear | `LinUCB` [Li et al., 2010; radius of Abbasi-Yadkori et al.]; `ConfidenceBall1` [Dani et al.]; `LinTS` [Agrawal and Goyal, 2013a]; `CBRAP` [Yu et al.]; `SOFUL` [Kuzborskij et al.]; `CBSCFD` [Chen et al.]; `LinGreedy`; random projections of LinUCB and ConfidenceBall1 [Izri et al., 2025; FJLT: Ailon and Chazelle] |
| Multi-user linear | `LinUCB(user_model=...)`, `CLUB` [Gentile et al.], `SCLUB` [Li et al., 2019] |

## Tests

```bash
pip install -e ".[dev]" && pytest
```

## References

* Abbasi-Yadkori, Pál, Szepesvári. Improved algorithms for linear stochastic bandits. NeurIPS 2011.
* Agrawal, Goyal. Analysis of Thompson Sampling for the multi-armed bandit problem. COLT 2012.
* Agrawal, Goyal. Thompson Sampling for contextual bandits with linear payoffs. ICML 2013 (2013a); scale of the final version, arXiv:1209.3352.
* Agrawal, Goyal. Further optimal regret bounds for Thompson Sampling. AISTATS 2013 (2013b).
* Ailon, Chazelle. The fast Johnson-Lindenstrauss transform and approximate nearest neighbors. SIAM Journal on Computing, 2009.
* Amer-Yahia, Roy, Chawla, Das, Yu. Group recommendation: semantics and efficiency. PVLDB 2(1), 2009.
* Audibert, Munos, Szepesvári. Exploration-exploitation tradeoff using variance estimates in multi-armed bandits. Theoretical Computer Science, 2009.
* Auer, Cesa-Bianchi, Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 2002.
* Chen, Luo, Zhang, Yu, Lian. Efficient and robust high-dimensional linear contextual bandits. IJCAI 2020.
* Dani, Hayes, Kakade. Stochastic linear optimization under bandit feedback. COLT 2008.
* Dolan, Moré. Benchmarking optimization software with performance profiles. Mathematical Programming, 2002.
* Drugan, Nowé. Designing multi-objective multi-armed bandits algorithms: a study. IJCNN 2013.
* Fan, Han, Lv, Xu, Zhou. Precise asymptotics and refined regret of variance-aware UCB. NeurIPS 2025.
* Garivier, Cappé. The KL-UCB algorithm for bounded stochastic bandits and beyond. COLT 2011.
* Gentile, Li, Zappella. Online clustering of bandits. ICML 2014.
* Gupta, Chaudhari, Joshi, Yağan. Multi-armed bandits with correlated arms. IEEE Transactions on Information Theory, 2021.
* Harper, Konstan. The MovieLens datasets: history and context. ACM Transactions on Interactive Intelligent Systems, 2015.
* Izri, Groz, Maniu. Implementing efficient linear bandits via sketches and random projections. IEEE BigData 2025, pp. 956-965.
* Izri, Groz, Maniu. UCB-min: combining bandits under variance heterogeneity. IEEE BigData 2026.
* Kozyriev. Game Recommendations on Steam. Kaggle dataset.
* Kuzborskij, Cella, Cesa-Bianchi. Efficient linear bandits through matrix sketching. AISTATS 2019.
* Li, Chen, Li, Leung. Improved algorithm on online clustering of bandits. IJCAI 2019.
* Li, Chu, Langford, Schapire. A contextual-bandit approach to personalized news article recommendation. WWW 2010.
* Mukherjee, Naveen, Sudarsanam, Ravindran. Efficient-UCBV: an almost optimal algorithm using variance estimates. AAAI 2018.
* Sutton, Barto. Reinforcement learning: an introduction. MIT Press, 1998.
* Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 1933.
* Wang, Xie, Liu, Li, Lui. Online clustering of bandits with misspecified user models. NeurIPS 2023.
* Yu, Lyu, King. CBRAP: contextual bandits with random projection. AAAI 2017.
