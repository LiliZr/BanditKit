# banditkit

Multi-armed, multi-objective and linear bandits behind one API, with reproducible
experiments, publication figures and explanatory figures.

**See results right after cloning:** open [`dashboard.html`](dashboard.html) in a browser.
It shows the experiments stored in [`results/`](results), with no install and no server.
See [Results and dashboard](#results-and-dashboard) to run new ones.

* **Any numerical data**: environments take plain NumPy arrays (mean rewards, action
  features, user parameters...). Bring your own data and your own preprocessing; see
  [Your own data](#your-own-data).
* **One API for four families**: classical MAB, multi-objective MAB, linear bandits,
  and efficient linear bandits (random projections, sketches, arm buckets, clustering of users).
* **The family is detected from the data**: `make_env(means=...)` is a MAB,
  `make_env(action_features=..., theta=...)` a linear bandit. A parameter that does
  not apply to the detected family raises an error that says where it applies.
* **Strict validation**: shapes, signs and ranges of every input are checked when
  objects are built, with messages that name the parameter.
* **Names of the big RL / ML libraries**: Gymnasium's `reset` / `step`, Stable-Baselines3's
  `predict`, `train_freq`, `log_interval`, scikit-learn's `fit`, `partial_fit`, `get_params`,
  `n_jobs`, `n_components`. A policy works alone on data, as a scikit-learn estimator; see
  [One policy on data](#one-policy-on-data-as-a-scikit-learn-estimator).

## Install

```bash
pip install -e ".[all]"     # numpy, pandas, scipy, matplotlib, plotly
pip install -e ".[dev]"     # + pytest
```

Python 3.10 or later. Only numpy is required; the extras add analysis, figures and datasets.

## Results and dashboard

Every experiment of the repository is a folder of [`results/`](results), produced by a
JSON configuration of [`configs/`](configs) with fixed seeds:

| Folder | Problem |
| --- | --- |
| `mab-bernoulli` | 10 Bernoulli MABs with 10 arms; UCB family, KL-UCB, Thompson Sampling |
| `linear-synthetic` | 10 linear bandits (1000 actions in R^20); LinUCB and its efficient variants, SOFUL, CBSCFD, LinTS |
| `linear-search` | theoretical hyperparameters against the best ones found by a search |
| `multi-user-synthetic` | 100 users in 5 clusters, 20 actions offered per step; LinUCB, CLUB, SCLUB |
| `pareto-two-objectives` | two-objective MAB with a concave Pareto front; Pareto UCB1 |
| `steam-bigdata2025` | real data: Steam instances of Izri et al. (BigData 2025), 1807 features |

A folder holds `config.json` (what was run), `results.npz` (every run, `bk.Results.load`)
and `summary.json` (what the dashboard draws). Rerun, add or compare experiments:

```bash
python -m banditkit run configs/linear-synthetic.json --generate-dashboard   # run, then rebuild dashboard.html
python -m banditkit run results/mab-bernoulli/config.json --name mab-50-runs --n-runs 5
python -m banditkit dashboard                                  # dashboard.html from every folder of results/
python -m banditkit dashboard results/mab-bernoulli results/mab-50-runs --out mab.html   # chosen folders only
```

`pip install -e .` also installs the `banditkit` command (`banditkit run ...`). In the
dashboard, the list on the left switches between folders. **Open a results folder…**
adds folders from disk (one folder, or a whole `results/` folder) without rebuilding
anything. Charts load plotly.js from a CDN; `banditkit dashboard --offline` embeds it.

A configuration names the environment, the policies and the settings of `bk.Experiment`:

```json
{
  "description": "One or two sentences shown in the dashboard.",
  "env": {"kind": "linear", "n_instances": 10, "n_actions": 1000, "n_features": 20, "seed": 0},
  "policies": [{"policy": "LinUCB"}, {"policy": "LinUCB", "n_buckets": "sqrt", "label": "LinUCB buckets"}],
  "experiment": {"horizon": 3000, "n_runs": 2, "seed": 0, "log_interval": 15}
}
```

`env.kind` is `bernoulli`, `gaussian`, `linear` or `multi_user` (synthetic, `bk.sample_instances`),
`make_env` (keywords of `bk.make_env`), `npz` (a file of your own, see
[Your own data](#your-own-data)) or `dataset` (a function of `banditkit.datasets`).
In Python, `banditkit.report.save_run(results, "results/my-run")` writes the same folder.

## Quick start

```python
import numpy as np
import banditkit as bk

# Classical MAB: the generic UCB becomes UCB1
env = bk.make_env(means=[0.2, 0.5, 0.7], noise_std=0.1)
results = bk.Experiment(env, [bk.UCB(), bk.UCBV(), bk.UCBMinV1(), bk.ThompsonSampling()],
                        horizon=5000, n_runs=20, seed=0).run()
results.plot("regret", band="ci95")[0].savefig("regret.pdf")
print(results.to_latex())

# Multi-objective MAB: reward vectors of dimension 2, scalarized by `weights`
env = bk.make_env(means=[[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]], noise_std=0.1)
res = bk.Experiment(env, [bk.UCB1(), bk.ParetoUCB1()], horizon=5000, n_runs=10, weights=[0.5, 0.5],
                    track=["pareto_share", "pareto_regret", "unfairness"]).run()
from banditkit import plot
plot.pareto_panel(res)  # the three measures of Drugan and Nowe (2013)

# Linear bandit: the same generic UCB becomes LinUCB; techniques stack as parameters
X = np.random.default_rng(0).normal(size=(1000, 50))
env = bk.make_env(action_features=X / np.linalg.norm(X, axis=1, keepdims=True), theta=np.ones(50) / 50 ** 0.5)
policies = [bk.UCB(), bk.LinUCB(n_components=10), bk.LinUCB(n_components=10, n_buckets="sqrt"),
            bk.SOFUL(sketch_size=10), bk.CBSCFD(sketch_size=10), bk.LinTS()]
bk.Experiment(env, policies, horizon=10_000, n_runs=10, n_jobs=-1).run()  # n_jobs > 1: from a .py file or a notebook
```

Online use, one step at a time (Gymnasium loop):

```python
obs, info = env.reset(seed=0)
policy = bk.LinUCB()
policy.reset(env.spec, seed=0)
for t in range(1000):
    action = policy.predict(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    policy.update(obs, action, reward)
    obs = next_obs
```

### One policy on data, as a scikit-learn estimator

`fit`, `partial_fit` and `predict` work on plain arrays, with no environment.
`X` holds the played actions and `y` their rewards:

| Problem | `X` in `fit` / `partial_fit` | `predict` returns |
| --- | --- | --- |
| MAB | arm indices, shape (n_samples,) | `predict()`: an arm |
| Linear | features of the played actions, shape (n_samples, n_features), or their row indices in `action_features` | `predict(A)`: a row of the candidate actions A (default: `action_features`) |
| Multi-user linear | same, with `users=` | `predict(A, user=u)` |

```python
model = bk.LinUCB(alpha=1.0, lambda_=1.0)
model.fit(X, y)                        # X: (n_samples, n_features), y: (n_samples,)
action = model.predict(A)              # A: (n_candidates, n_features) -> row index
model.partial_fit(A[[action]], [reward])   # learn from one more sample

arm = bk.UCB1().fit(arms, rewards).predict()        # MAB: arms are indices
bk.CLUB().fit(X, y, users=user_ids).predict(A, user=3)
```

`fit` forgets what was learned, `partial_fit` adds to it; both return the policy.
`X` can be empty for a cold start (`n_actions=` for a MAB). `fit` also takes
`action_features` (candidate actions; arm buckets need them), `n_actions`, `n_users`,
`horizon`, `weights` and `seed`. Without an environment, the noise level R, the norm S
and the reward range b are unknown: `fit` warns (`bk.ParameterWarning`) about assumed
and arbitrary values; give them to the constructor (`subgaussian`, `theta_bound`,
`reward_range`). `get_params`, `set_params` and `sklearn.base.clone` work as for any
scikit-learn estimator.

## Policies

| Family | Policies |
| --- | --- |
| Any (generic, resolved from the problem) | `UCB`, `ThompsonSampling`, `EpsilonGreedy`, `Random` |
| MAB, multi-objective MAB (scalarized) | `UCB1`, `UCB1Tuned`, `UCB1Normal`, `UCBV`, `UCBV2`, `UCBMinV1`, `UCBMinV2`, `EUCBV`, `KLUCB`, `Softmax`, `MABEpsilonGreedy`, `BernoulliThompsonSampling`, `GaussianThompsonSampling` |
| Multi-objective MAB (Pareto) | `ParetoUCB1` |
| Linear | `LinUCB`, `ConfidenceBall1`, `LinTS`, `LinGreedy`, `CBRAP`, `SOFUL`, `CBSCFD` |
| Multi-user linear | `LinUCB(user_model="shared" / "individual" / "kmeans")`, `CLUB`, `SCLUB` |

Efficiency techniques of the linear policies are parameters: `n_components` (random
projection, `projection="gaussian"` or `"fjlt"`), `n_buckets` (arm buckets), `user_model`
(clustering of users). Labels follow them: `LinUCB_RP_AB`, `LinUCB_IND_RP_AB_KMeans`.

`help(bk.LinUCB)` lists every parameter with its type, range, default and meaning.

## Hyperparameters: theoretical values or a search

Two ways to set them, announced when an experiment starts (`verbose=1`, the default).

**Theoretical values (default).** With `alpha=1`, every policy uses the confidence
bound of its paper. Constants of these bounds come from the environment when it knows
them: noise level R, norm S of theta, reward range b. Values with no theoretical
justification are reported as arbitrary: `lambda_=0.1` (any lambda > 0 keeps the bounds
valid), `epsilon`, `tau`. Values the theory needs but the environment does not give are
reported as assumed (R = 1, b = 1). `policy.parameter_notes()` lists them for one policy.

A few theoretical values are worth knowing, because they explore much more than the
values usually tuned in experiments:

| Policy | Default (value of the paper) |
| --- | --- |
| `LinTS` | sampling scale v_t = R sqrt(9 d ln(t / delta)) (Agrawal and Goyal, final arXiv version) |
| `GaussianThompsonSampling` | variance b² / (n + 1), b the reward range (Agrawal and Goyal, AISTATS 2013) |
| `SCLUB` | split threshold 4 R sqrt(d / lambda_x), lambda_x the smallest eigenvalue of E[x xᵀ] (Theorem 1) |
| `CLUB` | the paper tunes alpha and alpha_2 on the data: defaults 1 and 2 are reported as arbitrary |
| `ConfidenceBall1` | L1 ball of Dani et al. with the OFUL radius of Abbasi-Yadkori et al. (as in the thesis code) |

**Search of the best values (slower).** `param_grid` tries every combination on the
problem and keeps the best one per policy (lowest final regret by default):

```python
bk.Experiment(env, policies, horizon=5000, n_runs=20, param_grid=True)  # alpha 1e-4..100, lambda_ 1e-5..1
bk.Experiment(env, policies, horizon=5000, n_runs=20,
              param_grid={"alpha": np.logspace(-4, 2, 7), "lambda_": [0.01]})  # lambda_ fixed for everyone
bk.Experiment(env, {"LinUCB": bk.LinUCB(), "LinTS": bk.LinTS()}, horizon=5000, n_runs=20,
              param_grid={"LinUCB": {"alpha": [0.1, 1.0]}})                   # one grid per policy
```

Each combination runs `search_runs` times (default 3) per instance, on other seeds
than the reported runs; the best one is then run `n_runs` times. `results.best_params`
and `results.search_table()` give the outcome, `plot.search_landscape` draws it, and the
legend of `plot.curves` becomes a table of the best values. A searched value fits this
problem only: report it as such.

## Experiments and results

`Experiment(env, policies, horizon, n_runs, weights, seed, n_jobs, log_interval,
timeout, train_freq, track, measure_memory, param_grid, search_runs, search_metric,
checkpoint_dir, verbose, callback)`. Use 10 to 20 runs or more to compare policies.
Within a run all policies see the same environment randomness. The regret is
`max_a w.mu_a - w.r_t` summed over steps (observed rewards); `track=["pseudo_regret"]`
adds the noise-free version. Multi-objective problems can also track `pareto_regret`,
`pareto_share` (share of pulls on Pareto optimal arms) and `unfairness` (spread of the
pulls among them); `bk.TRACKABLE` lists every option.

`Results` holds curves of shape (n_runs, n_points): `summary()`, `to_latex()`,
`to_dataframe()`, `save()`, `load()`, `merge()` (runs from several machines).

### One problem or several instances

A synthetic experiment can run on one problem, or on several instances of the same
kind of problem:

* **one instance**: `Experiment(env, ...)`. The regret measures the policies on this
  problem only (one theta, one action set).
* **several instances**: `Experiment([env_1, ..., env_m], ...)`. Each instance gets
  `n_runs` runs and results pool them, so averages are over problems too (Bayesian
  regret). Use it to say which policy is better on a kind of problem.

```python
instances = bk.sample_instances("linear", 10, n_actions=1000, n_features=20, noise_std=0.1, seed=0)
res = bk.Experiment(instances, [bk.LinUCB(), bk.CBRAP(n_components=10)], horizon=5000, n_runs=5).run()
res.compare("LinUCB", "CBRAP")         # paired over instances: mean difference, 95% CI, share of wins
plot.performance_profile(res.by_instance())
```

Pooling makes sense only if the instances share one distribution: same number of
actions, dimension, noise level and norm of theta, so that regrets have the same scale.
`sample_instances` draws them this way (`"bernoulli"`, `"gaussian"`, `"linear"`), and
`min_gap` redraws instances whose two best actions are almost tied. Compare policies
instance by instance (`compare`, `performance_profile`) as well as on the pooled mean,
since a few hard instances can dominate a mean.

## Figures

Publication figures (`banditkit.plot`, matplotlib), on the runs of an experiment:
`curves`, `panel`, `pareto_panel`, `tradeoff`, `sensitivity`, `search_landscape`,
`dispersion`, `performance_profile`. Color = algorithm family,
marker = combination of techniques, hollow marker = variant. Band: `"std"`,
`"ci95"`, `"quantiles"` or `None`. `style="science"` uses SciencePlots.

Explanatory figures (`banditkit.plot.interactive`, plotly). Most show how a policy
decides, on one detailed run `tr = bk.trace(env, policy, horizon)`: `estimates`,
`why_this_arm`, `beliefs`, `prob_best`, `ellipsoid`, `ucbmin_choice`, `bucket_choice`.
`allocation`, `regret_by_arm` and `clusters` also accept a list of traces (several
runs) and average them. `gain_vs_uniform(results, env)` compares policies over the
runs of an experiment. `scalarization(env)` shows which Pareto
optimal arms a linear scalarization can choose (two objectives).
`examples/figure_gallery.py` builds a page with all of them.

## Your own data

banditkit is meant to test bandits on any numerical data. Environments take plain
arrays; where they come from does not matter. Any dataset and any preprocessing work,
as long as the output has these types:

| Problem | Arrays you provide | Environment |
| --- | --- | --- |
| MAB, multi-objective MAB | `means` (K,) or (K, D) | `make_env(means=means, noise_std=0.1)` |
| MAB from observed rewards | `samples`: K arrays, each (n_k,) or (n_k, D) | `make_env(samples=samples)` |
| Linear bandit | `action_features` (K, d), `theta` (d,) or (D, d) | `make_env(action_features=X, theta=theta)` |
| Multi-user linear bandit | `action_features` (K, d), `user_thetas` (U, d) or (U, D, d), optional `user_stream` (T,) of user indices | `make_env(action_features=X, user_thetas=thetas, user_stream=stream)` |
| Group recommendation | `relevance` (n_members, K) in [0, 1], for a group you formed | `GroupRecommendationMAB(relevance)` |

The same arrays can be saved in an `.npz` file and read with `datasets.load_npz(path)`
(keys in `banditkit/datasets/base.py`). `.npz` files of the Scaling-Linear-Bandits
repository load as is. For a setting outside this table, subclass `bk.Environment`.

## Datasets: example preprocessing

> **The dataset functions are EXAMPLES of preprocessing.** They turn MovieLens and Steam
> into bandit environments by following published experiments (references below).
> Nothing else in banditkit depends on them. Adapt them, or replace them with your own
> pipeline, as long as the output keeps the types of [Your own data](#your-own-data).

```python
from banditkit import datasets

data = datasets.movielens_linear("20m")                 # downloads MovieLens once
env = data.to_env()                                     # multi-user linear bandit, 20 items per step
steam = datasets.steam_linear("path/to/kaggle/steam")   # Kaggle files, downloaded by hand
```

**Multi-user linear bandits** (`datasets.ratings_to_linear`, usable on any ratings table
with columns `user_id`, `item_id`, `rating`). Based on the real-data experiments of
SCLUB (Li et al., IJCAI 2019), also used by later clustering-of-bandits papers:

1. keep the 1000 most rated items and the 1000 users who rate most;
2. factorize the users x items matrix: the ratings (Li et al., 2019), or with
   `feedback="binary"` 1 if rating > 3 else 0 (Wang et al., NeurIPS 2023);
3. truncated SVD of rank d - 1 (d = 20): user vectors and item vectors;
4. every vector v becomes (v / (√2 ‖v‖), 1 / √2), so that every mean reward lies in [0, 1];
5. at each step, 20 random items are offered (`MultiUserLinearEnv(n_candidates=20)`).

The papers do not state how the singular values are shared between users and items
(here: square root on each side), the noise level (here 0.1) or the order of user
arrivals (here uniform, or `arrivals="data"`). `steam_linear` applies the binary variant
to Steam (a recommended game is 1); these papers use MovieLens and Yelp, not Steam.

### High-dimensional single-user instances (Izri, Groz and Maniu, BigData 2025)

`datasets.steam_single_user(seed)`, `datasets.movielens_single_user(seed)` and
`datasets.bigdata2025_instances("steam", n_instances)` build the instances of the paper
from its processed data, downloaded once from
[Stochastic-linear-bandits](https://github.com/LiliZr/Stochastic-linear-bandits). One user,
drawn with the seed, per instance:

* Steam: 2000 games with 1807 features (metadata, rating class, one-hot tags). The user's
  parameter is the least-squares fit of "recommended" on the games they reviewed; mean
  rewards are min-max normalized over the catalog; noise 0.1.
* MovieLens: the movies the user rated, with their year, genres, rating statistics and
  the users who rated them (one-hot); the mean reward is the rating / 5.

`paper_code=True` rebuilds the Steam instances exactly as the repository code does. That
code keeps the CSV index column as a feature (about 90% of the squared norm of a game
vector) and pairs the reviewed games, in catalog order, with labels in review order. The
default drops the index column and aligns each game with its own label.

### Real rewards for classical MABs

Thompson Sampling and KL-UCB come with guarantees for Bernoulli or Gaussian rewards.
Real ratings are discrete and skewed. `movielens_mab` follows the MovieLens experiments
of Gupta et al. (IEEE Transactions on Information Theory, 2021): a step returns one
real rating of the chosen arm, divided by 5 (`EmpiricalMAB`).

| Call | Arms | Reward |
| --- | --- | --- |
| `movielens_mab("small", arms="genres")` | the genres; a movie with several genres gets one at random | a rating of a movie of the genre |
| `movielens_mab("small", arms="movies")` | the 50 most rated movies | a rating of the movie |

The paper samples ratings from half of the users; here every rating is used.
`datasets.rewards_to_mab({"action": observed_rewards, ...})` builds the same environment
from any data.

Group recommendation has no dataset function: groups depend on how they are formed
(an API, a choice of the experimenter...). Build the relevance matrix of each group
with your own pipeline and pass it to `GroupRecommendationMAB`.

## References

* Auer, Cesa-Bianchi, Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 2002 (UCB1, UCB1-Tuned, UCB1-Normal, epsilon-greedy).
* Audibert, Munos, Szepesvári. Exploration-exploitation tradeoff using variance estimates in multi-armed bandits. Theoretical Computer Science, 2009 (UCB-V).
* Fan, Han, Lv, Xu, Zhou. Precise asymptotics and refined regret of variance-aware UCB. NeurIPS 2025 (UCB-V2 here).
* Agrawal, Goyal. Further optimal regret bounds for Thompson Sampling. AISTATS 2013 (Gaussian Thompson Sampling).
* Mukherjee, Naveen, Sudarsanam, Ravindran. Efficient-UCBV: an almost optimal algorithm using variance estimates. AAAI 2018.
* Garivier, Cappé. The KL-UCB algorithm for bounded stochastic bandits and beyond. COLT 2011.
* Agrawal, Goyal. Thompson Sampling for contextual bandits with linear payoffs. ICML 2013; scale v_t of the final version, arXiv:1209.3352 (LinTS).
* Drugan, Nowé. Designing multi-objective multi-armed bandits algorithms: a study. IJCNN 2013 (Pareto UCB1).
* Abbasi-Yadkori, Pál, Szepesvári. Improved algorithms for linear stochastic bandits. NeurIPS 2011 (OFUL radius).
* Dani, Hayes, Kakade. Stochastic linear optimization under bandit feedback. COLT 2008 (ConfidenceBall1).
* Yu, Lyu, King. CBRAP: contextual bandits with random projection. AAAI 2017.
* Kuzborskij, Cella, Cesa-Bianchi. Efficient linear bandits through matrix sketching. AISTATS 2019 (SOFUL).
* Chen, Luo, Zhang, Yu, Lian. Efficient and robust high-dimensional linear contextual bandits. IJCAI 2020 (CBSCFD).
* Gentile, Li, Zappella. Online clustering of bandits. ICML 2014 (CLUB).
* Li, Chen, Li, Leung. Improved algorithm on online clustering of bandits. IJCAI 2019 (SCLUB; MovieLens preprocessing).
* Wang, Xie, Liu, Li, Lui. Online clustering of bandits with misspecified user models. NeurIPS 2023 (binary MovieLens preprocessing).
* Gupta, Chaudhari, Joshi, Yağan. Multi-armed bandits with correlated arms. IEEE Transactions on Information Theory, 2021 (MovieLens MABs).
* Izri, Groz, Maniu. Implementing efficient linear bandits via sketches and random projections. IEEE BigData 2025, pp. 956-965 (single-user Steam and MovieLens instances).
* Dolan, Moré. Benchmarking optimization software with performance profiles. Mathematical Programming, 2002.

## Tests

```bash
pytest
```
