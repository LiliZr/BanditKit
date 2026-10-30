# banditkit

Multi-armed, multi-objective and linear bandits behind one API, with reproducible
experiments, publication figures and explanatory figures.

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
  `predict`, `train_freq`, `log_interval`, scikit-learn's `get_params`, `n_jobs`, `n_components`.

## Install

```bash
pip install -e ".[all]"     # numpy, pandas, scipy, matplotlib, plotly
pip install -e ".[dev]"     # + pytest
```

Python 3.10 or later. Only numpy is required; the extras add analysis, figures and datasets.

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

> **The dataset functions are EXAMPLES of preprocessing.** They show one simple,
> documented way to turn MovieLens and Steam into bandit environments. They are not
> the reference preprocessing of these datasets, and nothing else in banditkit depends
> on them. Adapt them, or replace them with your own pipeline, as long as the output
> keeps the types of [Your own data](#your-own-data).

```python
from banditkit import datasets

data = datasets.movielens_linear("1m", n_features=100)   # downloads MovieLens once
env = data.to_env()                                     # multi-user linear bandit
steam = datasets.steam_linear("path/to/kaggle/steam")   # Kaggle files, downloaded by hand
```

Default choices of the example pipeline (`datasets.ratings_to_linear`, usable on any
ratings table with columns `user_id`, `item_id`, `rating`):

1. keep users and items with at least 20 ratings;
2. item features: side features, rating statistics, then an SVD of the rating matrix
   up to `n_features`, standardized, with a bias column, rows of norm 1;
3. one ridge regression per user gives `user_thetas`;
4. the user stream follows the order of the ratings.

### Real rewards for classical MABs

Thompson Sampling and KL-UCB come with guarantees for Bernoulli or Gaussian rewards.
Real rewards are skewed, heavy-tailed or discrete. These example environments resample
rewards actually observed for each action (`EmpiricalMAB`, rewards mapped to [0, 1]):

| Function | Actions | Reward | Data |
| --- | --- | --- | --- |
| `taxis_mab()` | NYC pickup zones | fare (or total, tip rate) of a real trip | 6,433 trips, downloaded once from GitHub |
| `movielens_mab("1m")` | genres, or the most rated movies | a real rating / 5 | MovieLens, downloaded once |
| `steam_mab(path)` | the most reviewed games | hours played (capped at the 99th percentile) | Kaggle files |

`datasets.rewards_to_mab({"action": observed_rewards, ...})` builds the same environment
from any data.

Group recommendation has no dataset function: groups depend on how they are formed
(an API, a choice of the experimenter...). Build the relevance matrix of each group
with your own pipeline and pass it to `GroupRecommendationMAB`.

## References

* Auer, Cesa-Bianchi, Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 2002 (UCB1, UCB1-Tuned, UCB1-Normal, epsilon-greedy).
* Audibert, Munos, Szepesvári. Exploration-exploitation tradeoff using variance estimates in multi-armed bandits. Theoretical Computer Science, 2009 (UCB-V).
* Fan, Han, Lv, Xu, Zhou. Precise asymptotics and refined regret of variance-aware UCB. NeurIPS 2025 (UCB-V2 here).
* Mukherjee, Naveen, Sudarsanam, Ravindran. Efficient-UCBV: an almost optimal algorithm using variance estimates. AAAI 2018.
* Garivier, Cappé. The KL-UCB algorithm for bounded stochastic bandits and beyond. COLT 2011.
* Agrawal, Goyal. Thompson Sampling for contextual bandits with linear payoffs. ICML 2013.
* Drugan, Nowé. Designing multi-objective multi-armed bandits algorithms: a study. IJCNN 2013 (Pareto UCB1).
* Abbasi-Yadkori, Pál, Szepesvári. Improved algorithms for linear stochastic bandits. NeurIPS 2011 (OFUL radius).
* Dani, Hayes, Kakade. Stochastic linear optimization under bandit feedback. COLT 2008 (ConfidenceBall1).
* Yu, King, Lyu. CBRAP: contextual bandits with random projection. AAAI 2017.
* Kuzborskij, Cella, Cesa-Bianchi. Efficient linear bandits through matrix sketching. AISTATS 2019 (SOFUL).
* Chen et al. Efficient and robust high-dimensional linear contextual bandits. IJCAI 2020 (CBSCFD).
* Gentile, Li, Zappella. Online clustering of bandits. ICML 2014 (CLUB).
* Li, Chen, Li, Leung. Improved algorithm on online clustering of bandits. IJCAI 2019 (SCLUB).
* Dolan, Moré. Benchmarking optimization software with performance profiles. Mathematical Programming, 2002.
