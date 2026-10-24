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
