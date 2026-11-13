"""Datasets: EXAMPLES of preprocessing, and the input format for your own data.

The functions of this module are examples. They turn MovieLens and Steam into bandit
environments by following published experiments (references in each docstring and
in the README). The rest of banditkit does not depend on them: any dataset and any
preprocessing works, as long as its output has the input types of the environments
(arrays passed to ``bk.make_env``, or an ``.npz`` file read by :func:`load_npz`).

* :func:`movielens_linear`, :func:`steam_linear`: multi-user linear bandits (Li et al., IJCAI 2019;
  Wang et al., NeurIPS 2023).
* :func:`ratings_to_linear`: the same steps on any ratings table (user_id, item_id, rating).
* :func:`movielens_mab`: classical MAB whose rewards are real ratings (Gupta et al., IEEE TIT 2021);
  :func:`rewards_to_mab` does the same on any data.
* :func:`load_npz`: environment from an .npz file written by any preprocessing (keys in :mod:`banditkit.datasets.base`).

Needs pandas and scipy (``pip install banditkit[datasets]``).
"""
from .base import LinearDataset, filter_min_counts, load_npz, ratings_to_linear
from .movielens import MOVIELENS_VARIANTS, fetch_movielens, get_data_home, movielens_linear
from .real_mab import movielens_mab, rewards_to_mab
from .steam import load_steam, steam_linear

__all__ = ["LinearDataset", "load_npz", "ratings_to_linear", "filter_min_counts",
           "fetch_movielens", "movielens_linear", "MOVIELENS_VARIANTS", "get_data_home",
           "load_steam", "steam_linear", "rewards_to_mab", "movielens_mab"]
