"""Datasets: EXAMPLES of preprocessing that turn ratings into multi-user linear bandits.

Any dataset and any preprocessing works, as long as its output has the input types of
the environments. Needs pandas and scipy (``pip install banditkit[datasets]``).
"""
from .base import LinearDataset, filter_min_counts, load_npz, ratings_to_linear
from .movielens import MOVIELENS_VARIANTS, fetch_movielens, get_data_home, movielens_linear

__all__ = ["LinearDataset", "load_npz", "ratings_to_linear", "filter_min_counts",
           "fetch_movielens", "movielens_linear", "MOVIELENS_VARIANTS", "get_data_home"]
