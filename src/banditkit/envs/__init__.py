"""Environments: they generate rewards and know the mean rewards (hence the optimum)."""
from .base import Environment
from .factory import make_env
from .linear import DynamicLinearEnv, LinearEnv
from .mab import BernoulliMAB, EmpiricalMAB, GaussianMAB, GroupRecommendationMAB
from .multi_user import MultiUserLinearEnv

__all__ = ["Environment", "make_env", "GaussianMAB", "BernoulliMAB", "EmpiricalMAB", "GroupRecommendationMAB",
           "LinearEnv", "DynamicLinearEnv", "MultiUserLinearEnv"]
