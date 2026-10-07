"""Environments: they generate rewards and know the mean rewards (hence the optimum)."""
from .base import Environment
from .linear import DynamicLinearEnv, LinearEnv
from .mab import BernoulliMAB, EmpiricalMAB, GaussianMAB, GroupRecommendationMAB

__all__ = ["Environment", "GaussianMAB", "BernoulliMAB", "EmpiricalMAB", "GroupRecommendationMAB", "LinearEnv", "DynamicLinearEnv"]
