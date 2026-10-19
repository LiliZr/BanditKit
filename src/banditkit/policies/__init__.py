"""Bandit policies."""
from .base import Policy
from .generic import UCB, EpsilonGreedy, ThompsonSampling
from .linear import CBRAP, ConfidenceBall1, LinGreedy, LinTS, LinUCB
from .mab import EUCBV, KLUCB, UCB1, UCBV, UCBV2, Random, Softmax, UCB1Normal, UCB1Tuned, UCBMin, UCBMinV1, UCBMinV2
from .mab import EpsilonGreedy as MABEpsilonGreedy
from .pareto import ParetoUCB1
from .sketch import CBSCFD, SOFUL
from .thompson import BernoulliThompsonSampling, GaussianThompsonSampling

__all__ = [
    "Policy", "UCB", "ThompsonSampling", "EpsilonGreedy", "Random", "UCB1", "UCB1Tuned", "UCB1Normal", "UCBV",
    "UCBV2", "UCBMin", "UCBMinV1", "UCBMinV2", "EUCBV", "KLUCB", "Softmax", "MABEpsilonGreedy",
    "BernoulliThompsonSampling", "GaussianThompsonSampling", "ParetoUCB1", "LinUCB", "CBRAP", "ConfidenceBall1",
    "LinTS", "LinGreedy", "SOFUL", "CBSCFD",
]
