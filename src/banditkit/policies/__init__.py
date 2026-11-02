"""Bandit policies.

Generic names, resolved from the problem: :class:`UCB`, :class:`ThompsonSampling`,
:class:`EpsilonGreedy`, :class:`Random`.
"""
from .base import ParameterWarning, Policy
from .generic import UCB, EpsilonGreedy, ThompsonSampling
from .linear import CBRAP, ConfidenceBall1, LinGreedy, LinTS, LinUCB
from .mab import EUCBV, KLUCB, UCB1, UCBV, UCBV2, Random, Softmax, UCB1Normal, UCB1Tuned, UCBMin, UCBMinV1, UCBMinV2
from .mab import EpsilonGreedy as MABEpsilonGreedy
from .multi_user import CLUB, SCLUB
from .pareto import ParetoUCB1
from .sketch import CBSCFD, SOFUL
from .thompson import BernoulliThompsonSampling, GaussianThompsonSampling

__all__ = [
    "Policy", "ParameterWarning",
    # generic
    "UCB", "ThompsonSampling", "EpsilonGreedy", "Random",
    # MAB and multi-objective MAB
    "UCB1", "UCB1Tuned", "UCB1Normal", "UCBV", "UCBV2", "UCBMin", "UCBMinV1", "UCBMinV2", "EUCBV", "KLUCB",
    "Softmax", "MABEpsilonGreedy", "BernoulliThompsonSampling", "GaussianThompsonSampling", "ParetoUCB1",
    # linear
    "LinUCB", "CBRAP", "ConfidenceBall1", "LinTS", "LinGreedy", "SOFUL", "CBSCFD",
    # multi-user linear
    "CLUB", "SCLUB",
]
