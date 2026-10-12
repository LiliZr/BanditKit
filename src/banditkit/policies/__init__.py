"""Bandit policies."""
from .base import Policy
from .mab import EUCBV, KLUCB, UCB1, UCBV, UCBV2, Random, Softmax, UCB1Normal, UCB1Tuned, UCBMin, UCBMinV1, UCBMinV2
from .mab import EpsilonGreedy as MABEpsilonGreedy

__all__ = [
    "Policy", "Random", "UCB1", "UCB1Tuned", "UCB1Normal", "UCBV", "UCBV2", "UCBMin", "UCBMinV1", "UCBMinV2",
    "EUCBV", "KLUCB", "Softmax", "MABEpsilonGreedy",
]
