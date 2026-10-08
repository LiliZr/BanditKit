"""banditkit: multi-armed, multi-objective and linear bandits behind one API."""
__version__ = "0.1.0"

from .envs import (  # noqa: E402
    BernoulliMAB,
    DynamicLinearEnv,
    EmpiricalMAB,
    Environment,
    GaussianMAB,
    GroupRecommendationMAB,
    LinearEnv,
    MultiUserLinearEnv,
    make_env,
)
from .spec import EnvSpec, Observation  # noqa: E402

__all__ = [
    "__version__",
    "EnvSpec",
    "Observation",
    "Environment",
    "make_env",
    "GaussianMAB",
    "BernoulliMAB",
    "EmpiricalMAB",
    "GroupRecommendationMAB",
    "LinearEnv",
    "DynamicLinearEnv",
    "MultiUserLinearEnv",
]
