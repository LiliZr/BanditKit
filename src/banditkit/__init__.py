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
from .experiment import DEFAULT_PARAM_GRID, TRACKABLE, Experiment  # noqa: E402
from .policies import *  # noqa: E402,F401,F403
from .policies import __all__ as _policies_all  # noqa: E402
from .results import Results  # noqa: E402
from .spec import EnvSpec, Observation  # noqa: E402
from .trace import Trace, trace  # noqa: E402

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
    "Experiment",
    "TRACKABLE",
    "DEFAULT_PARAM_GRID",
    "Results",
    "Trace",
    "trace",
    *_policies_all,
]
