"""banditkit: multi-armed, multi-objective and linear bandits behind one API."""
__version__ = "0.1.0"

from .envs import (  # noqa: E402
    Environment,
)
from .spec import EnvSpec, Observation  # noqa: E402

__all__ = [
    "__version__",
    "EnvSpec",
    "Observation",
    "Environment",
]
