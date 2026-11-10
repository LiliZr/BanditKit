"""Sketched linear bandits: SOFUL and CBSCFD, based on Frequent Directions.

Both replace the d x d design matrix by a sketch S: ``V = c I + S^T S``. With
``H = (S S^T + c I)^-1``, the Woodbury identity gives ``V^-1 = (I - S^T H S) / c``.

Each paper uses its own Frequent Directions variant:

* SOFUL: the sketch has m rows. At every step x is added, an SVD shrinks every
  squared singular value by ``delta = sigma_m^2`` (the m-th one), and m - 1
  non-zero rows remain.
* CBSCFD: rows are appended until the buffer holds 2m rows; only then the SVD
  shrinks by ``delta = sigma_m^2``, which leaves m - 1 non-zero rows.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from .._utils import argmax_random
from .._validation import ParamSpec
from ..spec import EnvSpec, Observation
from .linear import LinearPolicy

__all__ = ["SOFUL", "CBSCFD"]


class SketchPolicy(LinearPolicy):
    """Frequent Directions sketch shared by SOFUL and CBSCFD."""

    params = {"sketch_size": ParamSpec("int", default=10, low=2, doc="Sketch size m.")}
    #: CBSCFD only: the regularization c grows with the sketch error.
    adaptive_regularization = False
    #: "standard": shrink at every step (SOFUL); "fast": shrink when 2m rows are buffered (CBSCFD).
    fd_variant = "standard"

    @property
    def techniques(self) -> Tuple[str, ...]:
        return ()

    def _check_params(self) -> None:
        if self.n_components is not None:
            raise ValueError(f"{type(self).__name__} sketches the features itself; `n_components` is not supported.")

    def _check_spec(self, spec: EnvSpec) -> None:
        if self.sketch_size >= spec.n_features:
            raise ValueError(f"`sketch_size` ({self.sketch_size}) must be smaller than n_features ({spec.n_features}).")

    def _reset(self) -> None:
        super()._reset()
        self.sketch = np.zeros((0, self.dim))    # S: non-zero rows only
        self.gram_inv = np.zeros((0, 0))         # H = (S S^T + c I)^-1
        self.b = np.zeros(self.dim)              # sum reward * x
        self.theta = np.zeros(self.dim)          # estimate V^-1 b
        self.sketch_error = 0.0                  # sum of the shrinkages delta (rho in SOFUL, Delta in CBSCFD)
        self.n_obs = 0
        self._sq_norms = None

    @property
    def regularization(self) -> float:
        """Current c in V = c I + S^T S."""
        return self.lambda_ + (self.sketch_error if self.adaptive_regularization else 0.0)

    def _working_features(self, obs: Observation) -> np.ndarray:
        changed = obs.action_features is not self._raw_features
        X = super()._working_features(obs)
        if changed or self._sq_norms is None:
            self._sq_norms = np.einsum("ij,ij->i", X, X)
        return X

    def _widths(self, X: np.ndarray) -> np.ndarray:
        """``||x||_{V^-1}`` for every action, through the Woodbury identity."""
        if self.sketch.shape[0] == 0:
            return np.sqrt(self._sq_norms / self.regularization)
        SX = self.sketch @ X.T
        q = self._sq_norms - np.einsum("ik,ik->k", SX, self.gram_inv @ SX)
        return np.sqrt(np.maximum(q, 0.0) / self.regularization)

    def _predict(self, obs: Observation) -> int:
        X = self._working_features(obs)
        return argmax_random(X @ self.theta + self.alpha * self._beta(self.n_obs) * self._widths(X), self.rng)

    def _update(self, obs: Observation, action: int, reward: np.ndarray, scalar_reward: float) -> None:
        x = self._working_features(obs)[action]
        self.b += scalar_reward * x
        if self.fd_variant == "standard":
            self._shrink(np.vstack([self.sketch, x]))
        elif self.sketch.shape[0] + 1 < 2 * self.sketch_size:
            self._append_row(x)
        else:
            self._shrink(np.vstack([self.sketch, x]))
        S, c = self.sketch, self.regularization
        self.theta = (self.b - S.T @ (self.gram_inv @ (S @ self.b))) / c
        self.n_obs += 1

    def _append_row(self, x: np.ndarray) -> None:
        """Add x to the sketch and update H by blocks (Schur complement)."""
        Sx = self.sketch @ x
        p = self.gram_inv @ Sx
        schur = float(x @ x) + self.regularization - float(Sx @ p)
        r = self.gram_inv.shape[0]
        H = np.empty((r + 1, r + 1))
        H[:r, :r] = self.gram_inv + np.outer(p, p) / schur
        H[:r, r] = H[r, :r] = -p / schur
        H[r, r] = 1.0 / schur
        self.gram_inv, self.sketch = H, np.vstack([self.sketch, x])

    def _shrink(self, buffer: np.ndarray) -> None:
        """Frequent Directions step: subtract delta = sigma_m^2 from every squared singular value.

        With fewer than m rows, sigma_m = 0 and the step only rotates the rows.
        """
        m = self.sketch_size
        _, sing, Vt = np.linalg.svd(buffer, full_matrices=False)
        delta = sing[m - 1] ** 2 if sing.size >= m else 0.0
        self.sketch_error += delta
        shrunk = np.maximum(sing[: m - 1] ** 2 - delta, 0.0)
        keep = shrunk > 0.0
        self.sketch = np.sqrt(shrunk[keep])[:, None] * Vt[: m - 1][keep]
        self.gram_inv = np.diag(1.0 / (shrunk[keep] + self.regularization))  # rows of the sketch are orthogonal

    def diagnostics(self) -> Dict[str, Any]:
        return {"sketch_error": self.sketch_error}


class SOFUL(SketchPolicy):
    """SOFUL (Kuzborskij, Cella and Cesa-Bianchi, AISTATS 2019).

    Frequent Directions with m rows, shrunk at every step (Algorithm 4 of the
    paper). Radius (Theorem 2), with rho the accumulated sketch error::

        R sqrt(m ln(1 + n L^2 / (m lambda)) + 2 ln(1/delta) + k ln(1 + rho/lambda)) sqrt(1 + rho/lambda)
            + S sqrt(lambda) (1 + rho/lambda)
    """

    base_name = "SOFUL"

    def _beta(self, n_obs: int) -> float:
        m, k, lam = self.sketch_size, self.dim, self.lambda_
        ratio = self.sketch_error / lam
        inner = (m * math.log(1.0 + n_obs * self._L ** 2 / (m * lam)) + 2.0 * math.log(1.0 / self.delta)
                 + k * math.log(1.0 + ratio))
        return (self._R * math.sqrt(inner) * math.sqrt(1.0 + ratio)
                + self._S * math.sqrt(lam) * (1.0 + ratio))


class CBSCFD(SketchPolicy):
    """CBSCFD (Chen, Luo, Zhang, Yu and Lian, IJCAI 2020): Frequent Directions with an adaptive regularization.

    The sketch (SCFD) buffers 2m rows before each shrink, and the regularization
    grows with the sketch error: ``c = lambda + Delta``, Delta the sum of the
    shrinkages. Radius (Theorem 2)::

        R sqrt(2 ln(1/delta) + m ln(1 + n L^2 / (m lambda)) + k ln(1 + Delta/lambda)) + S sqrt(lambda + Delta)
    """

    base_name = "CBSCFD"
    adaptive_regularization = True
    fd_variant = "fast"

    def _beta(self, n_obs: int) -> float:
        m, k, lam, err = self.sketch_size, self.dim, self.lambda_, self.sketch_error
        inner = (2.0 * math.log(1.0 / self.delta) + m * math.log(1.0 + n_obs * self._L ** 2 / (m * lam))
                 + k * math.log(1.0 + err / lam))
        return self._R * math.sqrt(inner) + self._S * math.sqrt(lam + err)
