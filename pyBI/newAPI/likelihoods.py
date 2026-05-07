"""
Likelihood objects. All three implement the same interface:

    .loglike(theta) -> float
    .extra_param_names() -> list[str]   (parameters owned by the likelihood)
    .set_extra(param_name, value)        (sampler pokes inferred values back)

Three modes, matching the user spec:

  1. FullCovLike     -- Sigma in R^{NxN} supplied at construction.
                        Only beta inferred. Likelihood depends on beta only.

  2. FixedDiagLike   -- Sigma = sigma2 * I_N, sigma2 supplied.
                        Only beta inferred. Likelihood depends on beta only.

  3. InferredDiagLike -- Sigma = sigma2 * I_N, sigma2 inferred.
                        beta and sigma2 inferred. Likelihood depends on
                        beta AND on sigma2 set via set_extra("sigma", ...).

The "extra" mechanism is how an inferred nuisance parameter (sigma in
mode 3) is communicated to the likelihood without baking it into the
forward-model interface. The sampler treats sigma like any other block
(prior + proposal); the likelihood reads its current value through
set_extra.
"""

import numpy as np


class _BaseLike:
    """Common factor: residual computation, shape coercion."""

    def __init__(self, X, y, model):
        y = np.ravel(np.asarray(y, dtype=float))
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]} entries.")
        self.X = X
        self.y = y
        self.N = y.shape[0]
        self.model = model

    def _residual(self, beta):
        f = np.ravel(self.model(self.X, beta))
        if f.shape != self.y.shape:
            raise ValueError(
                f"model returned shape {f.shape}, expected {self.y.shape}.")
        return self.y - f

    # default extras: none
    def extra_param_names(self):
        return []

    def set_extra(self, name, value):
        raise KeyError(f"unknown extra parameter {name!r}")


# ----------------------------------------------------------------------
# 1. Full covariance, supplied
# ----------------------------------------------------------------------
class FullCovLike(_BaseLike):
    """y - f(X,beta) ~ N(0, Sigma) with Sigma in R^{NxN} known."""

    def __init__(self, X, y, model, Sigma, jitter=0.0):
        super().__init__(X, y, model)
        S = np.asarray(Sigma, dtype=float)
        if S.shape != (self.N, self.N):
            raise ValueError(
                f"Sigma shape must be ({self.N},{self.N}); got {S.shape}.")
        if not np.allclose(S, S.T, atol=1e-10):
            raise ValueError("Sigma must be symmetric.")
        if jitter > 0:
            S = S + jitter * np.eye(self.N)
        self.L = np.linalg.cholesky(S)
        self.logdet = 2.0 * float(np.sum(np.log(np.diag(self.L))))
        self._cst = -0.5 * self.N * np.log(2 * np.pi) - 0.5 * self.logdet

    def loglike(self, beta):
        r = self._residual(beta)
        z = np.linalg.solve(self.L, r)
        return self._cst - 0.5 * float(z @ z)


# ----------------------------------------------------------------------
# 2. Diagonal, sigma supplied
# ----------------------------------------------------------------------
class FixedDiagLike(_BaseLike):
    """y - f(X,beta) ~ N(0, sigma^2 I_N) with sigma known."""

    def __init__(self, X, y, model, sigma):
        super().__init__(X, y, model)
        sigma = float(sigma)
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.sigma = sigma
        self._cst = (-0.5 * self.N * np.log(2 * np.pi)
                     - self.N * np.log(self.sigma))

    def loglike(self, beta):
        r = self._residual(beta)
        return self._cst - 0.5 * float(r @ r) / self.sigma ** 2


# ----------------------------------------------------------------------
# 3. Diagonal, sigma to be inferred
# ----------------------------------------------------------------------
class InferredDiagLike(_BaseLike):
    """
    y - f(X,beta) ~ N(0, sigma^2 I_N) with sigma to be inferred.

    The current sigma is communicated by the sampler via
       set_extra("sigma", current_value).
    The likelihood holds it as a mutable attribute so loglike(beta)
    has the standard signature.
    """
    EXTRAS = ["sigma"]

    def __init__(self, X, y, model, sigma_init=1.0):
        super().__init__(X, y, model)
        self._sigma = float(sigma_init)
        self._neg_half_N_log_2pi = -0.5 * self.N * np.log(2 * np.pi)

    def extra_param_names(self):
        return list(self.EXTRAS)

    def set_extra(self, name, value):
        if name == "sigma":
            v = float(value)
            if v <= 0:
                # signal "out of support" without raising; loglike will
                # return -inf in that case.
                self._sigma = v
            else:
                self._sigma = v
        else:
            raise KeyError(f"unknown extra parameter {name!r}")

    def loglike(self, beta):
        if self._sigma <= 0:
            return -np.inf
        r = self._residual(beta)
        return (self._neg_half_N_log_2pi
                - self.N * np.log(self._sigma)
                - 0.5 * float(r @ r) / self._sigma ** 2)
