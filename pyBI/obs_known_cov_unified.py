"""
Scalar-output (dy=1) Bayesian observation object with KNOWN noise
covariance.

Likelihood (always):
    log p(y|beta) = -1/2 N log(2 pi) - 1/2 log|Sigma| - 1/2 r^T Sigma^{-1} r
    r = y - f(X, beta)

Sigma can be supplied in any of three ways:

  1. As a precomputed matrix:
        ObsKnownCov(X, y, model, Sigma=S_matrix)
     where S_matrix is either
       - 1D (N,)  : diagonal, S_ii = variances[i]
       - 2D (N,N) : full

  2. As a kernel/builder callable:
        ObsKnownCov(X, y, model, sigma_fn=k)
     where k(i, j, X, y) -> float returns Sigma[i, j].
     Called once at construction over all (i, j) pairs; never again.
     This lets you build Sigma from x_i, y_i, or anything else.
     (Sigma must be symmetric and positive definite; the class checks.)

REPLICATES BY STACKING
----------------------
There is no "replicate" API. If you have M measurements at the same x_i,
just include M identical rows in X (and the M values in y), and provide
Sigma entries that encode the within-replicate correlation, e.g.:
   - independent replicates : Sigma(x_i, x_j) = sigma^2 * delta_{ij}
   - perfectly correlated   : Sigma(x_i, x_j) = sigma^2 for all i,j with x_i=x_j
   - intermediate           : whatever your physics dictates.

The class does not need to know which rows are duplicates -- the structure
is fully encoded in Sigma.
"""

import numpy as np


class ObsKnownCov:
    def __init__(self, X, y, prev_model, Sigma=None, sigma_fn=None,
                 jitter=0.0):
        """
        Parameters
        ----------
        X : array, shape (N,) or (N, dx)
            Input locations.
        y : array, shape (N,) or (N, 1)
            Scalar observations.
        prev_model : callable
            prev_model(X, beta) -> array of shape (N,) or (N, 1).
            Output is coerced to (N,) by ravel().
        Sigma : None or scalar or 1D (N,) array or 2D (N, N) array
            Known noise covariance. Mutually exclusive with sigma_fn.
        sigma_fn : None or callable(i, j, X, y) -> float
            Element-wise covariance builder. Called for all i, j pairs
            (vectorised internally if it can be) at construction.
            Mutually exclusive with Sigma.
        jitter : float
            Small diagonal added to a 2D Sigma for numerical conditioning.
            Default 0; bump up if Cholesky fails.
        """
        # --- canonicalise X, y ----------------------------------------
        y = np.ravel(np.asarray(y, dtype=float))
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]} entries.")
        self.X = X                                  # (N, dx)
        self.y = y                                  # (N,)
        self.N = y.shape[0]
        self.prev_model = prev_model

        # --- canonicalise Sigma ---------------------------------------
        if (Sigma is None) == (sigma_fn is None):
            raise ValueError(
                "Provide exactly one of Sigma=... or sigma_fn=..., not both.")

        if sigma_fn is not None:
            S = self._build_from_kernel(sigma_fn)
            self._init_full(S, jitter)
            return

        S = np.asarray(Sigma, dtype=float)
        if S.ndim == 0:
            # scalar variance: homoscedastic Sigma = s2 * I_N
            if S <= 0:
                raise ValueError("scalar Sigma must be > 0.")
            self.kind = "scalar"
            self.sigma2 = float(S)
            self.logdet = self.N * np.log(self.sigma2)
        elif S.ndim == 1:
            if S.shape[0] != self.N:
                raise ValueError(
                    f"1D Sigma must have length N={self.N}, got {S.shape[0]}.")
            if np.any(S <= 0):
                raise ValueError("1D Sigma entries must be > 0.")
            self.kind = "diag"
            self.diag_var = S.copy()
            self.logdet = float(np.sum(np.log(self.diag_var)))
        elif S.ndim == 2:
            self._init_full(S, jitter)
        else:
            raise ValueError(f"Sigma has ndim={S.ndim}, expected 0/1/2.")

        self._cst = -0.5 * self.N * np.log(2 * np.pi) - 0.5 * self.logdet

    # ------------------------------------------------------------------
    def _build_from_kernel(self, sigma_fn):
        """Try a vectorised call first; fall back to a Python loop."""
        N = self.N
        # vectorised attempt: sigma_fn might accept arrays of indices
        try:
            ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
            S = np.asarray(sigma_fn(ii, jj, self.X, self.y), dtype=float)
            if S.shape != (N, N):
                raise TypeError("vectorised path returned wrong shape")
            return S
        except (TypeError, ValueError, IndexError):
            S = np.empty((N, N), dtype=float)
            for i in range(N):
                for j in range(N):
                    S[i, j] = float(sigma_fn(i, j, self.X, self.y))
            return S

    def _init_full(self, S, jitter):
        if S.shape != (self.N, self.N):
            raise ValueError(
                f"2D Sigma must be ({self.N},{self.N}), got {S.shape}.")
        if not np.allclose(S, S.T, atol=1e-10):
            raise ValueError("Sigma must be symmetric (within 1e-10).")
        if jitter > 0:
            S = S + jitter * np.eye(self.N)
        try:
            self.L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                "Cholesky of Sigma failed; Sigma may not be positive "
                "definite. Try increasing `jitter` or check your kernel."
            ) from e
        self.kind = "full"
        self.logdet = 2.0 * float(np.sum(np.log(np.diag(self.L))))
        self._cst = -0.5 * self.N * np.log(2 * np.pi) - 0.5 * self.logdet

    # ------------------------------------------------------------------
    def _residual(self, beta):
        f = np.ravel(self.prev_model(self.X, beta))
        if f.shape != self.y.shape:
            raise ValueError(
                f"prev_model returned shape {f.shape}, expected {self.y.shape}.")
        return self.y - f

    def _quad(self, r):
        if self.kind == "scalar":
            return float(np.dot(r, r) / self.sigma2)
        if self.kind == "diag":
            return float(np.sum(r * r / self.diag_var))
        # full
        z = np.linalg.solve(self.L, r)
        return float(z @ z)

    # ------------------------------------------------------------------
    def loglike(self, beta):
        return self._cst - 0.5 * self._quad(self._residual(beta))

    # legacy attribute names for compatibility with pyBI inference loop
    @property
    def Ndata(self):  return self.N
    @property
    def dimdata(self): return 1
    @property
    def obs(self):     return self.y[:, None]
    @property
    def cond_var(self): return self.X
