"""
Likelihood for: scalar output (dy=1), N inputs x_i, M independent
replicates per input, with KNOWN N x N covariance Sigma across the x_i
(interpretation C in the user's design space).

Model:
    Y[:, m] = f(X, beta) + eps_m,   eps_m ~ N(0, Sigma),   m = 1..M
    eps_m's are independent across m.

Likelihood:
    log p(Y|beta) = -1/2 * N*M * log(2*pi)
                    -1/2 * M * log|Sigma|
                    -1/2 * sum_m r_m^T Sigma^{-1} r_m
        where r_m = Y[:, m] - f(X, beta).

Equivalent sufficient-statistic form (used here when M > 1):
    sum_m r_m^T Sigma^{-1} r_m
        = M * (ybar - f)^T Sigma^{-1} (ybar - f)
          + trace(S Sigma^{-1})
        where ybar = mean_m Y[:,m]
              S    = sum_m (Y[:,m]-ybar)(Y[:,m]-ybar)^T
The second term is precomputable, leaving an O(N^2) cost per eval.

Sigma may be:
    scalar         => Sigma = sigma2 * I_N  (homoscedastic)
    1D (N,)        => diagonal with given variances
    2D (N, N)      => full covariance
"""

import numpy as np


class ScalarObsRepKnownCov:
    def __init__(self, X, Y, prev_model, Sigma, use_sufficient=True):
        # ---- canonicalise X, Y ------------------------------------------
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            # user passed a single replicate as a 1D array -> (N, 1)
            Y = Y[:, None]
        elif Y.ndim != 2:
            raise ValueError(
                f"Y must be 1D (single replicate) or 2D (N, M); got ndim={Y.ndim}.")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[0] != Y.shape[0]:
            if X.shape[1] == Y.shape[0]:
                X = X.T
            else:
                raise ValueError(
                    f"X shape {X.shape} incompatible with Y shape {Y.shape}; "
                    "expected Y to be (N, M) and X to be (N, dx).")
        self.X = X                              # (N, dx)
        self.Y = Y                              # (N, M)
        self.N, self.M = Y.shape
        self.prev_model = prev_model
        self.use_sufficient = use_sufficient and (self.M > 1)

        # ---- factorise Sigma --------------------------------------------
        S = np.asarray(Sigma, dtype=float)
        if S.ndim == 0:
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
            self.diag_var = S
            self.logdet = float(np.sum(np.log(S)))
        elif S.ndim == 2:
            if S.shape != (self.N, self.N):
                raise ValueError(
                    f"2D Sigma must be ({self.N},{self.N}), got {S.shape}.")
            if not np.allclose(S, S.T, atol=1e-10):
                raise ValueError("Sigma must be symmetric.")
            self.kind = "full"
            self.L = np.linalg.cholesky(S)         # Sigma = L L^T
            self.logdet = 2.0 * float(np.sum(np.log(np.diag(self.L))))
        else:
            raise ValueError(f"Sigma has ndim={S.ndim}, expected 0/1/2.")

        # ---- sufficient-statistic precomputations -----------------------
        # ybar:      (N,)
        # scatter:   trace(S Sigma^{-1}) where S = sum_m (Y_m - ybar)(Y_m-ybar)^T
        self.ybar = self.Y.mean(axis=1)            # (N,)
        if self.use_sufficient:
            D = self.Y - self.ybar[:, None]        # (N, M); zero-mean residuals
            self._scatter_trace = self._quad_columns(D)
        else:
            self._scatter_trace = 0.0

        # constant in front of the quadratic; independent of beta
        self._cst = (-0.5 * self.N * self.M * np.log(2 * np.pi)
                     - 0.5 * self.M * self.logdet)

    # ---- core: sum_m c_m^T Sigma^{-1} c_m for an (N, K) matrix C ----------
    def _quad_columns(self, C):
        """C shape (N, K).  Returns sum_k C[:,k]^T Sigma^{-1} C[:,k]."""
        if self.kind == "scalar":
            return float(np.sum(C * C) / self.sigma2)
        if self.kind == "diag":
            return float(np.sum((C * C) / self.diag_var[:, None]))
        # full: solve L Z = C with K right-hand sides, then ||Z||_F^2
        Z = np.linalg.solve(self.L, C)              # (N, K)
        return float(np.sum(Z * Z))

    # ---- residual --------------------------------------------------------
    def _f(self, beta):
        f = np.ravel(self.prev_model(self.X, beta))
        if f.shape != (self.N,):
            raise ValueError(
                f"prev_model returned shape {f.shape}, expected ({self.N},).")
        return f

    # ---- public log-likelihood -------------------------------------------
    def loglike(self, beta):
        f = self._f(beta)                           # (N,)
        if self.use_sufficient:
            d = (self.ybar - f)[:, None]            # (N, 1)
            quad = self.M * self._quad_columns(d) + self._scatter_trace
        else:
            R = self.Y - f[:, None]                 # (N, M)
            quad = self._quad_columns(R)
        return self._cst - 0.5 * quad
