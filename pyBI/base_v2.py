import numpy as np ## ONLY REFERENCE LIBRARY
from scipy.optimize import minimize ## FOR GP HYPERPARAM TUNING

class RandVar():
    def __init__(self, param=None):
        if param is not None: self.param = param

    def logprior(self, x):
        return NotImplemented
    
    def proposal(self):
        return NotImplemented
    
    def draw(self):
        return NotImplemented


def rnvmultiD(m, Lmat):
    # Lmat = np.linalg.cholesky(Cov)
    return m + np.ravel(Lmat @ np.random.randn(m.shape[0],1))

def covToCorr(Sig):
    Dinv = np.diag(1/np.sqrt(np.diag(Sig)))
    return Dinv @ Sig @ Dinv


class UnifVar(RandVar):
    def __init__(self, param=None):
        super().__init__(param)
        self.min = self.param[0]
        self.max = self.param[1]

    def logprior(self, x):
        bcheck = (x > self.param[0]) & (x < self.param[1])
        logp = 1/(self.param[1] - self.param[0]) if bcheck else -1000
        return logp
    
    def proposal(self, m=0, s=1):
        return np.random.randn()*np.array(s)+m
    
    def draw(self):
        return np.random.rand()*(self.param[1] - self.param[0]) + self.param[0]


class NormVar(RandVar):
    def __init__(self, param=None):
        super().__init__(param)
        self.min = self.param[0] - 6*self.param[1]
        self.max = self.param[1] + 6*self.param[1]

    def logprior(self, x):
        logcst = -0.5 * np.log(2 * np.pi * self.param[1]**2)
        logp = -0.5 * ((x - self.param[0]) / self.param[1])**2 + logcst
        return logp
    
    def proposal(self, m=0, s=1):
        return np.random.randn()*np.array(s)+m
    
    def draw(self):
        return np.random.randn()*self.param[1] + self.param[0]
    
class HalfNormVar(RandVar):
    def __init__(self, param=None):
        super().__init__(param)
        self.min = 0
        self.max = 6*self.param

    def logprior(self, x):
        logcst = 0.5 * np.log(2) - 0.5 * np.log(np.pi) - np.log(self.param)
        logp = -(x**2) / (2 * self.param**2) + logcst if x > 0 else -np.inf
        return logp
    
    def proposal(self, m=0, s=1):
        return np.random.randn()*np.array(s)+m
    
    def draw(self):
        return np.abs(np.random.randn() * self.param)
    
    def diagSmat(self, s=None, N=1):
        s0 = self.param if s is None else s
        return np.eye(N)*s0**2


class InvGaussVar(RandVar):
    def __init__(self, param=None):
        super().__init__(param)
        self.min = 0
        self.max = (self.param[0]*self.param[2] + self.param[1])*4
    
    def logprior(self, x, loc=None, scale=None):
        mu = self.param[0]
        loc = self.param[1] if loc is None else loc
        scale = self.param[2] if scale is None else scale
        x_shifted = x - loc
        nu = mu * scale
        lambda_val = scale
        log_pdf = (0.5 * np.log(lambda_val) - 0.5 * np.log(2 * np.pi)
                - 1.5 * np.log(x_shifted) - (lambda_val * np.power(
                    x_shifted - nu, 2)) / (2 * np.power(nu, 2) * x_shifted))
        return -1000 if np.isnan(log_pdf) else log_pdf

    def proposal(self, m, s):
        prop = np.exp(np.random.randn()*s+np.log(m)) 
        return prop if prop != -np.inf else 0.0

    def draw(self, N=1):
        rnd = np.random.wald(mean=self.param[0]*self.param[2],
                              scale=self.param[2], size=N) + self.param[1]
        return rnd[0] if N==1 else rnd

    def mean(self):
        return self.param[0]*self.param[2] + self.param[1] if \
              len(self.param) > 2 else self.param[0] + self.param[1]

    def diagSmat(self, s=None, N=1):
        s0 = self.param[0]*self.param[2] + self.param[1] if s is None else s
        return np.eye(N)*s0**2



## GP AND GAUSSIAN LIKELIHOOD IMPLEMENTATION
## SHOULD PROBABLY BE DOUBLE CHECKED
## (BELOW)


class ObsVar():
    def __init__(self, obs, prev_model, cond_var=None):
        self.obs = obs
        self.Ndata = obs.shape[0]
        self.dimdata = obs.shape[1]
        if cond_var is not None: self.cond_var = cond_var
        self.prev_model = prev_model

    def loglike(self, par, sigma):
        N = self.obs.shape[0]
        d = self.obs.shape[1]
        try:
            L = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is not positive semi-definite "+
                             "(cannot perform Cholesky decomposition).")
        log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
        diff = self.prev_model(self.cond_var,par) - self.obs
        mahalanobis_term = np.zeros(N)
        for i in range(N):
            y = np.linalg.solve(L, diff[i, :])
            mahalanobis_term[i] = np.sum(y**2)
        log_likelihood = - (N * d / 2) * np.log(2 * np.pi) \
                    - (N / 2) * log_det_sigma \
                    - (1 / 2) * np.sum(mahalanobis_term)
        return log_likelihood
    
    def loglike_uv(self, par, sigma):
        N = self.obs.shape[0]
        diff = np.ravel(self.prev_model(self.cond_var,par)) - np.ravel(self.obs)
        SSE = np.sum(diff**2)
        log_likelihood = - (N / 2) * np.log(2 * np.pi) \
                       - N * np.log(sigma) \
                       - (1 / 2) * SSE / (sigma**2)
        return log_likelihood


class KnownCovObs():
    """
    Observation object for the case where the noise covariance is KNOWN
    (fixed, not inferred).

    Shape convention enforced everywhere:
        cond_var (X) : (N, dx)
        obs      (y) : (N, dy)        -- dy=1 for scalar output
        prev_model(X, beta) -> (N, dy)

    The likelihood is the single Nd-dimensional MVN (one realisation,
    NOT N iid draws):

        log p(y|beta) = -1/2 Nd log(2 pi)
                        -1/2 log|Sigma|
                        -1/2 r^T Sigma^{-1} r,   r = vec(y - f(X,beta))

    Sigma can be supplied in three forms ("kind"):
      - "diag"  : Sigma = sigma2 * I_{Nd}, sigma2 a positive scalar
      - "full"  : full (Nd, Nd) matrix
      - "kron"  : Sigma = I_N (x) Sigma_d, with Sigma_d a (dy, dy) matrix.
                  No (Nd, Nd) matrix is ever materialised.

    Optional `with_scale=True` enables an inferred multiplier alpha so that
    the EFFECTIVE covariance is alpha * Sigma. Useful when the *shape* of
    Sigma is trusted but its overall magnitude is not.
    """

    def __init__(self, obs, prev_model, cond_var, Sigma, kind="full",
                 with_scale=False):
        obs = np.atleast_2d(obs)
        if obs.shape[0] == 1 and obs.shape[1] != 1:
            # canonicalise scalar-output row vectors to (N, 1)
            obs = obs.T
        self.obs = obs                       # (N, dy)
        self.N = obs.shape[0]
        self.dy = obs.shape[1]
        self.Nd = self.N * self.dy

        self.prev_model = prev_model
        self.cond_var = cond_var
        # legacy attribute names used elsewhere in the codebase
        self.Ndata = self.N
        self.dimdata = self.dy
        self.with_scale = with_scale

        self.kind = kind
        if kind == "diag":
            self.sigma2 = float(Sigma)
            if self.sigma2 <= 0:
                raise ValueError("kind='diag' requires sigma2 > 0.")
            self._logdet = self.Nd * np.log(self.sigma2)
        elif kind == "full":
            S = np.asarray(Sigma, dtype=float)
            if S.shape != (self.Nd, self.Nd):
                raise ValueError(
                    "Sigma shape {} does not match (N*dy, N*dy) = ({}, {})."
                    " Did you mean kind='kron' for a (dy, dy) block?".format(
                        S.shape, self.Nd, self.Nd))
            if not np.allclose(S, S.T, atol=1e-10):
                raise ValueError("Sigma is not symmetric.")
            self._L = np.linalg.cholesky(S)
            self._logdet = 2.0 * np.sum(np.log(np.diag(self._L)))
        elif kind == "kron":
            Sd = np.asarray(Sigma, dtype=float)
            if Sd.shape != (self.dy, self.dy):
                raise ValueError(
                    "For kind='kron', Sigma must be (dy, dy) = ({}, {});"
                    " got {}.".format(self.dy, self.dy, Sd.shape))
            if not np.allclose(Sd, Sd.T, atol=1e-10):
                raise ValueError("Sigma_d is not symmetric.")
            self._Ld = np.linalg.cholesky(Sd)
            self._logdet = 2.0 * self.N * np.sum(np.log(np.diag(self._Ld)))
        else:
            raise ValueError("Unknown kind={!r}".format(kind))

    def _residual(self, par):
        f = self.prev_model(self.cond_var, par)
        f = np.atleast_2d(f)
        # forward models in this codebase are inconsistent:
        #   some return (N,), some (1, N), some (N, 1).
        # Coerce to (N, dy).
        if f.shape == (1, self.N) and self.dy == 1:
            f = f.T
        if f.shape != self.obs.shape:
            raise ValueError(
                "prev_model returned shape {}, expected {}. "
                "Forward model must return (N, dy).".format(
                    f.shape, self.obs.shape))
        return self.obs - f

    def _quad(self, R):
        """R is the (N, dy) residual; returns r^T Sigma^{-1} r."""
        if self.kind == "diag":
            return np.sum(R * R) / self.sigma2
        if self.kind == "kron":
            # With Sigma = I_N (x) Sigma_d, the quadratic form decomposes
            # as sum_i r_i^T Sigma_d^{-1} r_i. Solve Sigma_d^{-1} R^T via
            # the Cholesky in one shot.
            Y = np.linalg.solve(self._Ld, R.T)        # (dy, N)
            return float(np.sum(Y * Y))
        # full
        r = R.reshape(-1)                              # (Nd,) row-major
        y = np.linalg.solve(self._L, r)
        return float(y @ y)

    def loglike(self, par, alpha=None):
        """
        log p(y | par) under the fixed Sigma (optionally scaled by alpha).
        """
        R = self._residual(par)
        q = self._quad(R)
        logdet = self._logdet
        if self.with_scale and alpha is not None:
            # Sigma_eff = alpha * Sigma  =>
            #   log|Sigma_eff|     = Nd*log(alpha) + log|Sigma|
            #   r^T Sigma_eff^-1 r = q / alpha
            logdet = logdet + self.Nd * np.log(alpha)
            q = q / alpha
        return -0.5 * self.Nd * np.log(2 * np.pi) \
               - 0.5 * logdet - 0.5 * q







class HGP():
    def __init__(self, XX, y, kpar=None, spar=None):
        self.XX = XX
        self.dim = XX.shape[1]
        self.y = y
        self.bounds = [[1e-5, 1e5]] + self.default_tbounds()[0] + \
                        [[1e-8, np.var(self.y)*0.01]]
        if kpar is None: 
            self.kpar = np.exp(self.rng_par('log')[:self.dim+1])
        else: self.kpar = kpar
        if spar is None: 
            self.spar = np.exp(self.rng_par('log')[self.dim+1])
        else: self.spar = spar
        self.cond()

    def cond(self):
        self.k_XX = self.KaXX(self.XX, self.kpar[1:],
                               [self.kpar[0], self.spar]) + \
              np.eye(self.XX.shape[0])*1e-8
        self.L_XX = np.linalg.cholesky(self.k_XX)
        self.alpha = np.linalg.solve(self.L_XX.T,
                                      np.linalg.solve(self.L_XX, self.y))

    def setpar(self, kpar, spar, cond=True):
        self.kpar = kpar
        self.spar = spar
        if cond: 
            self.k_XX = self.KaXX(self.XX, theta=kpar[1:],
                                  sigma=[kpar[0], spar]) + \
                 np.eye(self.XX.shape[0])*1e-8
            self.L_XX = np.linalg.cholesky(self.k_XX)
            self.alpha = np.linalg.solve(self.L_XX.T,
                                      np.linalg.solve(self.L_XX, self.y))
            # self.cond()
    
    def raXX(self, X, Y, scl=None):
        if scl is None: scl = self.kpar[1:]
        diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        scaled_diff_sq = (diff / scl)**2
        squared_weighted_distance = np.sum(scaled_diff_sq, axis=2)
        return squared_weighted_distance 

    def KaXX(self, X, theta=None, sigma=None):
        if theta is None: theta = self.kpar[:1]
        if sigma is None: sigma = [self.kpar[0], self.spar]
        return sigma[0]**2 * np.exp(-self.raXX(X,X,theta)/2) + \
            np.eye(X.shape[0])*sigma[1]
    
    def KaxX(self, x, X, theta=None, sigma=None):
        if theta is None: theta = self.kpar[:1]
        if sigma is None: sigma = [self.kpar[0], self.spar]
        return sigma[0]**2 * np.exp(-self.raXX(x,X, theta)/2)
    
    def m_predict(self, x):
        K_xX = self.KaxX(x, self.XX, self.kpar[1:], [self.kpar[0], self.spar])
        mu_pred = K_xX @ self.alpha
        return mu_pred
    
    def cov_predict(self, x, noise=True):
        K_xX = self.KaxX(x, self.XX, self.kpar[1:], [self.kpar[0], self.spar])
        v = np.linalg.solve(self.L_XX, K_xX.T)
        cov_pred = self.KaxX(x, x, self.kpar[1:],
                              [self.kpar[0], self.spar]) - v.T @ v
        if noise: cov_pred += np.eye(cov_pred.shape[0])*self.spar
        return np.maximum(0,cov_pred)
    
    def loglike(self, logpar=None):
        if logpar is None: params = np.log(np.hstack([self.kpar, self.spar]))
        else: params = logpar
        d = self.XX.shape[1]
        N = self.XX.shape[0]
        s = np.exp(params[0])
        t = np.exp(params[1:1+d])
        sn = np.exp(params[1+d])
        K_XX = self.KaXX(self.XX, t, [s, sn])
        L = np.linalg.cholesky(K_XX)
        log_det_K_noisy = 2 * np.sum(np.log(np.diag(L)))
        alpha = np.linalg.solve(L.T,np.linalg.solve(L, self.y))
        data_fit_term = self.y.T @ alpha
        log_marginal_likelihood = -0.5 * data_fit_term - \
                        0.5 * log_det_K_noisy - \
                        0.5 * N * np.log(2 * np.pi)
        return -log_marginal_likelihood
    
    def default_tbounds(self, default=[0.05, 5]):
        xmin = np.min(self.XX, axis=0)
        xmax = np.max(self.XX, axis=0)
        xL = xmax - xmin
        bounds = [[default[0]*L, default[1]*L] for L in xL]
        return bounds, xL
    
    
    def rng_par(self, dtype='log'):
        if dtype == 'log':
            bounds = np.log(np.array(self.bounds))
        elif dtype == 'lin': bounds = np.array(self.bounds)
        return np.random.rand() * (bounds[:,1] - \
                                   bounds[:,0]) + \
                                   bounds[:,0]   
    
    def default_tune(self, x0=None, update=False, verbose=True):
        x0 = self.rng_par() if x0 is None else x0
        result = minimize(
        fun=self.loglike,
        x0=x0,
        method='L-BFGS-B',
        bounds=np.log(self.bounds),
        # options={'disp': True, 'maxiter': 1000}
        )
        sol = np.exp(result.x)
        ll = result.fun
        if verbose: print(sol)
        if update:
            self.kpar = sol[:self.dim+1]
            self.spar = sol[self.dim+1]
            self.cond()
        return sol, ll
    
    def mtune(self, N=10, verbose=False):
        lmin = 1e6
        for i in range(N):
            sol, ll = self.default_tune(verbose=verbose)
            if ll < lmin:
                lmin = ll
                best_sol = sol
        self.setpar(best_sol[:self.dim+1], best_sol[self.dim+1], cond=True)
        return best_sol
    
    def __str__(self):
        strout = ""
        strout += "kernel parameters : " + str(list(self.kpar)) + "\n"
        strout += "relative lengthscale : " + ", ".join(["{:.2f}".format(el) 
            for el in list(self.kpar[1:]/self.default_tbounds()[1])]) + "\n"
        strout += "nugget : " + str(self.spar) + "\n"
        strout += "loglike : " + str(self.default_tune(verbose=False)[1]) + "\n"
        strout += "optim bounds : \n" + "\n".join([str(el) for el in 
                                       list(self.bounds)])
        return strout


class GaussLike():
    def __init__(self, obs, gmod, x, gpar, kpar, spar):
        self.obs = obs
        self.x = x
        self.gmod = gmod
        self.Ndata = obs.shape[0]
        self.dimdata = obs.shape[1]
        self.gpar = gpar
        self.kpar = kpar
        self.spar = spar
        self.mgp = HGP(self.x, np.ravel(self.obs - \
                self.g_predict(self.x, par=self.gpar)))
        self.mgp.setpar(self.kpar, self.spar)

    def setpar(self, gpar, kpar, spar):
        self.gpar = gpar # parameters of g(x,beta)
        self.kpar = kpar # parameters of s * exp (-d**2/t**2)
        self.spar = spar # diagonal parameter of sigma (or full Sigma)
        self.gp_update()
    
    def gp_update(self):
        self.mgp.y = np.ravel(self.obs - \
                self.g_predict(self.x, par=self.gpar))
        self.mgp.setpar(self.kpar, self.spar, cond=True)
    
    def g_predict(self, x, par=None):
        ptmp = self.gpar if par is None else par
        return self.gmod(x, ptmp) if ptmp is not None else \
              np.zeros(x.shape[0])
    
    def mean(self, x):
        return self.g_predict(x) + self.mgp.m_predict(x)
    
    def cov(self, x):
        return self.mgp.cov_predict(x, noise=False) + \
                np.eye(x.shape[0])*self.spar

    def loglike(self):
        N = self.obs.shape[0]
        d = self.obs.shape[1]
        try:
            L = np.linalg.cholesky(self.cov(self.x))
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is not positive semi-definite "+
                             "(cannot perform Cholesky decomposition).")
        log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
        diff = self.mean(self.x) - self.obs
        mahalanobis_term = np.zeros(N)
        for i in range(N):
            y = np.linalg.solve(L, diff[i, :])
            mahalanobis_term[i] = np.sum(y**2)
        log_likelihood = - (N * d / 2) * np.log(2 * np.pi) \
                    - (N / 2) * log_det_sigma \
                    - (1 / 2) * np.sum(mahalanobis_term)
        return log_likelihood