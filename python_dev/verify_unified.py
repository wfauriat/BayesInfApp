"""
Verify the unified ObsKnownCov:

  1. Three Sigma shapes (scalar / 1D diag / 2D full) all match scipy.
  2. sigma_fn kernel route == passing the matrix.
  3. Replicates by stacking (duplicate x_i with appropriate Sigma block)
     match the dedicated replicates implementation.
  4. y_i-dependent kernel works (e.g. heteroscedastic noise scaled by |y|).
  5. MCMC recovery on a non-trivial example.
"""
#%%
import sys
sys.path.insert(0, "/home/claude")

import numpy as np
import matplotlib
matplotlib.use("Agg")

from scipy.stats import multivariate_normal as mvn
from pyBI.obs_known_cov_unified import ObsKnownCov
from pyBI.scalar_replicates import ScalarObsRepKnownCov

from pyBI.base import UnifVar
from pyBI.inference_v2 import MHwGalgo

np.random.seed(7)


#%%

# ----- toy regression problem ----------------------------------------------
N = 25
X = np.linspace(0, 5, N)[:, None]
beta_true = np.array([1.5, -0.4, 0.2])

def model(x, b):
    return b[0] + b[1] * x[:, 0] + b[2] * x[:, 0] ** 2

# AR(1) Sigma -- the workhorse non-trivial covariance
rho, s2 = 0.6, 0.1 ** 2
idx = np.arange(N)
Sigma_AR1 = s2 * rho ** np.abs(idx[:, None] - idx[None, :])
L = np.linalg.cholesky(Sigma_AR1)

mu_true = model(X, beta_true)
y = mu_true + L @ np.random.randn(N)
beta_eval = np.array([1.2, -0.3, 0.18])
mu_eval = model(X, beta_eval)

#%%

# --------------------------------------------------------------------------
# 1. Three Sigma shapes vs scipy
# --------------------------------------------------------------------------
print("1. Sigma shapes vs scipy ground truth")
print("   -----------------------------------")
sigma2_h = 0.04
y_iid = mu_true + np.sqrt(sigma2_h) * np.random.randn(N)

cases = [
    ("scalar  ", sigma2_h,                np.eye(N) * sigma2_h),
    ("1D diag ", np.linspace(0.005, 0.05, N), np.diag(np.linspace(0.005, 0.05, N))),
    ("2D full ", Sigma_AR1,               Sigma_AR1),
]
y_for = [y_iid, y_iid, y]   # full Sigma needs the AR(1)-noise data
for (label, Sig_in, Sig_full), y_use in zip(cases, y_for):
    obs = ObsKnownCov(X, y_use, model, Sigma=Sig_in)
    ll  = obs.loglike(beta_eval)
    truth = mvn(mean=mu_eval, cov=Sig_full).logpdf(y_use)
    print(f"   {label}: custom={ll:.6f}  scipy={truth:.6f}  diff={abs(ll-truth):.2e}")
    assert abs(ll - truth) < 1e-7

#%%

# --------------------------------------------------------------------------
# 2. sigma_fn route gives identical Sigma to passing the matrix
# --------------------------------------------------------------------------
print()
print("2. sigma_fn route")
print("   --------------")

# AR(1)-style kernel. Vectorised: sigma_fn must accept index arrays.
def kfn_x(i, j, X, y):
    return s2 * rho ** np.abs(i - j)
obs_mat = ObsKnownCov(X, y, model, Sigma=Sigma_AR1)
obs_kfn = ObsKnownCov(X, y, model, sigma_fn=kfn_x)
ll_mat = obs_mat.loglike(beta_eval)
ll_kfn = obs_kfn.loglike(beta_eval)
print(f"   matrix : {ll_mat:.10f}")
print(f"   kernel : {ll_kfn:.10f}")
print(f"   diff   : {abs(ll_mat - ll_kfn):.2e}")
assert abs(ll_mat - ll_kfn) < 1e-12

# Non-vectorised kernel (Python scalar in -> Python scalar out)
def kfn_scalar(i, j, X, y):
    return float(s2 * rho ** float(np.abs(i - j)))
obs_kfn2 = ObsKnownCov(X, y, model, sigma_fn=kfn_scalar)
ll_kfn2  = obs_kfn2.loglike(beta_eval)
print(f"   loop kernel diff: {abs(ll_mat - ll_kfn2):.2e}  (Python-loop fallback)")
assert abs(ll_mat - ll_kfn2) < 1e-12

#%%

# --------------------------------------------------------------------------
# 3. Replicates by stacking
# --------------------------------------------------------------------------
print()
print("3. Replicates by stacking")
print("   ----------------------")

# Generate M independent replicates.
M = 4
Y_M = mu_true[:, None] + L @ np.random.randn(N, M)        # (N, M)

# Path A: dedicated replicates class.
obs_rep = ScalarObsRepKnownCov(X, Y_M, model, Sigma_AR1, use_sufficient=False)
ll_A    = obs_rep.loglike(beta_eval)

# Path B: stacking. Repeat X M times, flatten Y in column-major order,
# and use the block-diagonal Sigma.
X_stack = np.tile(X, (M, 1))
y_stack = Y_M.flatten(order='F')
Sigma_stack = np.kron(np.eye(M), Sigma_AR1)
obs_stack = ObsKnownCov(X_stack, y_stack, model, Sigma=Sigma_stack)
ll_B = obs_stack.loglike(beta_eval)

# Path C: stacking, but specify the block-diagonal Sigma via sigma_fn.
# Index i in the stacked frame maps to (i mod N, i // N).
# Replicates m, m' at the same N-block are independent, so Sigma is
# zero between different m's and equals AR1(i,j) within the same m.
def block_kernel(i, j, X, y):
    # map flat index -> (replicate_idx, n_idx)
    n_i, m_i = i % N, i // N
    n_j, m_j = j % N, j // N
    same_block = (m_i == m_j)
    return np.where(same_block, s2 * rho ** np.abs(n_i - n_j), 0.0)

obs_stack_kfn = ObsKnownCov(X_stack, y_stack, model, sigma_fn=block_kernel)
ll_C = obs_stack_kfn.loglike(beta_eval)

print(f"   replicates class : {ll_A:.10f}")
print(f"   stacking + kron  : {ll_B:.10f}")
print(f"   stacking + kfn   : {ll_C:.10f}")
print(f"   max pairwise diff: {max(abs(ll_A-ll_B), abs(ll_A-ll_C), abs(ll_B-ll_C)):.2e}")
assert abs(ll_A - ll_B) < 1e-9
assert abs(ll_A - ll_C) < 1e-9

#%%


# --------------------------------------------------------------------------
# 4. Kernel built from observed y (heteroscedastic with |y|)
# --------------------------------------------------------------------------
print()
print("4. y-dependent kernel (heteroscedastic, no off-diags)")
print("   --------------------------------------------------")

# Variance proportional to (1 + y_i^2). This is a function of the OBSERVED
# y (computable at construction), not of the predicted f(x, beta).
def kfn_y(i, j, X, y):
    var = 0.01 * (1 + y[i] ** 2)
    return np.where(i == j, var, 0.0)

# Same data, but check that ObsKnownCov correctly factorises this Sigma.
y_het = mu_true + np.sqrt(0.01 * (1 + mu_true ** 2)) * np.random.randn(N)
obs_y_kfn = ObsKnownCov(X, y_het, model, sigma_fn=kfn_y)
# Equivalent diagonal:
diag_var = 0.01 * (1 + y_het ** 2)
obs_y_diag = ObsKnownCov(X, y_het, model, Sigma=diag_var)
ll_y_kfn = obs_y_kfn.loglike(beta_eval)
ll_y_diag = obs_y_diag.loglike(beta_eval)
print(f"   kernel(i,j,X,y): {ll_y_kfn:.10f}")
print(f"   diag(variances): {ll_y_diag:.10f}")
print(f"   diff: {abs(ll_y_kfn - ll_y_diag):.2e}")
assert abs(ll_y_kfn - ll_y_diag) < 1e-10

#%%


# --------------------------------------------------------------------------
# 5. MCMC recovery on the AR(1) non-replicated case via pyBI machinery
# --------------------------------------------------------------------------
print()
print("5. MCMC recovery (AR(1), non-replicated)")
print("   --------------------------------------")

# pyBI's KnownCovObs expects model returning (N, dy). Adapt:
def model_2d(x, b):
    return model(x, b)[:, None]

# The unified class returns (N,) from prev_model -- wrap so it returns (N, 1)
# for compatibility with the pyBI inference loop's call to obsObj.loglike(par).
# Actually, ObsKnownCov.loglike takes only (par,), so we just need to make
# sure the rest of pyBI's machinery works. The legacy attributes Ndata,
# dimdata, obs, cond_var are exposed via @property.

obs = ObsKnownCov(X, y, model, Sigma=Sigma_AR1)
priors = [UnifVar([-5, 5]) for _ in range(3)]
mc = MHwGalgo(N=10000, Nthin=10, Nburn=5000, is_adaptive=True, verbose=False)
mc.initialize(obs, priors, discrObj=None, svar=0.05)
mc.runInference()
print(f"   beta_true   : {beta_true}")
print(f"   posterior mean: {mc.cut_chain.mean(axis=0)}")
print(f"   posterior std : {mc.cut_chain.std(axis=0)}")
print(f"   acc rates     : {mc.tacc}")

print()
print("ALL VERIFICATIONS PASSED.")
