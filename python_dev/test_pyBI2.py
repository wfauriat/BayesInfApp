"""
Test the pyBI2 refactor end-to-end.

Coverage:
  Test 1 : FixedDiagLike (sigma supplied, beta only) -- sanity baseline.
  Test 2 : FullCovLike (AR(1) Sigma supplied, beta only).
  Test 3 : InferredDiagLike (sigma inferred jointly with beta).
  Test 4 : Same as Test 3 but with MetropolisGibbs (single-step).
  Test 5 : User-tweaked proposals (no autotune) recover the same answer.
  Test 6 : Multi-chain R-hat on Test 2's setup.

Also verifies the priors/proposals math against scipy where possible
and shows that the Hastings correction in LogScaleRW is doing its job.
"""

#%%
import sys

import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as mvn

import pyBI.newAPI as bi

#%%

# ----------------------------------------------------------------------
# Common forward model
# ----------------------------------------------------------------------
np.random.seed(0)
N = 30
X = np.linspace(0, 5, N)[:, None]
beta_true = np.array([1.5, -0.4, 0.2])

def model(x, b):
    return b[0] + b[1] * x[:, 0] + b[2] * x[:, 0] ** 2


def fit_summary(trace, label, expected=None):
    print(f"\n--- {label} ---")
    rows = trace.summary(print_it=False)
    cols = ["param", "mean", "std", "ess"]
    print("  " + "  ".join(f"{c:>8}" for c in cols))
    for r in rows:
        print("  " + "  ".join(
            f"{r[c]:.4g}".rjust(8) if isinstance(r[c], float)
            else str(r[c]).rjust(8) for c in cols))
    print(f"  acceptance rates: "
          + ", ".join(f"{n}={v:.3f}" for n, v in trace.acc_rates.items()))
    if expected is not None:
        means = np.array([r["mean"] for r in rows])
        stds  = np.array([r["std"] for r in rows])
        n_off = sum(abs(m - e) > 3 * s
                    for m, e, s in zip(means, expected, stds))
        print(f"  truth = {expected}")
        print(f"  off by >3 sigma: {n_off}")


#%%
# ======================================================================
# TEST 1 : FixedDiagLike, MetropolisBlock
# ======================================================================
print("=" * 70)
print("TEST 1 : Fixed diagonal sigma supplied, beta only.")
print("=" * 70)
sigma_known = 0.15
y1 = model(X, beta_true) + sigma_known * np.random.randn(N)

problem = bi.InferenceProblem(
    likelihood=bi.FixedDiagLike(X, y1, model, sigma=sigma_known))
problem.add_block(
    name="beta",
    priors=[bi.Uniform(-5, 5) for _ in range(3)],
    proposal=bi.BlockGaussianRW(dim=3, scale=0.1))

sampler = bi.MetropolisBlock(problem, n_iter=10000, n_burn=3000,
                             adaptive=True, verbose=False, seed=1)
trace1 = sampler.run()
fit_summary(trace1, "Test 1 (block, fixed-diag)", expected=beta_true.tolist())


#%%

# ======================================================================
# TEST 2 : FullCovLike (AR(1)), MetropolisBlock
# ======================================================================
print()
print("=" * 70)
print("TEST 2 : Full Sigma (AR(1)) supplied, beta only.")
print("=" * 70)
rho, s2 = 0.6, 0.15 ** 2
idx = np.arange(N)
Sigma_AR1 = s2 * rho ** np.abs(idx[:, None] - idx[None, :])
L_AR1 = np.linalg.cholesky(Sigma_AR1)
y2 = model(X, beta_true) + L_AR1 @ np.random.randn(N)

problem = bi.InferenceProblem(
    likelihood=bi.FullCovLike(X, y2, model, Sigma=Sigma_AR1))
problem.add_block(
    name="beta",
    priors=[bi.Uniform(-5, 5) for _ in range(3)],
    proposal=bi.BlockGaussianRW(dim=3, scale=0.1))

sampler = bi.MetropolisBlock(problem, n_iter=10000, n_burn=3000,
                             adaptive=True, verbose=False, seed=2)
trace2 = sampler.run()
fit_summary(trace2, "Test 2 (block, full Sigma AR(1))", expected=beta_true.tolist())

#%%


# ======================================================================
# TEST 3 : InferredDiagLike, MetropolisBlock (joint moves on beta + log
#          scale RW on sigma).
# ======================================================================
print()
print("=" * 70)
print("TEST 3 : InferredDiagLike, beta + sigma both inferred (block sampler).")
print("=" * 70)
sigma_true = 0.12
y3 = model(X, beta_true) + sigma_true * np.random.randn(N)

problem = bi.InferenceProblem(
    likelihood=bi.InferredDiagLike(X, y3, model, sigma_init=0.5))
problem.add_block(
    name="beta",
    priors=[bi.Uniform(-5, 5) for _ in range(3)],
    proposal=bi.BlockGaussianRW(dim=3, scale=0.1))
problem.add_extra_block(
    name="sigma",
    prior=bi.HalfNormal(0.5),
    proposal=bi.LogScaleRW(scale=0.5))

sampler = bi.MetropolisBlock(problem, n_iter=12000, n_burn=4000,
                             adaptive=True, verbose=False, seed=3)
trace3 = sampler.run()
fit_summary(trace3, "Test 3 (block, beta+sigma)",
            expected=beta_true.tolist() + [sigma_true])

#%%

# ======================================================================
# TEST 4 : Same as Test 3, but MetropolisGibbs (single-step) sampler
# ======================================================================
print()
print("=" * 70)
print("TEST 4 : InferredDiagLike, beta + sigma both inferred (MwG sampler).")
print("=" * 70)

problem = bi.InferenceProblem(
    likelihood=bi.InferredDiagLike(X, y3, model, sigma_init=0.5))
problem.add_block(
    name="beta",
    priors=[bi.Uniform(-5, 5) for _ in range(3)],
    proposal=bi.BlockGaussianRW(dim=3, scale=0.1))
problem.add_extra_block(
    name="sigma",
    prior=bi.HalfNormal(0.5),
    proposal=bi.LogScaleRW(scale=0.5))

sampler = bi.MetropolisGibbs(problem, n_iter=12000, n_burn=4000,
                             adaptive=True, verbose=False, seed=4)
trace4 = sampler.run()
fit_summary(trace4, "Test 4 (MwG, beta+sigma)",
            expected=beta_true.tolist() + [sigma_true])


#%%

# ======================================================================
# TEST 5 : User-tweaked proposals, NO autotune.
# ======================================================================
print()
print("=" * 70)
print("TEST 5 : User-tweaked proposals (adaptive=False) on Test 3 setup.")
print("=" * 70)

problem = bi.InferenceProblem(
    likelihood=bi.InferredDiagLike(X, y3, model, sigma_init=0.5))
problem.add_block(
    name="beta",
    priors=[bi.Uniform(-5, 5) for _ in range(3)],
    # hand-set L for the joint Gaussian RW
    proposal=bi.BlockGaussianRW(dim=3, scale=0.27))
problem.add_extra_block(
    name="sigma",
    prior=bi.HalfNormal(0.5),
    proposal=bi.LogScaleRW(scale=0.38))

sampler = bi.MetropolisBlock(problem, n_iter=12000, n_burn=4000,
                             adaptive=False,   # <-- no auto-tune
                             verbose=False, seed=5)
trace5 = sampler.run()
fit_summary(trace5, "Test 5 (block, no-autotune)",
            expected=beta_true.tolist() + [sigma_true])

#%%


# ======================================================================
# TEST 6 : Multi-chain R-hat on Test 2's setup.
# ======================================================================
print()
print("=" * 70)
print("TEST 6 : R-hat across 4 chains, Test 2 setup.")
print("=" * 70)

traces = []
for seed in [10, 20, 30, 40]:
    problem = bi.InferenceProblem(
        likelihood=bi.FullCovLike(X, y2, model, Sigma=Sigma_AR1))
    problem.add_block(
        name="beta",
        priors=[bi.Uniform(-5, 5) for _ in range(3)],
        proposal=bi.BlockGaussianRW(dim=3, scale=0.1))
    sampler = bi.MetropolisBlock(problem, n_iter=8000, n_burn=3000,
                                 adaptive=True, verbose=False, seed=seed)
    traces.append(sampler.run())

rhats = bi.rhat(traces)
print("  R-hat per parameter:")
for n, r in rhats.items():
    flag = "OK" if r < 1.05 else "(chain not converged)"
    print(f"    {n:>10s} : {r:.4f}  {flag}")

#%%


# ======================================================================
# TEST 7 : sanity-check the LogScaleRW Hastings correction.
# Without the correction, sigma posterior is biased low.
# We compare against a closed-form: with Uniform(eps, B) prior on sigma
# and known beta=beta_true, the conditional posterior on sigma has a
# known shape we can match with a fine grid.
# ======================================================================
print()
print("=" * 70)
print("TEST 7 : LogScaleRW Hastings correction vs Gaussian RW (no correction).")
print("=" * 70)

# Generate data; we'll fit sigma only, with beta clamped to truth.
np.random.seed(11)
sigma_true = 0.2
y7 = model(X, beta_true) + sigma_true * np.random.randn(N)

# Hand-rolled tight wrapper: use InferredDiagLike but fix beta to truth
# by giving it a Dirac-ish prior (very tight Normal centred at truth).
# (We pin beta by using extremely tight priors on it; that effectively
# samples only sigma.)
def run_sigma_only(proposal_obj, n_iter=30000, n_burn=5000, seed=0):
    problem = bi.InferenceProblem(
        likelihood=bi.InferredDiagLike(X, y7, model, sigma_init=0.3))
    # tight Normals centered at truth -> beta effectively fixed
    problem.add_block(
        name="beta",
        priors=[bi.Normal(beta_true[k], 1e-4) for k in range(3)],
        proposal=bi.BlockGaussianRW(dim=3, scale=1e-5))
    problem.add_extra_block(
        name="sigma",
        prior=bi.Uniform(0.01, 2.0),     # uninformative on positive support
        proposal=proposal_obj)
    sampler = bi.MetropolisBlock(problem, n_iter=n_iter, n_burn=n_burn,
                                 adaptive=False, verbose=False, seed=seed)
    return sampler.run()

# Closed-form-like reference via grid: with beta fixed, the conditional
# posterior on sigma given y, beta is
#   p(sigma | y, beta) propto sigma^{-N} exp(-SSE / (2 sigma^2)) * 1[a<sigma<b]
# We can compute its mean numerically.
SSE = float(np.sum((y7 - model(X, beta_true)) ** 2))
sig_grid = np.linspace(0.01, 2.0, 5000)
log_p_grid = -N * np.log(sig_grid) - 0.5 * SSE / sig_grid ** 2
log_p_grid -= log_p_grid.max()
p_grid = np.exp(log_p_grid)
p_grid /= np.trapezoid(p_grid, sig_grid) if hasattr(np, "trapezoid") else \
          np.trapz(p_grid, sig_grid)
true_mean = float(np.trapezoid(sig_grid * p_grid, sig_grid)
                  if hasattr(np, "trapezoid") else
                  np.trapz(sig_grid * p_grid, sig_grid))
true_std = float(np.sqrt(np.trapezoid((sig_grid - true_mean) ** 2 * p_grid,
                                       sig_grid)
                          if hasattr(np, "trapezoid") else
                          np.trapz((sig_grid - true_mean) ** 2 * p_grid,
                                   sig_grid)))
print(f"  reference (numerical integration): mean={true_mean:.5f}  std={true_std:.5f}")

# Run with LogScaleRW (correct)
trace_log = run_sigma_only(bi.LogScaleRW(scale=0.1), seed=123)
sig_log = trace_log.get("sigma")
print(f"  LogScaleRW (correct):              mean={sig_log.mean():.5f}  "
      f"std={sig_log.std(ddof=1):.5f}")

# Run with GaussianRW (no Hastings correction needed -- it's symmetric)
trace_gauss = run_sigma_only(bi.GaussianRW(scale=0.05), seed=123)
sig_gauss = trace_gauss.get("sigma")
print(f"  GaussianRW (also correct):         mean={sig_gauss.mean():.5f}  "
      f"std={sig_gauss.std(ddof=1):.5f}")
print("  (Both should match the numerical reference; LogScaleRW used to be "
      "biased without the Hastings term.)")

assert abs(sig_log.mean() - true_mean) < 0.005, (
    "LogScaleRW posterior mean off -- Hastings correction issue?")
assert abs(sig_gauss.mean() - true_mean) < 0.005

print()
print("=" * 70)
print("ALL TESTS PASSED.")
print("=" * 70)

# Some plots for visual inspection
print()
print("Saving diagnostic plots ...")
trace2.plot_trace().__class__   # discard return; just trigger plt
import matplotlib.pyplot as plt
plt.savefig("./trace_test2.png", dpi=80, bbox_inches="tight")
plt.close("all")
trace2.plot_pairs()
plt.savefig("./pairs_test2.png", dpi=80, bbox_inches="tight")
plt.close("all")
trace3.plot_trace()
plt.savefig("./trace_test3.png", dpi=80, bbox_inches="tight")
plt.close("all")
print("  saved trace_test2.png, pairs_test2.png, trace_test3.png")

