"""
Test the proposal-history and snapshot features added to pyBI2.

  - progress_every=1000 should print live summaries during the run.
  - snapshot_every=500 should populate trace.snapshot_history.
  - Both samplers should record trace.proposal_history at each tuning event.
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

# ---------------- shared setup ----------------
np.random.seed(0)
N = 30
X = np.linspace(0, 5, N)[:, None]
beta_true = np.array([1.5, -0.4, 0.2])
def model(x, b):
    return b[0] + b[1] * x[:, 0] + b[2] * x[:, 0] ** 2

# AR(1) Sigma
rho, s2 = 0.6, 0.15 ** 2
idx = np.arange(N)
Sigma_AR1 = s2 * rho ** np.abs(idx[:, None] - idx[None, :])
L_AR1 = np.linalg.cholesky(Sigma_AR1)
y = model(X, beta_true) + L_AR1 @ np.random.randn(N)


#%%
# ===================================================================
# 1. MetropolisBlock with progress + snapshots, beta + sigma inferred
# ===================================================================
print("=" * 70)
print("MetropolisBlock with progress_every=1000, snapshot_every=500")
print("=" * 70)

problem = bi.InferenceProblem(
    likelihood=bi.InferredDiagLike(X, y, model, sigma_init=0.5))
problem.add_block(
    name="beta",
    priors=[bi.Uniform(-5, 5)] * 3,
    proposal=bi.BlockGaussianRW(dim=3, scale=0.1))
problem.add_extra_block(
    name="sigma",
    prior=bi.HalfNormal(0.5),
    proposal=bi.LogScaleRW(scale=0.5))

sampler = bi.MetropolisBlock(
    problem,
    n_iter=10000, n_burn=4000,
    adaptive=True, seed=1,
    progress_every=1000,
    snapshot_every=500,
)
trace_block = sampler.run()

print()
print(f"Recorded {len(trace_block.snapshot_history)} snapshots.")
print(f"Recorded {len(trace_block.proposal_history)} proposal-history entries.")
print(f"Final acceptance: {trace_block.acc_rates}")

#%%

# ===================================================================
# 2. Inspect proposal evolution
# ===================================================================
print()
print("=" * 70)
print("Proposal evolution during burn-in (block sampler)")
print("=" * 70)

# Tabulate the scale evolution for sigma (scalar) and the block scale
# factor for beta. Skip the per-iteration L matrices for legibility.
print(f"\n{'iteration':>10}  {'sigma_scale':>12}  {'beta_scale_factor':>18}  "
      f"{'beta_coord_std':>30}")
for entry in trace_block.proposal_history:
    it = entry["iteration"]
    sigma_state = entry["blocks"]["sigma"]
    beta_state = entry["blocks"]["beta"]
    coord_std_str = "[" + ", ".join(f"{x:.4f}" for x in beta_state["coord_std"]) + "]"
    print(f"{it:>10d}  {sigma_state['scale']:>12.5f}  "
          f"{beta_state['scale_factor']:>18.4f}  {coord_std_str:>30}")

print()
print("Final tuned proposals:")
final = trace_block.final_proposals()
for name, snap in final["blocks"].items():
    print(f"  {name}: {snap}")


# ===================================================================
# 3. Plot the snapshot evolution
# ===================================================================
print()
print("Plotting snapshot evolution to /home/claude/evolution_block.png ...")
fig, axs = trace_block.plot_evolution()
fig.suptitle("MetropolisBlock chain evolution (every 500 iters)", y=1.02)
plt.savefig("./evolution_block.png", dpi=80, bbox_inches="tight")
plt.close("all")


# ===================================================================
# 4. Same with MetropolisGibbs to verify it works there too
# ===================================================================
print()
print("=" * 70)
print("MetropolisGibbs with progress_every=2000, snapshot_every=1000")
print("=" * 70)

problem2 = bi.InferenceProblem(
    likelihood=bi.InferredDiagLike(X, y, model, sigma_init=0.5))
problem2.add_block(
    name="beta",
    priors=[bi.Uniform(-5, 5)] * 3,
    proposal=bi.BlockGaussianRW(dim=3, scale=0.1))
problem2.add_extra_block(
    name="sigma",
    prior=bi.HalfNormal(0.5),
    proposal=bi.LogScaleRW(scale=0.5))

sampler2 = bi.MetropolisGibbs(
    problem2,
    n_iter=10000, n_burn=4000,
    adaptive=True, seed=2,
    progress_every=2000,
    snapshot_every=1000,
)
trace_mwg = sampler2.run()

print()
print(f"Recorded {len(trace_mwg.snapshot_history)} snapshots.")
print(f"Final acceptance: {trace_mwg.acc_rates}")

print()
print("MwG-specific: per-coordinate scales for the beta block over time")
print(f"{'iteration':>10}  {'mwg_coord_scales':>40}")
for entry in trace_mwg.proposal_history:
    it = entry["iteration"]
    beta_state = entry["blocks"]["beta"]
    cs = beta_state.get("mwg_coord_scales")
    cs_str = "[" + ", ".join(f"{x:.4f}" for x in cs) + "]"
    print(f"{it:>10d}  {cs_str:>40}")

print()
print("Plotting MwG snapshot evolution to /home/claude/evolution_mwg.png ...")
fig, axs = trace_mwg.plot_evolution()
fig.suptitle("MetropolisGibbs chain evolution (every 1000 iters)", y=1.02)
plt.savefig("./evolution_mwg.png", dpi=80, bbox_inches="tight")
plt.close("all")


# ===================================================================
# 5. Sanity check: snapshots and proposal-history off by default
# ===================================================================
print()
print("=" * 70)
print("Sanity: omitting both flags -> empty histories, no extra prints")
print("=" * 70)

sampler3 = bi.MetropolisBlock(problem, n_iter=2000, n_burn=500,
                              adaptive=True, seed=3)
trace_quiet = sampler3.run()
assert trace_quiet.snapshot_history == []
print(f"snapshot_history empty: {trace_quiet.snapshot_history == []}")
print(f"proposal_history still recorded "
      f"(at every tuning event): {len(trace_quiet.proposal_history)} entries")
print(f"final tuned proposals are still there: "
      f"{trace_quiet.final_proposals()['blocks']}")

print()
print("=" * 70)
print("ALL MONITORING TESTS PASSED.")
print("=" * 70)
