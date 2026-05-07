"""
Proposals. Each proposal knows:
  - how to draw a candidate given the current value (propose)
  - what Hastings log-ratio correction is needed (returned by propose)
  - how to adapt its own scale (adapt)

Proposals are state. Each one carries its own internal scale, so a
sampler can ask "give me a candidate" and "adapt yourself based on
recent acceptance rate" without knowing the proposal's internal
mechanism.

The propose() method returns (candidate, log_q_ratio) where
log_q_ratio = log q(current | candidate) - log q(candidate | current).
For symmetric proposals this is zero; for log-scale RW it's nonzero.
The Metropolis ratio uses:
    log alpha = log_post(cand) - log_post(curr) + log_q_ratio
"""

import numpy as np


# ----------------------------------------------------------------------
#  Adaptive scale ladder shared by all scalar RW proposals.
# ----------------------------------------------------------------------
def _tune_step(scale, acc_rate, target=0.44):
    """
    Multiplicative scale adjustment based on acceptance rate.
    Default target 0.44 (RGG optimum for 1D Metropolis); the joint
    Gaussian proposal overrides this with target=0.234.

    Single-pass cascade: smaller-than-thresholds checked first, biggest
    deviations get biggest corrections. Always returns a strictly
    positive scale.
    """
    if acc_rate < 0.001:    return scale * 0.1
    if acc_rate < 0.05:     return scale * 0.5
    if acc_rate < target - 0.10:  return scale * 0.9
    if acc_rate > 0.95:     return scale * 10.0
    if acc_rate > 0.75:     return scale * 2.0
    if acc_rate > target + 0.10:  return scale * 1.1
    return scale  # within tolerance, leave alone


# ======================================================================
#  Scalar proposals
# ======================================================================

class Proposal:
    """Abstract base for scalar proposals."""
    def propose(self, x_current):
        raise NotImplementedError
    def adapt(self, acc_rate):
        raise NotImplementedError
    def state(self):
        """Return a dict describing current internal state (for logging)."""
        raise NotImplementedError


class GaussianRW(Proposal):
    """Symmetric Gaussian random walk: x' = x + s * N(0, 1)."""
    def __init__(self, scale=1.0, target_acc=0.44):
        self.scale = float(scale)
        self.target_acc = target_acc

    def propose(self, x):
        return float(x + self.scale * np.random.randn()), 0.0

    def adapt(self, acc_rate):
        self.scale = _tune_step(self.scale, acc_rate, self.target_acc)

    def state(self):
        return {"type": "GaussianRW", "scale": self.scale}

    def __repr__(self):
        return f"GaussianRW(scale={self.scale:.4g})"


class LogScaleRW(Proposal):
    """
    Multiplicative random walk on positive support:
        log x' = log x + s * N(0, 1),  i.e.  x' = x * exp(s * Z).

    The proposal is asymmetric in x-space: q(x'|x) and q(x|x') differ.
    The Hastings correction is log(x'/x), which we return alongside
    the candidate.
    """
    def __init__(self, scale=0.5, target_acc=0.44):
        self.scale = float(scale)
        self.target_acc = target_acc

    def propose(self, x):
        if x <= 0:
            # caller's job to keep x positive; refuse silently with a
            # log-ratio that will get rejected when evaluated.
            return float(x), 0.0
        x_new = float(x * np.exp(self.scale * np.random.randn()))
        log_q_ratio = np.log(x_new / x)
        return x_new, log_q_ratio

    def adapt(self, acc_rate):
        self.scale = _tune_step(self.scale, acc_rate, self.target_acc)

    def state(self):
        return {"type": "LogScaleRW", "scale": self.scale}

    def __repr__(self):
        return f"LogScaleRW(scale={self.scale:.4g})"


# ======================================================================
#  Block (multivariate) proposals
# ======================================================================

class BlockGaussianRW(Proposal):
    """
    Joint Gaussian random walk on a d-dimensional vector:
        x' = x + L Z,  Z ~ N(0, I_d), L lower triangular.

    Stores L (the Cholesky factor of the proposal covariance). Adapts L
    from the empirical covariance of recent samples using the
    Roberts-Gelman-Gilks scaling (2.38^2 / d). Symmetric, so log_q_ratio = 0.
    """
    RGG_FACTOR = 2.38 ** 2

    def __init__(self, dim, scale=0.1, target_acc=0.234, jitter=1e-8):
        self.dim = int(dim)
        # Initial L: scaled identity
        self.L = scale * np.eye(self.dim)
        self.target_acc = target_acc
        self.jitter = jitter
        # accumulator for empirical covariance during adaptation
        self._scale_factor = 1.0   # multiplicative tuning on top of cov

    def propose(self, x):
        x = np.asarray(x, dtype=float)
        z = np.random.randn(self.dim)
        return x + self._scale_factor * (self.L @ z), 0.0

    def adapt_from_samples(self, samples):
        """
        Recompute L from the empirical covariance of `samples` (M, d).
        Uses RGG scaling 2.38^2 / d. This is the dominant adaptation
        mechanism for joint moves -- shaping the proposal to match
        posterior correlations.
        """
        samples = np.asarray(samples, dtype=float)
        if samples.shape[0] < self.dim + 2:
            # not enough samples to estimate covariance reliably
            return
        cov = np.cov(samples.T)
        if self.dim == 1:
            cov = np.atleast_2d(cov)
        scaled = cov * self.RGG_FACTOR / self.dim + self.jitter * np.eye(self.dim)
        try:
            self.L = np.linalg.cholesky(scaled)
        except np.linalg.LinAlgError:
            # bump jitter and retry
            scaled = cov * self.RGG_FACTOR / self.dim \
                     + max(self.jitter * 100, 1e-6) * np.eye(self.dim)
            self.L = np.linalg.cholesky(scaled)

    def adapt(self, acc_rate):
        """
        Cheap multiplicative tweak applied AFTER adapt_from_samples.
        Useful when the empirical-cov shape is right but overall scale
        is off (e.g. early in burn-in). This adjusts the scalar
        _scale_factor, leaving the proposal shape from cov() intact.
        """
        self._scale_factor = _tune_step(self._scale_factor, acc_rate,
                                        self.target_acc)

    def state(self):
        # Effective per-coord std = scale_factor * sqrt(diag(L L^T))
        diag_var = np.sum(self.L * self.L, axis=1)
        coord_std = self._scale_factor * np.sqrt(np.maximum(diag_var, 0.0))
        return {
            "type": "BlockGaussianRW",
            "scale_factor": self._scale_factor,
            "coord_std": coord_std.copy(),
            "L": self.L.copy(),
        }

    def __repr__(self):
        return f"BlockGaussianRW(dim={self.dim}, scale_factor={self._scale_factor:.3g})"