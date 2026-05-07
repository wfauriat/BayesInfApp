"""
The InferenceProblem ties together blocks (each block = one or more
parameters with a shared prior+proposal) and a likelihood.

A "block" is a unit of joint update. Two flavors:
  - SCALAR block  : a single 1D parameter (e.g. sigma) with a scalar
                    Prior and a scalar Proposal (GaussianRW or LogScaleRW).
  - VECTOR block  : a multi-D parameter (e.g. beta) with a list of
                    scalar Priors (one per component) and a single
                    BlockGaussianRW for joint moves. (The single-step
                    sampler will use scalar moves derived from the
                    BlockGaussianRW's diagonal; see samplers.py.)

The likelihood may declare "extras" (parameters it owns that get inferred,
like sigma in InferredDiagLike). Those are added as scalar blocks
automatically, with priors and proposals that the user passes in via
add_extra_block(name, prior, proposal).

The sampler does not see inside the problem. It calls:
   - .blocks                 -> list of block objects
   - .logpost(state)         -> evaluate full log-posterior at a state dict
   - .draw_initial()         -> dict of initial values from priors
"""

import numpy as np

from .priors import Prior
from .proposals import Proposal, BlockGaussianRW


# ----------------------------------------------------------------------
class _Block:
    """
    Unifying interface for scalar and vector blocks.

    Attributes:
        name      : str, e.g. "beta" or "sigma"
        dim       : int, dimension of the block (1 for scalar, d for vector)
        priors    : list[Prior] of length dim
        proposal  : Proposal (scalar) for dim=1, BlockGaussianRW for dim>1
    """
    def __init__(self, name, priors, proposal):
        if isinstance(priors, Prior):
            priors = [priors]
        if not all(isinstance(p, Prior) for p in priors):
            raise TypeError(f"block {name!r}: priors must be Prior instances.")
        if not isinstance(proposal, Proposal):
            raise TypeError(f"block {name!r}: proposal must be a Proposal.")
        self.name = name
        self.priors = list(priors)
        self.dim = len(self.priors)
        self.proposal = proposal
        if self.dim > 1 and not isinstance(proposal, BlockGaussianRW):
            raise TypeError(
                f"block {name!r}: dim>1 requires a BlockGaussianRW proposal.")
        if self.dim > 1 and proposal.dim != self.dim:
            raise ValueError(
                f"block {name!r}: proposal.dim={proposal.dim} != block dim {self.dim}.")

    def logprior(self, value):
        """value: scalar (dim=1) or 1D array (dim>1). Returns sum logpdf."""
        if self.dim == 1:
            v = float(value) if np.ndim(value) == 0 else float(value[0])
            return self.priors[0].logpdf(v)
        v = np.asarray(value, dtype=float)
        s = 0.0
        for k in range(self.dim):
            s += self.priors[k].logpdf(float(v[k]))
            if not np.isfinite(s):
                return -np.inf
        return s

    def draw_initial(self):
        if self.dim == 1:
            return self.priors[0].draw()
        return np.array([p.draw() for p in self.priors])


# ----------------------------------------------------------------------
class InferenceProblem:
    """
    Holds blocks and a likelihood. Computes log-posterior at a given
    state dict.

    Conceptually:
        log p(theta | data) ∝ log p(data | theta) + sum_b log p(theta_b)

    where theta is the dict {block_name: block_value}. The likelihood
    takes only beta-like parameters by call signature loglike(beta_value),
    and any "extras" (e.g. sigma) are pushed into the likelihood via
    set_extra() before the likelihood is evaluated.

    The single "main" block is the one passed to loglike() positionally.
    By convention this is the block named "beta", but it can be any name
    -- declare it via the `main_block_name` argument.
    """

    def __init__(self, likelihood, main_block_name="beta"):
        self.likelihood = likelihood
        self.main_block_name = main_block_name
        self.blocks = []          # list[_Block], order matters (visualisation)
        self._block_by_name = {}
        # set of block names that the likelihood expects via set_extra
        self._extra_names = set(likelihood.extra_param_names())
        self._main_added = False
        self._extras_added = set()

    # ------------------------------------------------------------------
    def add_block(self, name, priors, proposal):
        """Add a generic block. Must be called for the main (beta) block."""
        if name in self._block_by_name:
            raise ValueError(f"duplicate block name {name!r}.")
        block = _Block(name, priors, proposal)
        self.blocks.append(block)
        self._block_by_name[name] = block
        if name == self.main_block_name:
            self._main_added = True
        if name in self._extra_names:
            self._extras_added.add(name)
        return block

    def add_extra_block(self, name, prior, proposal):
        """
        Add a block for a likelihood-owned parameter (e.g. sigma).
        Must match a name in likelihood.extra_param_names().
        """
        if name not in self._extra_names:
            raise ValueError(
                f"likelihood does not declare {name!r} as an extra "
                f"parameter; declared extras: {sorted(self._extra_names)}.")
        return self.add_block(name, prior, proposal)

    # ------------------------------------------------------------------
    def validate(self):
        """Sanity check before sampling."""
        if not self._main_added:
            raise RuntimeError(
                f"main block {self.main_block_name!r} not added.")
        missing = self._extra_names - self._extras_added
        if missing:
            raise RuntimeError(
                f"likelihood declares extras {sorted(missing)} but no "
                f"corresponding blocks were added.")

    # ------------------------------------------------------------------
    def draw_initial(self):
        return {b.name: b.draw_initial() for b in self.blocks}

    def logpost(self, state):
        """Evaluate log p(theta) + log p(data | theta) at a state dict."""
        # 1. priors first -- short-circuit on -inf
        lp = 0.0
        for b in self.blocks:
            lp_b = b.logprior(state[b.name])
            if not np.isfinite(lp_b):
                return -np.inf
            lp += lp_b
        # 2. push extras into likelihood
        for name in self._extra_names:
            self.likelihood.set_extra(name, state[name])
        # 3. likelihood
        beta = state[self.main_block_name]
        ll = self.likelihood.loglike(beta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    @property
    def param_names(self):
        """Flat parameter names, e.g. ['beta_0','beta_1','sigma']."""
        names = []
        for b in self.blocks:
            if b.dim == 1:
                names.append(b.name)
            else:
                names.extend(f"{b.name}_{k}" for k in range(b.dim))
        return names
