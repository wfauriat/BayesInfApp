"""
Two samplers, both block-agnostic.

  - MetropolisBlock : at each iteration, propose a joint move within
                      each block (vector blocks via BlockGaussianRW;
                      scalar blocks via their scalar Proposal).
                      Each block update is a separate Metropolis step.

  - MetropolisGibbs : at each iteration, sweep through every scalar
                      coordinate (across all blocks) in random order
                      and update each via a 1D Metropolis step. For
                      vector blocks, per-coordinate proposals are
                      derived from the diagonal of the block's
                      BlockGaussianRW covariance.

Both samplers share the same adaptation philosophy:
  - During burn-in: every TUNE_INTERVAL iterations, adapt each
    proposal's scale based on the recent acceptance rate. For
    vector blocks, also reshape the proposal covariance from the
    empirical covariance of recent samples.
  - After burn-in: NO adaptation (proposals are frozen, chain is
    reversible).

Both return a Trace object.
"""

import numpy as np
from copy import deepcopy

from .proposals import BlockGaussianRW, GaussianRW, _tune_step
from .trace import Trace


TUNE_INTERVAL = 200
SHAPE_ADAPT_AFTER = 500   # don't try to estimate cov until we have samples


# ======================================================================
def _state_to_vec(state, blocks):
    """Pack a state dict into a flat 1D array, in block order."""
    pieces = []
    for b in blocks:
        v = state[b.name]
        pieces.append(np.atleast_1d(np.asarray(v, dtype=float)))
    return np.concatenate(pieces)


def _vec_to_state(vec, blocks):
    """Unpack a flat 1D array into a state dict."""
    out = {}
    i = 0
    for b in blocks:
        if b.dim == 1:
            out[b.name] = float(vec[i])
        else:
            out[b.name] = vec[i:i + b.dim].copy()
        i += b.dim
    return out


def _block_layout(blocks):
    """{block_name: slice_into_flat_vector}"""
    out = {}
    i = 0
    for b in blocks:
        out[b.name] = slice(i, i + b.dim)
        i += b.dim
    return out


# ======================================================================
class _SamplerBase:
    """Common machinery: state, run loop, trace assembly."""

    def __init__(self, problem, n_iter, n_burn=None, n_thin=1,
                 adaptive=True, verbose=False, seed=None,
                 snapshot_every=None, progress_every=None):
        """
        snapshot_every : int or None
            If set, record (iteration, state, logpost) every K steps for
            later inspection via Trace.plot_evolution() and
            Trace.snapshot_history. Snapshots cover the whole run
            (including burn-in) so you can see convergence behaviour.

        progress_every : int or None
            If set, print a one-line summary of mean / std / acceptance
            over the last `progress_every` iterations every K steps.
            Lightweight live monitoring; does not modify the chain.
        """
        problem.validate()
        self.problem = problem
        self.blocks = problem.blocks
        self.n_iter = int(n_iter)
        self.n_burn = int(n_burn if n_burn is not None else n_iter // 5)
        self.n_thin = max(1, int(n_thin))
        self.adaptive = adaptive
        self.verbose = verbose
        self.snapshot_every = (None if snapshot_every is None
                               else max(1, int(snapshot_every)))
        self.progress_every = (None if progress_every is None
                               else max(1, int(progress_every)))
        if seed is not None:
            np.random.seed(int(seed))

    # ------------------------------------------------------------------
    def _initial_state(self, init=None):
        return self.problem.draw_initial() if init is None else dict(init)

    def _flat_dim(self):
        return sum(b.dim for b in self.blocks)

    def _block_acceptance_keys(self):
        return [b.name for b in self.blocks]

    # ------------------------------------------------------------------
    def _build_trace(self, raw, raw_lp, acc_counts, n_total_proposals,
                     proposal_history=None, snapshot_history=None):
        # Apply burn-in then thinning
        post = raw[self.n_burn::self.n_thin]
        post_lp = raw_lp[self.n_burn::self.n_thin]
        acc_rates = {
            name: (acc_counts[name] / max(1, n_total_proposals[name]))
            for name in acc_counts
        }
        param_names = self.problem.param_names
        layout = _block_layout(self.blocks)
        return Trace(samples=post, logpost=post_lp,
                     acc_rates=acc_rates, param_names=param_names,
                     block_layout=layout, raw_chain=raw,
                     n_burn=self.n_burn, n_thin=self.n_thin,
                     proposal_history=proposal_history,
                     snapshot_history=snapshot_history)

    # to be implemented
    def run(self, init=None):
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _capture_proposals(self, iteration):
        """Snapshot every block's proposal state at the given iteration."""
        return {
            "iteration": int(iteration),
            "blocks": {b.name: b.proposal.state() for b in self.blocks},
        }

    def _print_progress(self, it, raw, raw_lp, window):
        """One-line live summary over the last `window` iterations."""
        lo = max(0, it - window + 1)
        recent = raw[lo:it + 1]
        means = recent.mean(axis=0)
        stds  = recent.std(axis=0)
        lp_now = raw_lp[it]
        names = self.problem.param_names
        n_print = min(len(names), 4)   # avoid flooding for many params
        summary = "  ".join(
            f"{names[k]}={means[k]:+.3f}±{stds[k]:.3f}"
            for k in range(n_print))
        more = f" [+{len(names)-n_print} more]" if len(names) > n_print else ""
        phase = "burn" if it < self.n_burn else "post"
        print(f"  it {it:>7d} ({phase})  logpost={lp_now:+.2f}   {summary}{more}")


# ======================================================================
class MetropolisBlock(_SamplerBase):
    """
    Block Metropolis: at each iteration, do one Metropolis update per
    block. Vector blocks use BlockGaussianRW (joint move); scalar
    blocks use their scalar proposal.
    """

    def run(self, init=None):
        state = self._initial_state(init)
        log_post = self.problem.logpost(state)
        if not np.isfinite(log_post):
            raise RuntimeError(
                "Initial log-posterior is -inf. Check priors and initial "
                "state -- did you draw inside the support?")

        D = self._flat_dim()
        raw = np.zeros((self.n_iter, D))
        raw_lp = np.zeros(self.n_iter)
        raw[0] = _state_to_vec(state, self.blocks)
        raw_lp[0] = log_post

        block_names = self._block_acceptance_keys()
        acc = {n: 0 for n in block_names}                     # post-burn count
        prop = {n: 0 for n in block_names}
        interval_acc = {n: 0 for n in block_names}
        interval_prop = {n: 0 for n in block_names}

        proposal_history = [self._capture_proposals(0)]
        snapshot_history = []
        if self.snapshot_every is not None:
            snapshot_history.append((0, raw[0].copy(), raw_lp[0]))

        for it in range(1, self.n_iter):
            # ---- one Metropolis step per block ----
            for b in self.blocks:
                cand_state = dict(state)
                cur = state[b.name]
                cand_val, log_q = b.proposal.propose(cur)
                cand_state[b.name] = cand_val
                cand_lp = self.problem.logpost(cand_state)

                interval_prop[b.name] += 1
                if it > self.n_burn:
                    prop[b.name] += 1

                ldiff = cand_lp - log_post + log_q
                if np.log(np.random.rand()) < ldiff:
                    state = cand_state
                    log_post = cand_lp
                    interval_acc[b.name] += 1
                    if it > self.n_burn:
                        acc[b.name] += 1

            raw[it] = _state_to_vec(state, self.blocks)
            raw_lp[it] = log_post

            # ---- adaptation, burn-in only ----
            if self.adaptive and it < self.n_burn and it % TUNE_INTERVAL == 0:
                # 1. multiplicative scale tweak per block
                for b in self.blocks:
                    rate = interval_acc[b.name] / max(1, interval_prop[b.name])
                    b.proposal.adapt(rate)
                # 2. for vector blocks, reshape from empirical cov
                if it > SHAPE_ADAPT_AFTER:
                    layout = _block_layout(self.blocks)
                    for b in self.blocks:
                        if b.dim > 1 and isinstance(b.proposal, BlockGaussianRW):
                            window = raw[max(0, it - 500):it, layout[b.name]]
                            b.proposal.adapt_from_samples(window)
                # record proposal state after this tuning event
                proposal_history.append(self._capture_proposals(it))
                # reset interval counters
                for n in block_names:
                    interval_acc[n] = 0
                    interval_prop[n] = 0

            # ---- snapshots and live progress (no chain side-effects) ----
            if (self.snapshot_every is not None
                    and it % self.snapshot_every == 0):
                snapshot_history.append((it, raw[it].copy(), raw_lp[it]))
            if (self.progress_every is not None
                    and it % self.progress_every == 0):
                self._print_progress(it, raw, raw_lp, self.progress_every)

            if self.verbose and it % max(1, self.n_iter // 10) == 0:
                print(f"  it {it:>7d} / {self.n_iter}  logpost={log_post:.3f}")

        # final proposal snapshot (after the run, no further tuning)
        proposal_history.append(self._capture_proposals(self.n_iter - 1))

        return self._build_trace(raw, raw_lp, acc, prop,
                                 proposal_history=proposal_history,
                                 snapshot_history=snapshot_history)


# ======================================================================
class MetropolisGibbs(_SamplerBase):
    """
    Metropolis-within-Gibbs: at each iteration, sweep through every
    scalar coordinate of every block in random order, doing a 1D
    Metropolis update per coordinate.

    For vector blocks (e.g. beta), per-coordinate proposals are
    derived from the diagonal of the block's BlockGaussianRW
    covariance. The block's joint covariance is still adapted from
    empirical samples (so the diagonal stays accurate).
    """

    def run(self, init=None):
        state = self._initial_state(init)
        log_post = self.problem.logpost(state)
        if not np.isfinite(log_post):
            raise RuntimeError(
                "Initial log-posterior is -inf. Check priors and initial "
                "state -- did you draw inside the support?")

        D = self._flat_dim()
        raw = np.zeros((self.n_iter, D))
        raw_lp = np.zeros(self.n_iter)
        raw[0] = _state_to_vec(state, self.blocks)
        raw_lp[0] = log_post

        # Per-coordinate scratch: which block, which position-within-block
        coord_table = []
        for bi, b in enumerate(self.blocks):
            for k in range(b.dim):
                coord_table.append((bi, k))
        n_coords = len(coord_table)

        # For vector blocks: scalar proposals derived per-coord from
        # diag of L L^T  (since L L^T is the proposal covariance, diag
        # is the per-coord variance).
        # We rebuild these on each adaptation.
        per_coord_scales = self._compute_per_coord_scales()

        acc = {n: 0 for n in self._block_acceptance_keys()}
        prop = {n: 0 for n in self._block_acceptance_keys()}
        # per-coord interval counters (so we can adapt the diag-derived
        # scales of vector blocks coordinate-by-coordinate)
        interval_acc_coord = np.zeros(n_coords)
        interval_prop_coord = np.zeros(n_coords)

        proposal_history = [self._capture_proposals_mwg(0, per_coord_scales,
                                                        coord_table)]
        snapshot_history = []
        if self.snapshot_every is not None:
            snapshot_history.append((0, raw[0].copy(), raw_lp[0]))

        for it in range(1, self.n_iter):
            order = np.random.permutation(n_coords)
            for ci in order:
                bi, k = coord_table[ci]
                b = self.blocks[bi]
                cand_state = dict(state)
                if b.dim == 1:
                    cand_val, log_q = b.proposal.propose(state[b.name])
                    cand_state[b.name] = cand_val
                else:
                    cur_vec = state[b.name].copy()
                    # scalar Gaussian RW on coordinate k
                    cur_vec[k] += per_coord_scales[ci] * np.random.randn()
                    cand_state[b.name] = cur_vec
                    log_q = 0.0

                cand_lp = self.problem.logpost(cand_state)
                interval_prop_coord[ci] += 1
                if it > self.n_burn:
                    prop[b.name] += 1
                ldiff = cand_lp - log_post + log_q
                if np.log(np.random.rand()) < ldiff:
                    state = cand_state
                    log_post = cand_lp
                    interval_acc_coord[ci] += 1
                    if it > self.n_burn:
                        acc[b.name] += 1

            raw[it] = _state_to_vec(state, self.blocks)
            raw_lp[it] = log_post

            # ---- adaptation, burn-in only ----
            if self.adaptive and it < self.n_burn and it % TUNE_INTERVAL == 0:
                # per-coordinate scale tweak based on coord acc rate
                for ci in range(n_coords):
                    bi, k = coord_table[ci]
                    b = self.blocks[bi]
                    rate = (interval_acc_coord[ci]
                            / max(1, interval_prop_coord[ci]))
                    if b.dim == 1:
                        # let the scalar Proposal adapt itself
                        b.proposal.adapt(rate)
                    else:
                        # adjust per-coord scale via the same ladder
                        per_coord_scales[ci] = _tune_step(
                            per_coord_scales[ci], rate, target=0.44)
                # reshape vector-block proposal covariance from samples
                if it > SHAPE_ADAPT_AFTER:
                    layout = _block_layout(self.blocks)
                    for b in self.blocks:
                        if b.dim > 1 and isinstance(b.proposal, BlockGaussianRW):
                            window = raw[max(0, it - 500):it, layout[b.name]]
                            b.proposal.adapt_from_samples(window)
                    # rebuild per-coord scales from the new covariance,
                    # but DON'T overwrite hand-tuned scalar block scales.
                    new_scales = self._compute_per_coord_scales()
                    for ci in range(n_coords):
                        bi, _ = coord_table[ci]
                        if self.blocks[bi].dim > 1:
                            per_coord_scales[ci] = new_scales[ci]
                # record proposal state
                proposal_history.append(self._capture_proposals_mwg(
                    it, per_coord_scales, coord_table))
                interval_acc_coord[:] = 0
                interval_prop_coord[:] = 0

            # ---- snapshots and live progress ----
            if (self.snapshot_every is not None
                    and it % self.snapshot_every == 0):
                snapshot_history.append((it, raw[it].copy(), raw_lp[it]))
            if (self.progress_every is not None
                    and it % self.progress_every == 0):
                self._print_progress(it, raw, raw_lp, self.progress_every)

            if self.verbose and it % max(1, self.n_iter // 10) == 0:
                print(f"  it {it:>7d} / {self.n_iter}  logpost={log_post:.3f}")

        proposal_history.append(self._capture_proposals_mwg(
            self.n_iter - 1, per_coord_scales, coord_table))

        return self._build_trace(raw, raw_lp, acc, prop,
                                 proposal_history=proposal_history,
                                 snapshot_history=snapshot_history)

    # ------------------------------------------------------------------
    def _capture_proposals_mwg(self, iteration, per_coord_scales, coord_table):
        """
        Capture proposals AND the per-coordinate scalar scales used by
        MwG for vector blocks. The block's BlockGaussianRW.state() shows
        the joint-cov view; we also report what MwG actually uses
        coordinate-by-coordinate.
        """
        out = {"iteration": int(iteration), "blocks": {}}
        # collect per-coord scales by block
        block_coord_scales = {b.name: [] for b in self.blocks}
        for ci, (bi, k) in enumerate(coord_table):
            block_coord_scales[self.blocks[bi].name].append(
                float(per_coord_scales[ci]))
        for b in self.blocks:
            snap = b.proposal.state()
            if b.dim > 1:
                snap = dict(snap)  # copy
                snap["mwg_coord_scales"] = np.asarray(
                    block_coord_scales[b.name], dtype=float)
            out["blocks"][b.name] = snap
        return out

    # ------------------------------------------------------------------
    def _compute_per_coord_scales(self):
        scales = []
        for b in self.blocks:
            if b.dim == 1:
                # for scalar blocks, the scalar proposal manages its own
                # scale; the per-coord entry is unused.
                scales.append(1.0)  # placeholder
            else:
                # diag of (L L^T) = sum row-wise of squared entries of L
                if isinstance(b.proposal, BlockGaussianRW):
                    L = b.proposal.L
                    diag_var = np.sum(L * L, axis=1)
                    scales.extend(np.sqrt(np.maximum(diag_var, 1e-12)).tolist())
                else:
                    scales.extend([1.0] * b.dim)
        return np.asarray(scales, dtype=float)