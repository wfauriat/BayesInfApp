"""
Trace: a chain (or set of chains) plus diagnostics.

A Trace holds:
   samples       : (n_iter, n_params)  -- raw chain (post-burn-in)
   logpost       : (n_iter,)
   acc_rates     : dict[block_name, float]
   param_names   : list[str], flat
   block_layout  : dict[block_name, slice]  (so we can pick out beta etc.)

It exposes:
   summary()           -> mean, std, ESS, HDI per parameter
   autocorr(lag_max)   -> autocorrelation function
   ess()               -> effective sample size per parameter
   plot_trace()        -> trace + histogram per parameter
   plot_pairs()        -> scatter pairs

Multiple chains can be merged via Trace.combine([t1, t2, ...]) for R-hat.
"""

import numpy as np


# ----------------------------------------------------------------------
def _autocorr_1d(x, max_lag=None):
    """
    Autocorrelation function up to max_lag, computed via FFT for speed.
    Returns r[0..max_lag] with r[0]=1.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if max_lag is None:
        max_lag = n // 4
    x = x - x.mean()
    # zero-pad to next power of 2 for FFT efficiency
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(f * np.conjugate(f))[:n]
    acf /= acf[0]
    return acf[:max_lag + 1]


def _ess_1d(x):
    """
    Effective sample size via initial monotone sequence estimator
    (Geyer 1992). Sums autocorrelation pairs r[2k] + r[2k+1] until
    a pair becomes non-positive.
    """
    n = len(x)
    acf = _autocorr_1d(x, max_lag=min(n - 1, 2000))
    # Geyer's initial monotone sequence: pair adjacent lags
    pair_sums = acf[1::2] + acf[2::2]
    if len(pair_sums) == 0:
        return float(n)
    # truncate at first non-positive pair
    idx = np.where(pair_sums <= 0)[0]
    cutoff = idx[0] if len(idx) > 0 else len(pair_sums)
    pair_sums = pair_sums[:cutoff]
    tau = 1.0 + 2.0 * float(np.sum(pair_sums))
    tau = max(tau, 1.0)
    return n / tau


def _hdi_1d(x, prob=0.95):
    """Highest density interval for a 1D sample."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    k = int(np.floor(prob * n))
    if k < 1 or k >= n:
        return float(x[0]), float(x[-1])
    widths = x[k:] - x[:n - k]
    j = int(np.argmin(widths))
    return float(x[j]), float(x[j + k])


# ======================================================================
class Trace:
    """One chain (post-burn-in)."""

    def __init__(self, samples, logpost, acc_rates, param_names,
                 block_layout, raw_chain=None, n_burn=None, n_thin=1,
                 proposal_history=None, snapshot_history=None):
        self.samples = np.asarray(samples)              # (n, p)
        self.logpost = np.asarray(logpost)              # (n,)
        self.acc_rates = dict(acc_rates)
        self.param_names = list(param_names)
        self.block_layout = dict(block_layout)
        self.raw_chain = raw_chain                      # full chain incl burn-in
        self.n_burn = n_burn
        self.n_thin = n_thin
        # New fields:
        # proposal_history : list of dicts, one entry per tuning event:
        #     {"iteration": int, "blocks": {name: snapshot_dict}}
        # snapshot_history : list of (iteration, state_vec, logpost) tuples
        #     captured every `snapshot_every` iterations during the run
        self.proposal_history = list(proposal_history or [])
        self.snapshot_history = list(snapshot_history or [])
        if self.samples.shape[1] != len(self.param_names):
            raise ValueError(
                f"samples has {self.samples.shape[1]} cols, "
                f"param_names has {len(self.param_names)}.")

    # ---- access ------------------------------------------------------
    def __len__(self):
        return self.samples.shape[0]

    def get(self, block_name):
        """Return columns for a named block (1D for scalar block, 2D for vector)."""
        sl = self.block_layout[block_name]
        out = self.samples[:, sl]
        return out[:, 0] if out.shape[1] == 1 else out

    @property
    def map_estimate(self):
        i = int(np.argmax(self.logpost))
        return {b: (self.samples[i, sl][0] if (sl.stop - sl.start) == 1
                    else self.samples[i, sl])
                for b, sl in self.block_layout.items()}

    # ---- diagnostics -------------------------------------------------
    def ess(self):
        return {n: _ess_1d(self.samples[:, k])
                for k, n in enumerate(self.param_names)}

    def autocorr(self, max_lag=None):
        return {n: _autocorr_1d(self.samples[:, k], max_lag)
                for k, n in enumerate(self.param_names)}

    def summary(self, prob=0.95, print_it=True):
        rows = []
        for k, n in enumerate(self.param_names):
            x = self.samples[:, k]
            lo, hi = _hdi_1d(x, prob)
            rows.append({
                "param":  n,
                "mean":   float(x.mean()),
                "std":    float(x.std(ddof=1)),
                "median": float(np.median(x)),
                f"hdi_{int(prob*100)}_lo": lo,
                f"hdi_{int(prob*100)}_hi": hi,
                "ess":    _ess_1d(x),
            })
        if print_it:
            self._print_summary(rows, prob)
        return rows

    @staticmethod
    def _print_summary(rows, prob):
        cols = ["param", "mean", "std", "median",
                f"hdi_{int(prob*100)}_lo", f"hdi_{int(prob*100)}_hi", "ess"]
        widths = {c: max(len(c), max(
            len(f"{r[c]:.4g}") if isinstance(r[c], float) else len(str(r[c]))
            for r in rows)) for c in cols}
        header = "  ".join(c.rjust(widths[c]) for c in cols)
        sep = "-" * len(header)
        print(header)
        print(sep)
        for r in rows:
            line = "  ".join(
                f"{r[c]:.4g}".rjust(widths[c]) if isinstance(r[c], float)
                else str(r[c]).rjust(widths[c]) for c in cols)
            print(line)
        print(sep)
        print("acceptance rates:")
        # acc_rates printed externally if desired

    # ---- plotting ----------------------------------------------------
    def plot_trace(self, figsize=None):
        import matplotlib.pyplot as plt
        p = len(self.param_names)
        figsize = figsize or (10, 1.6 * p)
        fig, axs = plt.subplots(p, 2, figsize=figsize)
        if p == 1:
            axs = axs[None, :]
        for k, name in enumerate(self.param_names):
            x = self.samples[:, k]
            axs[k, 0].plot(x, lw=0.4, color='k')
            axs[k, 0].set_ylabel(name)
            if k == p - 1:
                axs[k, 0].set_xlabel("iteration (post-burn)")
            axs[k, 1].hist(x, bins=40, edgecolor='k', alpha=0.5)
            axs[k, 1].set_xlabel(name)
        fig.tight_layout()
        return fig, axs

    def plot_pairs(self, figsize=None):
        import matplotlib.pyplot as plt
        from itertools import combinations
        p = len(self.param_names)
        if p < 2:
            raise ValueError("need at least 2 parameters for pair plot.")
        pairs = list(combinations(range(p), 2))
        ncols = int(np.ceil(np.sqrt(len(pairs))))
        nrows = int(np.ceil(len(pairs) / ncols))
        figsize = figsize or (3.5 * ncols, 3 * nrows)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        for q, (i, j) in enumerate(pairs):
            ax = axs[q // ncols, q % ncols]
            ax.scatter(self.samples[:, i], self.samples[:, j],
                       c=self.logpost, cmap='viridis', s=4)
            ax.set_xlabel(self.param_names[i])
            ax.set_ylabel(self.param_names[j])
        for q in range(len(pairs), nrows * ncols):
            axs[q // ncols, q % ncols].set_visible(False)
        fig.tight_layout()
        return fig, axs

    # ---- proposal-history introspection ------------------------------
    def proposal_summary(self, print_it=True):
        """
        Print the proposal state at each tuning event during burn-in,
        plus the final state. Returns the history list (also stored on
        self.proposal_history).
        """
        rows = []
        for entry in self.proposal_history:
            it = entry["iteration"]
            for bname, snap in entry["blocks"].items():
                rows.append({"iteration": it, "block": bname, **snap})
        if print_it and rows:
            keys = sorted({k for r in rows for k in r if k not in ("iteration", "block")})
            cols = ["iteration", "block"] + keys
            widths = {c: max(len(c), max(
                len(_fmt(r.get(c, ""))) for r in rows)) for c in cols}
            header = "  ".join(c.rjust(widths[c]) for c in cols)
            print(header)
            print("-" * len(header))
            for r in rows:
                print("  ".join(_fmt(r.get(c, "")).rjust(widths[c]) for c in cols))
        return self.proposal_history

    def final_proposals(self):
        """The proposal state at the LAST recorded tuning event (or None)."""
        return self.proposal_history[-1] if self.proposal_history else None

    # ---- snapshot evolution plot -------------------------------------
    def plot_evolution(self, figsize=None, include_burn_in=True):
        """
        Plot the chain trajectory through the snapshots collected during
        the run. One subplot per parameter; x-axis is iteration count,
        y-axis is parameter value. Useful to see convergence in real time
        (after the fact).
        """
        if not self.snapshot_history:
            raise RuntimeError(
                "no snapshots recorded; pass snapshot_every=K to the sampler.")
        import matplotlib.pyplot as plt
        its = np.array([s[0] for s in self.snapshot_history])
        states = np.array([s[1] for s in self.snapshot_history])  # (S, p)
        lp = np.array([s[2] for s in self.snapshot_history])
        p = states.shape[1]
        figsize = figsize or (10, 1.5 * (p + 1))
        fig, axs = plt.subplots(p + 1, 1, figsize=figsize, sharex=True)
        for k in range(p):
            axs[k].plot(its, states[:, k], "-o", ms=3, lw=0.7, color="k")
            axs[k].set_ylabel(self.param_names[k])
            if self.n_burn is not None:
                axs[k].axvline(self.n_burn, color="r", ls="--", lw=1,
                               label="end of burn-in" if k == 0 else None)
        axs[-1].plot(its, lp, "-o", ms=3, lw=0.7, color="b")
        axs[-1].set_ylabel("logpost")
        axs[-1].set_xlabel("iteration")
        if self.n_burn is not None:
            axs[-1].axvline(self.n_burn, color="r", ls="--", lw=1)
        if self.n_burn is not None:
            axs[0].legend(loc="best", fontsize=8)
        fig.tight_layout()
        return fig, axs


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, np.ndarray):
        if v.ndim == 1:
            return "[" + ", ".join(f"{x:.3g}" for x in v) + "]"
        return f"<{v.shape}>"
    return str(v)


# ----------------------------------------------------------------------
def rhat(traces):
    """
    Gelman-Rubin R-hat for multiple chains. traces: list[Trace], all
    with the same param_names. Returns dict[name, rhat].
    """
    if len(traces) < 2:
        raise ValueError("rhat needs >= 2 chains.")
    names = traces[0].param_names
    out = {}
    for k, n in enumerate(names):
        chains = np.array([t.samples[:, k] for t in traces])  # (m, ndraw)
        m, ndraw = chains.shape
        chain_means = chains.mean(axis=1)
        chain_vars = chains.var(axis=1, ddof=1)
        W = chain_vars.mean()
        B = ndraw * chain_means.var(ddof=1)
        var_hat = (1 - 1.0 / ndraw) * W + B / ndraw
        out[n] = float(np.sqrt(var_hat / W)) if W > 0 else float('inf')
    return out