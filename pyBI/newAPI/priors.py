"""
Priors. Each prior knows how to:
  - evaluate its log-pdf at a value (logpdf)
  - draw a sample from itself (draw)
  - report its support (lower, upper) for plotting

That is all. Proposals are SEPARATE objects; see proposals.py.

Why separate? In random-walk MCMC the proposal mechanism (Gaussian RW,
log-scale RW, reflective, ...) is independent of the prior. Mixing them
into one class causes bugs (e.g. a Gaussian RW proposal on a positive-
support variable produces invalid draws; a log-scale proposal needs a
Hastings correction). Keeping them apart lets each concern be tested
and reasoned about in isolation.
"""

import numpy as np


class Prior:
    """Abstract base."""
    def logpdf(self, x):
        raise NotImplementedError
    def draw(self):
        raise NotImplementedError
    @property
    def lower(self):
        raise NotImplementedError
    @property
    def upper(self):
        raise NotImplementedError


class Uniform(Prior):
    """Uniform(a, b)."""
    def __init__(self, a, b):
        if not b > a:
            raise ValueError(f"Uniform: need b > a, got a={a}, b={b}.")
        self.a = float(a)
        self.b = float(b)
        self._log_width = np.log(self.b - self.a)

    def logpdf(self, x):
        if (x > self.a) and (x < self.b):
            return -self._log_width
        return -np.inf

    def draw(self):
        return float(np.random.rand() * (self.b - self.a) + self.a)

    @property
    def lower(self): return self.a
    @property
    def upper(self): return self.b

    def __repr__(self):
        return f"Uniform({self.a}, {self.b})"


class Normal(Prior):
    """Normal(mu, sigma)."""
    def __init__(self, mu, sigma):
        if sigma <= 0:
            raise ValueError(f"Normal: need sigma > 0, got {sigma}.")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self._cst = -0.5 * np.log(2 * np.pi * self.sigma ** 2)

    def logpdf(self, x):
        return self._cst - 0.5 * ((x - self.mu) / self.sigma) ** 2

    def draw(self):
        return float(np.random.randn() * self.sigma + self.mu)

    @property
    def lower(self): return self.mu - 6 * self.sigma
    @property
    def upper(self): return self.mu + 6 * self.sigma

    def __repr__(self):
        return f"Normal({self.mu}, {self.sigma})"


class HalfNormal(Prior):
    """HalfNormal(sigma) -- support on [0, inf)."""
    def __init__(self, sigma):
        if sigma <= 0:
            raise ValueError(f"HalfNormal: need sigma > 0, got {sigma}.")
        self.sigma = float(sigma)
        self._cst = 0.5 * np.log(2.0 / np.pi) - np.log(self.sigma)

    def logpdf(self, x):
        if x <= 0:
            return -np.inf
        return self._cst - x ** 2 / (2 * self.sigma ** 2)

    def draw(self):
        return float(abs(np.random.randn() * self.sigma))

    @property
    def lower(self): return 0.0
    @property
    def upper(self): return 6 * self.sigma

    def __repr__(self):
        return f"HalfNormal({self.sigma})"


class LogNormal(Prior):
    """LogNormal: log(x) ~ Normal(mu, sigma).  Support (0, inf)."""
    def __init__(self, mu, sigma):
        if sigma <= 0:
            raise ValueError(f"LogNormal: need sigma > 0, got {sigma}.")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self._cst = -0.5 * np.log(2 * np.pi * self.sigma ** 2)

    def logpdf(self, x):
        if x <= 0:
            return -np.inf
        return (self._cst - np.log(x)
                - 0.5 * ((np.log(x) - self.mu) / self.sigma) ** 2)

    def draw(self):
        return float(np.exp(np.random.randn() * self.sigma + self.mu))

    @property
    def lower(self): return float(np.exp(self.mu - 6 * self.sigma))
    @property
    def upper(self): return float(np.exp(self.mu + 6 * self.sigma))

    def __repr__(self):
        return f"LogNormal({self.mu}, {self.sigma})"
