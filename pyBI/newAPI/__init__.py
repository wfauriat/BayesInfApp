"""
pyBI2 -- minimal numpy-only Bayesian regression toolkit.

Public API:
    priors      : Uniform, Normal, HalfNormal, LogNormal
    proposals   : GaussianRW, LogScaleRW, BlockGaussianRW
    likelihoods : FullCovLike, FixedDiagLike, InferredDiagLike
    problem     : InferenceProblem
    samplers    : MetropolisBlock, MetropolisGibbs
    trace       : Trace, rhat
"""

from .priors import Uniform, Normal, HalfNormal, LogNormal
from .proposals import GaussianRW, LogScaleRW, BlockGaussianRW
from .likelihoods import FullCovLike, FixedDiagLike, InferredDiagLike
from .problem import InferenceProblem
from .samplers import MetropolisBlock, MetropolisGibbs
from .trace import Trace, rhat

__all__ = [
    "Uniform", "Normal", "HalfNormal", "LogNormal",
    "GaussianRW", "LogScaleRW", "BlockGaussianRW",
    "FullCovLike", "FixedDiagLike", "InferredDiagLike",
    "InferenceProblem",
    "MetropolisBlock", "MetropolisGibbs",
    "Trace", "rhat",
]
