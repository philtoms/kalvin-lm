"""The rationalisation model (docs/rationalisation.md). Cogitator is the
default runner (rationalisation/cogitator.py); ReferenceCogitator is the
retained baseline (rationalisation/model.py)."""

from rationalisation.cogitator import Cogitator, rationalise
from rationalisation.model import (
    BETA,
    Band,
    Decomposition,
    Domain,
    End,
    Kline,
    Proposal,
    Rationalisation,
    Row,
    ReferenceCogitator,
    band,
    can,
    contained,
    decompose,
    effort,
    entropy,
    meet,
    mu,
    proposals,
    residue,
    significance,
)

__all__ = [
    "BETA",
    "Band",
    "Cogitator",
    "Decomposition",
    "Domain",
    "End",
    "Kline",
    "Proposal",
    "Rationalisation",
    "Row",
    "ReferenceCogitator",
    "band",
    "can",
    "contained",
    "decompose",
    "effort",
    "entropy",
    "meet",
    "mu",
    "proposals",
    "rationalise",
    "residue",
    "significance",
]
