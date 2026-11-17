"""The rationalisation model (docs/rationalisation.md): the shared algebra
(algebra.py), the domain (domain.py), the report vocabulary (report.py),
and the runner — the Cogitator (cogitator.py)."""

from kalvin.kline import KLine
from rationalisation.algebra import (
    BETA,
    Band,
    Decomposition,
    band,
    can,
    contained,
    decompose,
    effort,
    entropy,
    fit,
    mu,
    underfit,
    significance,
)
from rationalisation.cogitator import Cogitator, rationalise
from rationalisation.domain import Domain
from rationalisation.report import End, Event, Proposal, Rationalisation, proposals

__all__ = [
    "BETA",
    "Band",
    "Cogitator",
    "Decomposition",
    "Domain",
    "End",
    "Event",
    "KLine",
    "Proposal",
    "Rationalisation",
    "band",
    "can",
    "contained",
    "decompose",
    "effort",
    "entropy",
    "fit",
    "mu",
    "proposals",
    "rationalise",
    "underfit",
    "significance",
]
