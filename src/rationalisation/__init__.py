"""The rationalisation model (docs/rationalisation.md): the shared algebra
(algebra.py), the domain (domain.py), the report vocabulary (report.py),
and the runner — the Cogitator (cogitator.py)."""

from kalvin.kline import KLine
from rationalisation.algebra import (
    BETA,
    Band,
    band,
    can,
    contained,
    effort,
    entropy,
    fit,
    mu,
    overfit,
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
    "Domain",
    "End",
    "Event",
    "KLine",
    "Proposal",
    "Rationalisation",
    "band",
    "can",
    "contained",
    "effort",
    "entropy",
    "fit",
    "mu",
    "overfit",
    "proposals",
    "rationalise",
    "underfit",
    "significance",
]
