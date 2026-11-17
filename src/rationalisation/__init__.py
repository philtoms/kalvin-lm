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
from rationalisation.report import Event, Proposal, Rationalisation

__all__ = [
    "BETA",
    "Band",
    "Cogitator",
    "Domain",
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
    "rationalise",
    "underfit",
    "significance",
]
