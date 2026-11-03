"""The rationalisation model (docs/rationalisation.md): the shared algebra
(algebra.py), the domain (domain.py), the report vocabulary (report.py),
and two runners — the Cogitator (cogitator.py, the default) and the
ReferenceCogitator (model.py, the retained baseline)."""

from rationalisation.algebra import (
    BETA,
    Band,
    Decomposition,
    Kline,
    band,
    can,
    contained,
    decompose,
    effort,
    entropy,
    meet,
    mu,
    residue,
    significance,
)
from rationalisation.cogitator import Cogitator, rationalise
from rationalisation.domain import Domain
from rationalisation.model import ReferenceCogitator
from rationalisation.report import End, Event, Proposal, Rationalisation, proposals

__all__ = [
    "BETA",
    "Band",
    "Cogitator",
    "Decomposition",
    "Domain",
    "End",
    "Event",
    "Kline",
    "Proposal",
    "Rationalisation",
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
