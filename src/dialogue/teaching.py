r"""User-significance teaching material.

A supervisor's graded response to a recently proposed kline is teaching
material, distinct from script significance:

- **S1** — ratification: the proposal grounds (handled by the fast path).
- **S2** — pattern: this shape is the kind of answer to give here.
- **S3** — pivot: this shape is a useful alignment base, not an answer.
- **S4** — refusal (handled by the S4 route).

S2/S3 arrive here when the stamped kline's signature is one K asked —
the stamp answers K's own proposal. The exemplar is recorded against
the *ask context* (the asked canon then in STM) so a later ask of the
same shape can be answered from learned behaviour rather than structure
alone. The goal: teach simple semantics — ``what`` and ``did`` trigger
response shapes — learned from responses that attracted significance,
never hard coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kalvin.kline import KLine, KNode, is_canon
from kalvin.kvalue import KValue

__all__ = ["Teaching", "Exemplar"]


@dataclass(frozen=True)
class Exemplar:
    """One taught response: the proposal, under the ask it answered."""

    proposal: KLine
    ask: KLine | None  #: the asked canon in STM at record time, if any

    def matches_ask(self, kline: KLine, signifier) -> bool:
        """Does ``kline`` pose the ask this exemplar answered?"""
        if self.ask is None:
            return False
        if kline.signature == self.ask.signature:
            return True
        return is_canon(kline, signifier) and bool(
            set(kline.nodes) & set(self.ask.nodes)
        )


@dataclass
class Teaching:
    """Learned patterns and pivots from supervisor-graded proposals."""

    patterns: list[Exemplar] = field(default_factory=list)
    pivots: list[Exemplar] = field(default_factory=list)

    def record(self, band: str, proposal: KLine, ask: KLine | None) -> None:
        """File a graded proposal as teaching material.

        ``band`` is the supervisor's stamp ("S2" pattern / "S3" pivot);
        anything else is ignored (S1/S4 take their engine routes).
        """
        exemplar = Exemplar(proposal, ask)
        if band == "S2":
            if not any(e.proposal == proposal for e in self.patterns):
                self.patterns.append(exemplar)
        elif band == "S3":
            if not any(e.proposal == proposal for e in self.pivots):
                self.pivots.append(exemplar)

    def pattern_for(self, kline: KLine, signifier) -> KValue | None:
        """The taught answer for an ask ``kline`` poses, or None."""
        for exemplar in reversed(self.patterns):
            if exemplar.matches_ask(kline, signifier):
                return KValue(exemplar.proposal, SIG_TAUGHT)
        return None

    def pivot_for(self, kline: KLine, signifier) -> Exemplar | None:
        """The taught pivot base for an ask ``kline`` poses, or None."""
        for exemplar in reversed(self.pivots):
            if exemplar.matches_ask(kline, signifier):
                return exemplar
        return None


#: Significance byte for a taught (learned, not yet ratified) response.
SIG_TAUGHT = 0xC0
