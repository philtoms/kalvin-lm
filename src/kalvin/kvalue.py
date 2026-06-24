"""KValue — the unit of exchange: a KLine paired with a significance.

A KValue pairs an objective KLine with a sender's significance assessment of
it. The KLine is the data Kalvin stores; the significance is an assessment
that is re-derived on retrieval and never persisted.

See specs/kvalue.md (§Definition, §Construction, §Equality and Hashing,
§What a KValue is Not) for the authoritative contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from kalvin.kline import KLine


@dataclass(frozen=True)
class KValue:
    """An objective KLine paired with a sender's significance assessment.

    Two fields, both required (KV-1):

    - ``kline`` — the objective structure (immutable, shared).
    - ``significance`` — the sender's assessment (a uint64 on the inverted
      significance scale).

    Identity is structural: equality and hashing consider *only* ``kline``,
    ignoring ``significance`` entirely (KV-2, KV-3). Two participants may
    assess the same KLine differently without producing different KValues.

    Note: ``__eq__`` and ``__hash__`` are defined explicitly below because the
    ``@dataclass`` default would derive them from *all* fields, wrongly
    including ``significance``. Do not remove them — doing so would silently
    break structural identity.
    """

    kline: KLine
    significance: int

    def __eq__(self, other: object) -> bool:
        # Structural identity: compare kline only, ignoring significance.
        # Returns NotImplemented for non-KValue so Python falls back to
        # identity comparison (``kv == 42`` -> False).
        if not isinstance(other, KValue):
            return NotImplemented
        return self.kline == other.kline

    def __hash__(self) -> int:
        # Hash over kline only, so equal KValues (by kline) hash equally.
        return hash(self.kline)
