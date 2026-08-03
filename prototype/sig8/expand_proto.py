"""sig8.expand_proto — the new compose-on-return ``expand()``, against real model.

A faithful port of ``src/kalvin/expand.py:expand`` that reuses the real
``Model``, ``NLPSignifier``, ``KLine``, production ``edge_hops`` and ``is_s1``
(traversal is unchanged — Q13/the grill's "expand is sound and stays"), but
swaps the sum-and-invert aggregation for the compose-on-return scheme.

What this file is for: running the new aggregation against the same scenarios
as ``tests/test_expand.py`` and printing a before/after table, so we can
*observe* whether the redesign behaves as expected before designing further.

Differences from production ``expand()``:

- per-node resolution is captured into a ``list[float]`` of accountedness
  (Q11/12 + Q17a), not summed into ``total_distance``;
- the terminal significance is the compose-on-return byte (Q10/Q16), not
  ``(~total_distance) & MASK64``;
- the matched-ungrounded penalty (production ``+1``) becomes ``decay(1)``
  (Q17a);
- signifies / S3-connotation side-candidates are still yielded (Q17b), but
  now carry bytes derived from their own hop counts via the decay seam.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from kalvin.expand import edge_hops, is_s1  # reuse production traversal helpers
from kalvin.kline import KLine

from .aggregate import Aggregator, ExpansionResult
from .byte import SIG_MAX, distance_to_byte

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier
    from kalvin.model import Model


class ProtoQueryCandidate:
    """Mirrors production ``QueryCandidate`` + the diagnostic intermediate."""

    __slots__ = ("query", "candidate", "result", "kind")

    def __init__(self, query: KLine, candidate: KLine, result: ExpansionResult, kind: str):
        self.query = query
        self.candidate = candidate
        self.result = result
        self.kind = kind  # "terminal" | "signifies"  (Q18: E recurses, no side-candidate)

    @property
    def significance(self) -> int:
        return self.result.significance

    def __repr__(self) -> str:
        return (
            f"ProtoQC(q={self.query.signature:#x}, c={self.candidate.signature:#x}, "
            f"kind={self.kind}, sig={self.significance:#04x}, "
            f"frac={self.result.accounted_fraction:.3f})"
        )


def expand_proto(
    model: Model,
    query: KLine,
    candidate: KLine,
    signifier: KSignifier,
    aggregator: Aggregator,
    *,
    distance: int = 0,
    _visited: set[tuple[int, int]] | None = None,
) -> Iterator[ProtoQueryCandidate]:
    """Compose-on-return ``expand()``. See module docstring."""
    if _visited is None:
        _visited = set()

    key = (query.signature, candidate.signature)
    if key in _visited:
        return
    _visited.add(key)

    q_set = set(query.nodes)
    c_set = set(candidate.nodes)
    mismatched_q = q_set - c_set
    mismatched_c = c_set - q_set
    matched = q_set & c_set

    s3_connotations: dict[int, int] = {}  # sig -> min hops from any query node

    # Q17b: per-node accountedness, in slot order. One float per slot.
    slot_values: list[float] = []

    # ── mismatched query nodes ────────────────────────────────────────
    for n in mismatched_q:
        accounted = 0.0  # unresolvable default (case F)
        q_kline = model.find(n)
        if q_kline is not None:
            for hops, match_sig in edge_hops(model, n, signifier):
                if match_sig in mismatched_c:
                    # case C: resolves to opposing mismatch set (S2 direct).
                    accounted = aggregator.decay(hops)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        yield from expand_proto(
                            model, q_kline, c_kline, signifier, aggregator,
                            distance=hops, _visited=_visited,
                        )
                    break
                elif signifier.signifies(n, match_sig):
                    # case D: signifies (S2 loose). Yields a side-candidate
                    # (Q17b) AND contributes decay(hops) to this slot.
                    accounted = aggregator.decay(hops)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        side = aggregator.compose_terminal([aggregator.decay(hops)])
                        yield ProtoQueryCandidate(q_kline, c_kline, side, kind="signifies")
                    break
                elif match_sig not in s3_connotations or hops < s3_connotations[match_sig]:
                    s3_connotations[match_sig] = hops
        slot_values.append(accounted)

    # ── mismatched candidate nodes ────────────────────────────────────
    for n in mismatched_c:
        accounted = 0.0
        q_kline = model.find(n)
        if q_kline is not None:
            for hops, match_sig in edge_hops(model, n, signifier):
                if match_sig in mismatched_q:
                    accounted = aggregator.decay(hops)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        yield from expand_proto(
                            model, q_kline, c_kline, signifier, aggregator,
                            distance=hops, _visited=_visited,
                        )
                    break
                elif signifier.signifies(n, match_sig):
                    accounted = aggregator.decay(hops)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        side = aggregator.compose_terminal([aggregator.decay(hops)])
                        yield ProtoQueryCandidate(q_kline, c_kline, side, kind="signifies")
                    break
                elif match_sig in s3_connotations:
                    # case E: S3 connotation bridge.
                    s3_hop = s3_connotations[match_sig] + hops
                    accounted = aggregator.decay(s3_hop)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        # Q18 (a): preserve production's C/D/E asymmetry.
                        # E recurses (real structural edge) but does NOT emit a
                        # side-candidate at this level — only D (signifies) does.
                        yield from expand_proto(
                            model, q_kline, c_kline, signifier, aggregator,
                            distance=s3_hop, _visited=_visited,
                        )
                    break
        slot_values.append(accounted)

    # ── matched nodes ─────────────────────────────────────────────────
    # case A (matched & grounded) -> 1.0; case B (matched & ungrounded) -> decay(1) (Q17a).
    for n in matched:
        kl = model.find(n)
        if kl is not None and is_s1(model, kl, signifier):
            slot_values.append(1.0)
        elif kl is not None:
            slot_values.append(aggregator.decay(1))  # matched-ungrounded
        else:
            slot_values.append(aggregator.decay(1))  # not in model: treat as ungrounded match

    # Edge case: if both klines have no nodes at all, there are no slots.
    # Vacuously fully accounted — but this never happens for real klines.
    if not slot_values:
        slot_values = [1.0]

    terminal = aggregator.compose_terminal(slot_values)
    yield ProtoQueryCandidate(query, candidate, terminal, kind="terminal")


__all__ = ["ProtoQueryCandidate", "expand_proto"]
