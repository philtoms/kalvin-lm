r"""The rationalising engine.

A :class:`Engine` derives one turn from ``(state, incoming)`` and returns
``(batch, observations)`` — dialogue emissions and K's internal S1 groundings
this turn. The engine is stateless about its own emissions; dedup lives in the
actor.

The engine is pure mechanism: it holds a :class:`EngineState` and a
:class:`MisfitStrategy`, both fully constructed by the caller. The factories
that assemble them (signifier, state, strategy, engine) live in
:mod:`dialogue.harness`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dialogue.engine_state import EngineState
from kalvin.kline import (
    KLine,
    is_canon,
    is_identity,
    is_misfit,
    is_unknown,
    sig_level,
    using_resolver,
)
from kalvin.kvalue import KValue
from kalvin.significance import (
    SIG_S1,
    SIG_S3,
    SIG_S4,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["Engine", "EngineState", "MisfitStrategy"]


@runtime_checkable
class MisfitStrategy(Protocol):
    """Propose for one pending misfit ``entry`` against the grounded store.

    Returns S2 proposals for the actor to emit, and may ground an entry
    directly via ``ground`` when a candidate fully accounts for it (the
    expand strategy's S1 case). The strategy shares the engine's
    :class:`EngineState` (set at construction).
    """

    def propose(
        self,
        entry: KLine,
        ground: Callable[[KLine], None],
    ) -> list[KValue]:
        ...


class Engine:
    """Derives one turn from ``incoming``.

    Holds the :class:`EngineState` it mutates in place and the
    :class:`MisfitStrategy` it consults for the S2 arm — both supplied fully
    constructed. The signifier is read off the state.
    """

    def __init__(
        self,
        state: EngineState,
        misfit: MisfitStrategy,
    ) -> None:
        self._state: EngineState = state
        self._misfit: MisfitStrategy = misfit

    @property
    def state(self) -> EngineState:
        """The engine's mutable memory, mutated in place each turn."""
        return self._state

    @property
    def signifier(self) -> KSignifier:
        """The state's signifier (the single source of truth)."""
        return self._state.signifier

    def rationalise(
        self, incoming: Sequence[KValue]
    ) -> tuple[list[KValue], list[KValue]]:
        """Route every incoming query, then cogitate. Returns ``(batch, observations)``."""
        self._state._dbg_step += 1

        self.observations: list[KValue] = []
        # The incoming queries this turn, in arrival order, retained so
        # cogitation can reply to them. The engine always replies when it
        # can support a reply from its own state; the actor filters per role.
        self._incoming: list[KValue] = []

        resolver = self._state.find
        with using_resolver(resolver):
            for query in incoming:
                self.route(query)
            return self.finish(self.cogitate())

    # ── Routing ──────────────────────────────────────────────────────

    def route(self, query: KValue) -> None:
        """Apply one incoming query as bookkeeping; emit nothing.

        Dispatch is on the query's **structural** significance:

        - **S1/S4 (fast route)** — match against the frame. An S1 (identity or
          canon) match promotes the kline (grounds it at S1 and cascades); an
          S4 (the empty ask ``{X:[]}``) pops the matching identity ask. Either
          way the framed kline is consumed. Unmatched queries are dropped.
        - **S2/S3 (slow route)** — append to the work-list, then unpack an S2
          misfit's unrecognised nodes and signature as identity asks.
        """
        self._incoming.append(query)
        kline = query.kline
        if sig_level(kline, self._state.signifier) in ("S1", "S4"):
            self._fast_route(query)
            return
        self._slow_route(query)

    def _slow_route(self, query: KValue) -> None:
        kline = query.kline
        self._state.work_list.append(kline)
        for node in kline.nodes:
            if not self._state.is_seen(node):
                self._state.work_list.append(KLine(node, [], kline.dbg))
        if not self._state.is_seen(kline.signature):
            self._state.work_list.append(KLine(kline.signature, [], kline.dbg))

    def _fast_route(self, query: KValue) -> None:
        # Structural S1 (an identity or canon) grounds whenever its signature
        # has been seen (framed as an ask, pending on the work-list, or already
        # grounded) — not only when an ask-shape is currently framed.
        # Identities are signature-only, and the ask may be emitted by this
        # same turn's cogitation (route-all-then-cogitate ordering), so the
        # work-list and grounded views matter as much as the frame.
        kline = query.kline
        if is_identity(kline) or is_canon(kline, self._state.signifier):
            if not self._state.signature_seen(kline.signature):
                self._slow_route(query)
                return
            self._state.unframe(kline)
            self._state.pop_identity(kline.signature)
            self._promote(kline)
            return

        if not self._state.in_frame(kline):
            return
        self._state.unframe(kline)
        if is_unknown(kline):
            self._state.pop_identity(kline.signature)
        else:
            self._promote(kline)

    # ── Cogitation ───────────────────────────────────────────────────

    def cogitate(self) -> list[KValue]:
        """One LIFO pass over the work-list: ask, countersign, propose, or ground.

        Per entry, in priority order: an identity becomes an S4 ask; a
        countersignable entry takes the S3 path and eventually grounds; a misfit
        takes the S2 path. a structurally-S1 entry is promoted (grounded).
        Entries that match no path persist for a later turn.
        """
        batch: list[KValue] = []

        for idx in range(len(self._state.work_list) - 1, -1, -1):
            kline = self._state.work_list[idx]

            if is_unknown(kline):
                del self._state.work_list[idx]
                batch.append(KValue(KLine(kline.signature, []), SIG_S4))

            elif self._state.is_countersignable(kline):
                pairings = self._countersignature_proposals(kline)
                if pairings:
                    batch.extend(pairings)
                else:
                    # All pairings resolved: the countersignature is complete.
                    del self._state.work_list[idx]
                    self._promote(kline)

            elif is_misfit(kline, self._state.signifier):
                batch.extend(
                    self._misfit.propose(kline, self._promote)
                )

            elif self._state._is_groundable(kline):
                del self._state.work_list[idx]
                self._promote(kline)

        return batch

    # ── Grounding ────────────────────────────────────────────────────

    def _promote(self, kline: KLine) -> None:
        """Ground ``kline`` at S1, then cascade any node-resolution it unblocks.

        A grounding may make other work-list entries groundable (an identity
        whose signature just landed, a canon whose nodes are now all seen, a
        relationship whose reciprocal just grounded). Cascade until fixed point.

        """
        self._ground(kline)
        changed = True
        while changed:
            changed = False
            for i, entry in enumerate(self._state.work_list):
                if self._state._is_groundable(entry):
                    del self._state.work_list[i]
                    self._ground(entry)
                    changed = True
                    break

    def _ground(self, kline: KLine, countersigning: bool = False) -> None:
        """Record that K grounded ``kline`` and observe it at S1.

        Idempotent on nodes. Grounding a single-node relationship whose two
        operands have seen canons also grounds its reciprocal (both directions
        end up at S1); ``countersigning`` guards against recursing on that mirror.
        """
        bucket = self._state.grounded.setdefault(kline.signature, [])
        if any(existing.nodes == kline.nodes for existing in bucket):
            return
        bucket.append(kline)
        self.observations.append(KValue(kline, SIG_S1))
        if not countersigning and self._state.is_countersignable(kline):
            reciprocal = KLine(self._state.signifier.signature_of(kline.nodes), [kline.signature])
            self._ground(reciprocal, countersigning=True)

    # ── Frame (emission memory) ──────────────────────────────────────

    def finish(self, batch: list[KValue]) -> tuple[list[KValue], list[KValue]]:
        """Keep only genuinely-new emissions (adding them to the frame) and return ``(batch, observations)``."""
        new_batch = [v for v in batch if not self._state.is_framed(v.kline)]
        for value in new_batch:
            self._state.frame_kline(value.kline)
        return new_batch, self.observations

    # ── S3 path: countersignature ────────────────────────────────────

    def _countersignature_proposals(self, entry: KLine) -> list[KValue]:
        """Every unresolved operand pairing for ``entry`` as CONNOTES at S3.

        Pair the two canons' operands left-to-right at group size 1; when one
        side reaches a single node, synthesise the other's residual into one
        operand. Returns ``[]`` once every pairing is grounded — the signal
        that the countersignature is complete and the entry should ground itself.
        """
        right = entry.nodes
        assert len(right) == 1, "S3 pairings expect a single-node relationship entry"
        left_nodes = self._state.canon_nodes(entry.signature)
        right_nodes = self._state.canon_nodes(right[0])
        if left_nodes is None or right_nodes is None:
            raise NotImplementedError("S3 pairings: an operand canon is missing")

        batch: list[KValue] = []
        for lhs_sig, rhs_node, residual in self._operand_pairings(left_nodes, right_nodes):
            if self._pairing_resolved(lhs_sig, rhs_node, residual):
                continue
            head_sig = self._state.signifier.signature_of(residual) if residual else lhs_sig
            batch.append(KValue(KLine(head_sig, [rhs_node]), SIG_S3))
        return batch

    def _operand_pairings(
        self, left_nodes: list[int], right_nodes: list[int]
    ) -> list[tuple[int, int, list[int]]]:
        """Pair two canons' operands into ``(lhs_sig, rhs_node, residual)`` tuples.

        Pair left-to-right while both sides have more than one node remaining;
        when one side reaches a single node, group the other's entire residual
        into one synthesised operand (returned as ``residual``).
        """
        signifier = self._state.signifier
        plan: list[tuple[int, int, list[int]]] = []
        i = j = 0
        while i < len(left_nodes) and j < len(right_nodes):
            left_rem = len(left_nodes) - i
            right_rem = len(right_nodes) - j
            if left_rem == 1 and right_rem == 1:
                plan.append((left_nodes[i], right_nodes[j], []))
                i += 1
                j += 1
            elif left_rem == 1:
                residual = list(right_nodes[j:])
                plan.append((left_nodes[i], signifier.signature_of(residual), residual))
                break
            elif right_rem == 1:
                residual = list(left_nodes[i:])
                plan.append((signifier.signature_of(residual), right_nodes[j], residual))
                break
            else:
                plan.append((left_nodes[i], right_nodes[j], []))
                i += 1
                j += 1
        return plan

    def _pairing_resolved(self, lhs_sig: int, rhs_node: int, residual: list[int]) -> bool:
        """Is this pairing's CONNOTES proposal ``{head_sig:[rhs_node]}`` grounded?

        For a grouped residual, ``head_sig`` is synthesised from the residual;
        for a 1:1 pair it is ``lhs_sig``.
        """
        head_sig = self._state.signifier.signature_of(residual) if residual else lhs_sig
        return any(
            list(kline.nodes) == [rhs_node]
            for kline in self._state.grounded.get(head_sig, [])
        )
