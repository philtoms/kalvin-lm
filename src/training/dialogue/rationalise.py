r"""The rationalising engine — the pure-logic core of a rationalising trainee.

A :class:`Rationaliser` derives one turn from ``(state, incoming)`` and returns
``(batch, observations)``:

- **batch** — dialogue emissions, speech acts K addresses to T (S4 identity
  asks, S3 connotation proposals, S2 similar-fit proposals).
- **observations** — K's internal S1 groundings this turn (same shape as the
  batch), surfaced for white-box verification. Grounding never emits into the
  dialogue.

The engine is **stateless about its own emissions**: it may re-derive a
proposal on successive turns, and the :class:`~training.dialogue.actors.RationalisingTrainee`
actor is the single deduplication point. This keeps the engine's state a pure
model of what K has grounded.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from kalvin.expand import (
    SIG_S1,
    SIG_S2,
    SIG_S3,
    SIG_S4,
    boundaries,
    structural_significance,
)
from kalvin.kline import KLine, is_canon, is_identity, is_misfit
from kalvin.kvalue import KValue

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["Rationaliser", "RationaliserState"]


@dataclass
class RationaliserState:
    """The engine's mutable memory, owned by the actor.

    - **work_list** — pending klines awaiting cogitation. Entries carry no
      significance band; dispatch is structural.
    - **grounded** — K's grounded model, keyed by signature.
    - **frame** — klines K has previously emitted, keyed by signature. The
      fast route matches incoming S1/S4 queries against it.
    """

    work_list: list[KLine] = field(default_factory=list)
    grounded: dict[int, list[KLine]] = field(default_factory=dict)
    frame: dict[int, list[KLine]] = field(default_factory=dict)
    _dbg_step: int = 0

    # -- persistence -------------------------------------------------
    #
    # State is plain ints (signature + node lists); ``dbg`` is debug-only and
    # not part of the model, so it is dropped on save. A saved state is a
    # **grounded prior**: an engine snapshot produced once (typically by
    # bootstrapping an actor as a trainee against the supervisor) and injected
    # into an actor at construction so it can lead from a populated model
    # rather than from an empty one.

    def to_dict(self) -> dict:
        """A JSON-serialisable snapshot of the model (no ``dbg``)."""
        def _kl(k: KLine) -> list[int]:
            return [k.signature, list(k.nodes)]
        return {
            "work_list": [_kl(k) for k in self.work_list],
            "grounded": {
                str(sig): [_kl(k) for k in bucket]
                for sig, bucket in self.grounded.items()
            },
            "frame": {
                str(sig): [_kl(k) for k in bucket]
                for sig, bucket in self.frame.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> RationaliserState:
        """Rebuild a state from :meth:`to_dict` output."""
        def _kl(pair: list[int]) -> KLine:
            sig, nodes = pair[0], pair[1]
            return KLine(sig, list(nodes))
        return cls(
            work_list=[_kl(p) for p in data.get("work_list", [])],
            grounded={
                int(sig): [_kl(k) for k in bucket]
                for sig, bucket in data.get("grounded", {}).items()
            },
            frame={
                int(sig): [_kl(k) for k in bucket]
                for sig, bucket in data.get("frame", {}).items()
            },
        )

    def save(self, path: str | Path) -> None:
        """Write the state snapshot to ``path`` (JSON). Creates parent dirs."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> RationaliserState:
        """Load a state snapshot from ``path`` (JSON)."""
        return cls.from_dict(json.loads(Path(path).read_text()))


class Rationaliser:
    """Derives one turn from ``(state, incoming)``.

    Holds only the signifier and the significance boundaries; all per-turn
    state lives on :class:`RationaliserState` (passed in, mutated in place).
    """

    def __init__(self, signifier: KSignifier) -> None:
        self._signifier = signifier
        self._s12, self._s23, self._s34 = boundaries()

    def rationalise(
        self, state: RationaliserState, incoming: Sequence[KValue]
    ) -> tuple[list[KValue], list[KValue]]:
        """Route every incoming query, then cogitate. Returns ``(batch, observations)``."""
        state._dbg_step += 1
        turn = _Turn(state, self._signifier, self._s12, self._s23, self._s34)
        for query in incoming:
            turn.route(query)
        return turn.finish(turn.cogitate())


class _Turn:
    """A single rationalise turn: scoped accumulators over a shared state."""

    def __init__(
        self,
        state: RationaliserState,
        signifier: KSignifier,
        s12: int,
        s23: int,
        s34: int,
    ) -> None:
        self._state = state
        self._signifier = signifier
        self._s12, self._s23, self._s34 = s12, s23, s34
        self.observations: list[KValue] = []
        # The incoming queries this turn, in arrival order, retained so
        # cogitation can reply to them. The engine always replies when it
        # can support a reply from its own state; the actor filters per role.
        self._incoming: list[KValue] = []

    # ── Routing ──────────────────────────────────────────────────────

    def route(self, query: KValue) -> None:
        """Apply one incoming query as bookkeeping; emit nothing.

        Dispatch is on the query's **subjective** significance (the producer's
        surface stamp):

        - **S1/S4 (fast route)** — match against the frame. An S1 match
          promotes the kline (grounds it at S1 and cascades); an S4 match pops
          the matching identity ask. Either way the framed kline is consumed.
          Unmatched queries are dropped.
        - **S2/S3 (slow route)** — append to the work-list, then unpack an S2
          misfit's unrecognised nodes and signature as identity asks.
        """
        self._incoming.append(query)
        if query.significance in (SIG_S1, SIG_S4):
            self._fast_route(query)
            return

        kline = query.kline
        self._state.work_list.append(kline)
        for node in kline.nodes:
            if not self._is_seen(node):
                self._state.work_list.append(KLine(node, [], kline.dbg))
        if not self._is_seen(kline.signature):
            self._state.work_list.append(KLine(kline.signature, [], kline.dbg))

    def _fast_route(self, query: KValue) -> None:
        # An S1 identity reply grounds whenever its signature has been seen
        # (framed as an ask, pending on the work-list, or already grounded) —
        # not only when an ask-shape is currently framed. Identities are
        # signature-only, and the ask may be emitted by this same turn's
        # cogitation (route-all-then-cogitate ordering), so the work-list and
        # grounded views matter as much as the frame.
        if query.significance == SIG_S1 and is_identity(query.kline):
            if not self._signature_seen(query.kline.signature):
                return
            self._unframe(query.kline)
            self._pop_identity(query.kline.signature)
            self._promote(query.kline)
            return
        if not self._in_frame(query.kline):
            return
        self._unframe(query.kline)
        if query.significance == SIG_S4:
            self._pop_identity(query.kline.signature)
        else:
            self._promote(query.kline)

    def _signature_seen(self, signature: int) -> bool:
        """Has ``signature`` been seen — framed, pending on the work-list, or grounded?"""
        if self._in_frame(KLine(signature, [])):
            return True
        if signature in self._state.grounded:
            return True
        return any(
            entry.signature == signature and is_identity(entry)
            for entry in self._state.work_list
        )

    def _pop_identity(self, signature: int) -> None:
        """Drop the first pending identity ask for ``signature`` (T answered it)."""
        for i, entry in enumerate(self._state.work_list):
            if entry.signature == signature and is_identity(entry):
                del self._state.work_list[i]
                return

    # ── Cogitation ───────────────────────────────────────────────────

    def cogitate(self) -> list[KValue]:
        """One LIFO pass over the work-list: ground, countersign, or propose.

        Per entry, in priority order: an identity becomes an S4 ask; a
        structurally-S1 entry is promoted (grounded); a countersignable entry
        takes the S3 path; a multi-node misfit takes the S2 path. The first
        non-identity proposal short-circuits the pass and is returned alone.
        Entries that match no path persist for a later turn.
        """
        batch: list[KValue] = []
        # Reply pass — the engine always replies to an incoming query when it
        # can support a reply from its own state: an S4 identity ask is
        # answered with the asked signature's canon (S2, or S1 when every node
        # is already grounded), and an S3 proposal is ratified at S1. These
        # are emissions the actor filters per role (T answers/ratifies; K asks
        # and proposes). Runs before the work-list pass so a supported reply
        # takes precedence over re-deriving the query as a fresh ask.
        batch.extend(self._replies())
        for idx in range(len(self._state.work_list) - 1, -1, -1):
            kline = self._state.work_list[idx]

            if is_identity(kline):
                batch.append(self._emit_identity(idx, kline.signature))
                continue

            if batch:
                continue

            if structural_significance(kline, self._signifier) == SIG_S1:
                del self._state.work_list[idx]
                self._promote(kline)
                continue

            if self._is_countersignable(kline):
                pairings = self._countersignature_proposals(kline)
                if pairings:
                    return pairings
                # All pairings resolved: the countersignature is complete.
                # Ground the entry itself (the canonical reciprocal); _ground
                # mirrors its reciprocal (MHALL:[SVO] → SVO:[MHALL]).
                del self._state.work_list[idx]
                self._promote(kline)
                continue

            if is_misfit(kline, self._signifier):
                batch = self._similar_fit_proposal(kline)

        return batch

    def _emit_identity(self, idx: int, signature: int) -> KValue:
        """Emit ``{signature: []}`` at S4 and pop the entry."""
        del self._state.work_list[idx]
        return KValue(KLine(signature, []), SIG_S4)

    # ── Replies (engine answers/ratifies from its own state) ────────

    def _replies(self) -> list[KValue]:
        """Emissions that answer the incoming queries from the engine's state.

        Two reply families, both earned (no oracle):

        - **S4 identity ask** ``{X: []}`` — answer with ``X``'s canon if the
          engine knows it: S1 when every canon node is already grounded, else
          S2 (some part still unknown). An asked signature with no known canon
          is left for the work-list pass to (re-)emit as a fresh S4 ask.
        - **S3 proposal** ``{X: [n]}`` — ratify it at S1. The engine supports
          the pairing (K offered it; the reciprocal is now grounded).
        """
        replies: list[KValue] = []
        for query in self._incoming:
            if query.significance == SIG_S4:
                reply = self._reply_identity_ask(query.kline.signature)
                if reply is not None:
                    replies.append(reply)
            elif query.significance == SIG_S3:
                replies.append(KValue(query.kline, SIG_S1))
        return replies

    def _reply_identity_ask(self, signature: int) -> KValue | None:
        """Answer an S4 ask for ``signature`` from the engine's own state.

        An identity ask is answered only when the engine has a genuine
        grounding for the signature: a **canon** (teach its parts — S1 when
        every node is grounded, else S2) or a **compound identity** (the
        subword grounding that decodes back into text, at S1). A signature
        merely seen (e.g. as a node) is not enough — its reply shape is the
        author's to choose (a CONNOTES gloss, a pedagogical S2), so the ask
        is left unanswered for the actor's other paths. Returns None when the
        engine has no canon or compound for ``signature``.
        """
        # Canon — teach the parts. Significance is structural: S1 when every
        # node is grounded, else S2. This is the engine's bookkeeping, not a
        # protocol choice: whether a canon is *proposed* at S2 (a trainer
        # speech act, to direct the listener to cogitate over its nodes) is a
        # kline-level decision owned by the actor, not the engine.
        nodes = self._canon_nodes(signature)
        if nodes:
            kline = KLine(signature, list(nodes))
            sig = SIG_S1 if all(n in self._state.grounded for n in nodes) else SIG_S2
            return KValue(kline, sig)
        # Compound identity — the text-recoverable grounding.
        for kline in self._state.grounded.get(signature, []):
            if is_identity(kline) and kline.nodes:
                return KValue(kline, SIG_S1)
        return None

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
                if self._is_groundable(entry):
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
        if not countersigning and self._is_countersignable(kline):
            reciprocal = KLine(self._signifier.make_signature(kline.nodes), [kline.signature])
            self._ground(reciprocal, countersigning=True)

    def _is_groundable(self, kline: KLine) -> bool:
        """Can ``kline`` be grounded at S1 right now?

        An identity whose signature is grounded; a canon whose nodes are all
        grounded; or a single-node relationship whose reciprocal is grounded.
        Multi-node misfits never ground here — they propose.
        """
        if is_identity(kline):
            return kline.signature in self._state.grounded
        if is_canon(kline, self._signifier):
            return all(node in self._state.grounded for node in kline.nodes)
        if len(kline.nodes) != 1:
            return False
        return self._is_grounded(KLine(kline.nodes[0], [kline.signature]))

    def _is_grounded(self, kline: KLine) -> bool:
        """Is an isomorphic kline (same signature and nodes) in grounded memory?"""
        return any(
            existing.nodes == kline.nodes
            for existing in self._state.grounded.get(kline.signature, [])
        )

    def _is_seen(self, signature: int) -> bool:
        """Has K seen ``signature`` — grounded or pending as an identity?"""
        if signature in self._state.grounded:
            return True
        return any(
            entry.signature == signature and is_identity(entry)
            for entry in self._state.work_list
        )

    def _is_countersignable(self, entry: KLine) -> bool:
        """Is ``entry`` a single-node relationship whose two operands both have canons?"""
        if is_identity(entry) or len(entry.nodes) != 1:
            return False
        return (
            self._canon_nodes(entry.signature) is not None
            and self._canon_nodes(entry.nodes[0]) is not None
        )

    # ── Frame (emission memory) ──────────────────────────────────────

    def finish(self, batch: list[KValue]) -> tuple[list[KValue], list[KValue]]:
        """Keep only genuinely-new emissions (adding them to the frame) and return ``(batch, observations)``."""
        new_batch = [v for v in batch if not self._is_framed(v.kline)]
        for value in new_batch:
            self._frame(value.kline)
        return new_batch, self.observations

    def _in_frame(self, kline: KLine) -> bool:
        """Is ``kline`` already in play in the frame?

        Identities are keyed by signature alone: an identity is one lexical
        item with multiple shapes (the S4 ask ``X:[]`` and the S1 groundings
        ``X:[X]``, ``X:[COMPOUND, x, y]``), and any shape recognises any
        other — an S4 ask framed by K matches the S1 reply T sends back.
        Non-identities match on structural significance, as before.
        """
        bucket = self._state.frame.get(kline.signature, [])
        if is_identity(kline):
            return any(is_identity(framed) for framed in bucket)
        target = structural_significance(kline, self._signifier)
        return any(
            structural_significance(framed, self._signifier) == target
            for framed in bucket
        )

    def _is_framed(self, kline: KLine) -> bool:
        """Is an isomorphic kline in the frame?

        Identities match by signature (any shape); everything else by exact
        nodes, as before.
        """
        bucket = self._state.frame.get(kline.signature, [])
        if is_identity(kline):
            return any(is_identity(existing) for existing in bucket)
        return any(existing.nodes == kline.nodes for existing in bucket)

    def _frame(self, kline: KLine) -> None:
        self._state.frame.setdefault(kline.signature, []).append(kline)

    def _unframe(self, kline: KLine) -> None:
        bucket = self._state.frame.get(kline.signature)
        if not bucket:
            return
        if is_identity(kline):
            # An identity reply consumes the framed ask (any shape): drop every
            # identity entry under this signature.
            kept = [k for k in bucket if not is_identity(k)]
        else:
            kept = [
                k for k in bucket
                if not (k.signature == kline.signature and k.nodes == kline.nodes)
            ]
        if kept:
            self._state.frame[kline.signature] = kept
        else:
            del self._state.frame[kline.signature]

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
        left_nodes = self._canon_nodes(entry.signature)
        right_nodes = self._canon_nodes(right[0])
        if left_nodes is None or right_nodes is None:
            raise NotImplementedError("S3 pairings: an operand canon is missing")

        batch: list[KValue] = []
        for lhs_sig, rhs_node, residual in self._operand_pairings(left_nodes, right_nodes):
            if self._pairing_resolved(lhs_sig, rhs_node, residual):
                continue
            head_sig = self._signifier.make_signature(residual) if residual else lhs_sig
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
                plan.append((left_nodes[i], self._signifier.make_signature(residual), residual))
                break
            elif right_rem == 1:
                residual = list(left_nodes[i:])
                plan.append((self._signifier.make_signature(residual), right_nodes[j], residual))
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
        head_sig = self._signifier.make_signature(residual) if residual else lhs_sig
        return any(
            list(kline.nodes) == [rhs_node]
            for kline in self._state.grounded.get(head_sig, [])
        )

    def _canon_nodes(self, signature: int) -> list[int] | None:
        """The nodes of ``signature``'s canon, in grounded memory or the work-list."""
        for kline in self._state.grounded.get(signature, []):
            if is_canon(kline, self._signifier):
                return list(kline.nodes)
        for entry in self._state.work_list:
            if entry.signature == signature and is_canon(entry, self._signifier):
                return list(entry.nodes)
        return None

    # ── S2 path: similar-fit proposal ────────────────────────────────

    def _similar_fit_proposal(self, entry: KLine) -> list[KValue]:
        """Shape one S2 proposal for a misfit ``entry`` by recombining grounded klines.

        Two rules in preference order: (1) **node-expansion** — replace each
        node that is a grounded kline's signature with that kline's nodes;
        (2) **node-graft** — fold in each grounded kline sharing a node value
        with the entry, resolving the accumulated target against it. Every
        substituted node comes from a grounded kline (no invention). The entry
        persists in the work-list until ratified.
        """
        target = self._expand_nodes(list(entry.nodes))
        for candidate in self._similar_fit_candidates(entry):
            core = self._resolve_against(target, list(candidate.nodes))
            if core:
                target = core + [n for n in candidate.nodes if n not in core]

        proposal = KLine(entry.signature, target)
        if self._is_grounded(proposal):
            return []
        return [KValue(proposal, SIG_S2)]

    def _expand_nodes(self, target: list[int]) -> list[int]:
        """Rule 1 — replace each node that is a grounded kline's signature with its nodes."""
        expanded: list[int] = []
        for node in target:
            sub = self._grounded_nodes(node)
            expanded.extend(sub if sub is not None else [node])
        return expanded

    def _resolve_against(self, target: list[int], candidate_nodes: list[int]) -> list[int]:
        """Rule 2 — the portion of ``target`` that resolves into ``candidate_nodes``.

        A target node resolves if it is in the candidate directly, or via a
        grounded kline whose signature is in the candidate. Unresolvable nodes
        drop out (open slots the caller fills from the candidate's surplus).
        Iterates to fixed point.
        """
        candidate_set = set(candidate_nodes)
        core: list[int] = []
        remaining = list(target)
        while True:
            direct = [n for n in remaining if n in candidate_set]
            failed = [n for n in remaining if n not in candidate_set]
            if not failed:
                return core + direct
            resolved = self._cover_with_groundeds(failed)
            newly_matched = [n for n in resolved if n in candidate_set]
            core.extend(direct)
            core.extend(newly_matched)
            if not newly_matched:
                return core
            remaining = [n for n in resolved if n not in candidate_set]

    def _cover_with_groundeds(self, failed: list[int]) -> list[int]:
        """Maximally cover ``failed`` with disjoint grounded-kline node-sets.

        Each coverable subset is replaced by its kline's signature; uncoverable
        leftovers are passed through. Greedy is insufficient (a larger kline may
        block two smaller ones covering more), so this searches for a maximal
        disjoint cover.
        """
        failed_set = set(failed)
        covers: list[tuple[tuple[int, ...], int]] = []
        seen_sigs: set[int] = set()
        for bucket in self._state.grounded.values():
            for kline in bucket:
                if kline.signature in seen_sigs or not kline.nodes:
                    continue
                if set(kline.nodes).issubset(failed_set):
                    covers.append((tuple(kline.nodes), kline.signature))
                    seen_sigs.add(kline.signature)

        best = self._max_disjoint_cover(covers)
        if not best:
            return list(failed)
        covered: set[int] = set()
        resolved: list[int] = []
        for canon_nodes, canon_sig in best:
            resolved.append(canon_sig)
            covered.update(canon_nodes)
        resolved.extend(n for n in failed if n not in covered)
        return resolved

    @staticmethod
    def _max_disjoint_cover(
        covers: list[tuple[tuple[int, ...], int]]
    ) -> list[tuple[tuple[int, ...], int]]:
        """The disjoint subset of ``covers`` maximising total nodes covered."""
        best: list[tuple[tuple[int, ...], int]] = []
        best_covered = 0

        def _recurse(idx: int, chosen, used: set[int], covered: int) -> None:
            nonlocal best, best_covered
            if covered > best_covered:
                best_covered = covered
                best = list(chosen)
            for i in range(idx, len(covers)):
                kline_nodes, _ = covers[i]
                if used & set(kline_nodes):
                    continue
                chosen.append(covers[i])
                _recurse(i + 1, chosen, used | set(kline_nodes), covered + len(kline_nodes))
                chosen.pop()

        _recurse(0, [], set(), 0)
        return best

    def _similar_fit_candidates(self, entry: KLine) -> list[KLine]:
        """Grounded klines sharing a node value with ``entry``, excluding the
        entry's own canon (its resolution, not a recombination ingredient)."""
        entry_nodes = set(entry.nodes)
        candidates: list[KLine] = []
        for bucket in self._state.grounded.values():
            for kline in bucket:
                if kline is entry or not kline.nodes:
                    continue
                if kline.signature == entry.signature and is_canon(kline, self._signifier):
                    continue
                if entry_nodes & set(kline.nodes):
                    candidates.append(kline)
        return candidates

    def _grounded_nodes(self, signature: int) -> list[int] | None:
        """The nodes of any grounded kline under ``signature`` with non-empty nodes."""
        for kline in self._state.grounded.get(signature, []):
            if kline.nodes:
                return list(kline.nodes)
        return None
