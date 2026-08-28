r"""The engine's mutable memory.

:class:`EngineState` holds three stores that mirror the kalvin memory
relations:

- **stm** — Short-Term Memory: what cogitation is currently attending to —
  incoming entries plus the ungrounded signatures and nodes their routing
  unpacked. Written by attention: whatever routing or cogitation touches
  lands here until it grounds or is asked about.
- **frame** — the outgoing kline proposals and identity requests K has emitted.
- **ltm** — grounded klines (Long-Term Memory): what Kalvin counts on.

Reads are continuous, layered access points over these stores — STM
(attention) first, then the Frame (emissions), then LTM (grounded):
:meth:`find`, :meth:`find_sig`, :meth:`findCanons`, :meth:`where`,
:meth:`sig_nodes`, and :meth:`canon_nodes` all span the layers in that
order. Store-specific predicates (``is_grounded`` = in LTM, ``is_framed``
= in Frame) keep their single-store meaning.

State is plain nodes (signature + node lists); ``dbg`` is debug-only and dropped
on save. A saved state is a grounded prior injected into an actor at
construction.

The :class:`dialogue.engine.Engine` is stateless about its own emissions; all
per-turn memory lives here, owned by the actor and mutated in place.
"""

from __future__ import annotations
from collections.abc import Iterator

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from kalvin.kline import (
    KLine,
    KNode,
    is_canon,
    is_identity,
    is_relationship,
    is_terminal,
    is_unknown,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["EngineState"]


@dataclass
class EngineState:
    """The engine's mutable memory, owned by the actor.

    - **_signifier** — the structural-significance oracle the state's queries
      dispatch through; set at construction.
    - **stm** — Short-Term Memory: what cogitation is attending to — incoming
      entries and the ungrounded signatures/nodes unpacked from them. Written
      by attention. Entries carry no significance band; dispatch is structural.
    - **ltm** — Long-Term Memory. Grounded klines, keyed by signature.
    - **frame** — klines K has previously emitted, keyed by signature. The
      fast route matches incoming S1/S4 queries against it.
    """

    _signifier: KSignifier
    stm: list[KLine] = field(default_factory=list)
    ltm: dict[KNode, list[KLine]] = field(default_factory=dict)
    frame: dict[KNode, list[KLine]] = field(default_factory=dict)
    refused: set[tuple[KNode, tuple[KNode, ...]]] = field(default_factory=set)
    _dbg_step: int = 0

    @property
    def signifier(self) -> KSignifier:
        """The structural-significance oracle this state's queries dispatch through."""
        return self._signifier

    # -- Layered read access (STM → Frame → LTM) ---------------------------

    def find(self, signature: KNode) -> KLine | None:
        """The most recent kline under ``signature`` across the layers.

        Searches STM (attention) first, then the Frame (emissions), then
        LTM (grounded). Within a bucket the last entry (most recent) wins.
        """
        for entry in reversed(self.stm):
            if entry.signature == signature:
                return entry
        for store in (self.frame, self.ltm):
            bucket = store.get(signature)
            if bucket:
                return bucket[-1]
        return None

    def find_sig(self, signature: KNode) -> list[KLine]:
        """Every kline under ``signature`` across the layers.

        All STM entries with the signature, then the Frame bucket, then the
        LTM bucket — in attention-first order.
        """
        entries = [e for e in self.stm if e.signature == signature]
        entries.extend(self.frame.get(signature, ()))
        entries.extend(self.ltm.get(signature, ()))
        return entries

    def findCanons(self, signature: KNode) -> list[KLine]:
        return [
            item
            for item in self.find_sig(signature)
            if is_canon(item, self.signifier)
        ]

    def is_grounded(self, kline: KLine) -> bool:
        """Is an isomorphic kline (same signature and nodes) in LTM?"""
        return any(
            existing.nodes == kline.nodes
            for existing in self.ltm.get(kline.signature, [])
        )

    def where(self, predicate: Callable[[KLine], bool]) -> list[KLine]:
        """All klines matching ``predicate`` across the layers, STM first."""
        matches = [kline for kline in self.stm if predicate(kline)]
        for store in (self.frame, self.ltm):
            for bucket in store.values():
                matches.extend(kline for kline in bucket if predicate(kline))
        return matches

    def sig_nodes(self, signature: KNode) -> list[KNode] | None:
        """The nodes of the first kline under ``signature`` with non-empty
        nodes, searching STM, Frame, then LTM."""
        for kline in self.find_sig(signature):
            if kline.nodes:
                return list(kline.nodes)
        return None

    def ground(self, kline: KLine, stm_idx = -1) -> bool:
        """Record ``kline`` in LTM. Idempotent on nodes.

        Returns ``True`` when a new entry was added, ``False`` when an
        isomorphic kline (same signature and nodes) was already grounded.
        """
        bucket = self.ltm.setdefault(kline.signature, [])
        if any(existing.nodes == kline.nodes for existing in bucket):
            return False

        bucket.append(kline)
        return True

    def _is_groundable(self, kline: KLine) -> bool:
        """Can ``kline`` be grounded at S1 right now?

        Universal rule: a signature grounds only once every one of its nodes
        is in LTM. An identity is the exception — it is self-referential
        (``{S:[S]}``), so its single node is itself and it grounds
        unconditionally when promoted. An unknown (``{S: []}``) never grounds.
        """
        if is_unknown(kline):
            return False
        if is_identity(kline):
            return True
        return all(node in self.ltm for node in kline.nodes)

    def _is_denoted(self, kline: KLine) -> bool:
        """Does the store already denote ``kline``'s signature?

        A cascade may only promote an entry that is self-denoting (a canon)
        or whose signature already has some grounded kline under it.
        """
        return kline.signature in self.ltm or is_canon(kline, self._signifier)

    def canon_nodes(self, signature: KNode) -> list[KNode] | None:
        """The nodes of ``signature``'s canon, searching STM, Frame, then LTM."""
        signifier = self._signifier
        for kline in self.find_sig(signature):
            if is_canon(kline, signifier):
                return list(kline.nodes)
        return None

    # -- STM (attention) ---------------------------------------------

    def add_stm(self, kline: KLine) -> None:
        """Append ``kline`` to STM — cogitation is now attending to it."""
        self.stm.append(kline)

    def remove_stm_at(self, idx: int) -> KLine | None:
        """Remove and return the STM entry at ``idx``."""
        if idx < len(self.stm):
            return self.stm.pop(idx)
        return None

    def remove_stm(self, kline: KLine) -> None:
        """Drop every STM entry matching ``kline`` by signature and nodes."""
        self.stm = [
            e for e in self.stm
            if not (e.signature == kline.signature and e.nodes == kline.nodes)
        ]

    def refuse(self, kline: KLine) -> None:
        """Record ``kline`` as rejected at S4 — not to be re-proposed."""
        self.refused.add((kline.signature, tuple(kline.nodes)))

    def is_refused(self, kline: KLine) -> bool:
        return (kline.signature, tuple(kline.nodes)) in self.refused

    def is_seen(self, signature: KNode) -> bool:
        """Has K seen ``signature`` — grounded or pending as an Unknown ask in STM?"""
        if signature in self.ltm:
            return True
        return any(
            entry.signature == signature and is_unknown(entry)
            for entry in self.stm
        )

    def is_countersignable(self, entry: KLine) -> bool:
        """Is ``entry`` a relationship whose two operands both have canons?"""
        if not is_relationship(entry):
            return False
        return (
            self.canon_nodes(entry.signature) is not None
            and self.canon_nodes(entry.nodes[0]) is not None
        )

    # -- frame (emission memory) ------------------------------------

    def is_framed(self, kline: KLine) -> bool:
        """Is an isomorphic kline in the frame?

        Terminals match by signature (any shape); everything else by exact
        nodes, as before.
        """
        bucket = self.frame.get(kline.signature, [])
        if is_terminal(kline):
            return any(is_terminal(existing) for existing in bucket)
        return any(existing.nodes == kline.nodes for existing in bucket)

    def frame_kline(self, kline: KLine) -> None:
        self.frame.setdefault(kline.signature, []).append(kline)

    def unframe(self, kline: KLine) -> None:
        bucket = self.frame.get(kline.signature)
        if not bucket:
            return
        if is_terminal(kline):
            # A terminal reply consumes the framed ask (any shape): drop every
            # terminal entry under this signature.
            kept = [k for k in bucket if not is_terminal(k)]
        else:
            kept = [
                k for k in bucket
                if not (k.signature == kline.signature and k.nodes == kline.nodes)
            ]
        if kept:
            self.frame[kline.signature] = kept
        else:
            del self.frame[kline.signature]

    # -- persistence -------------------------------------------------
    #
    # State is plain nodes (signature + node lists); ``dbg`` is debug-only and
    # dropped on save. A saved state is a grounded prior injected into an actor
    # at construction.

    def to_dict(self) -> dict:
        """A JSON-serialisable snapshot of the model (no ``dbg``)."""
        def _n(n: KNode) -> list:
            return [int(n), getattr(n, "label", "")]
        def _kl(k: KLine) -> list:
            return [_n(k.signature), [_n(n) for n in k.nodes]]
        return {
            "stm": [_kl(k) for k in self.stm],
            "ltm": {
                str(int(sig)): [_kl(k) for k in bucket]
                for sig, bucket in self.ltm.items()
            },
            "frame": {
                str(int(sig)): [_kl(k) for k in bucket]
                for sig, bucket in self.frame.items()
            },
        }

    @classmethod
    def from_dict(cls, signifier: KSignifier, data: dict) -> EngineState:
        """Rebuild a state from :meth:`to_dict` output."""
        def _n(p) -> KNode:
            if isinstance(p, list):
                return KNode(p[0], p[1]) if len(p) > 1 and p[1] else KNode(p[0])
            return KNode(p)
        def _kl(pair) -> KLine:
            sig, nodes = _n(pair[0]), pair[1]
            return KLine(sig, [_n(n) for n in nodes])
        return cls(
            signifier,
            stm=[_kl(p) for p in data.get("stm", [])],
            ltm={
                KNode(int(sig)): [_kl(k) for k in bucket]
                for sig, bucket in data.get("ltm", {}).items()
            },
            frame={
                KNode(int(sig)): [_kl(k) for k in bucket]
                for sig, bucket in data.get("frame", {}).items()
            },
        )

    def save(self, path: str | Path) -> None:
        """Write the state snapshot to ``path`` (JSON). Creates parent dirs."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, signifier: KSignifier, path: str | Path) -> EngineState:
        """Load a state snapshot from ``path`` (JSON)."""
        return cls.from_dict(signifier, json.loads(Path(path).read_text()))
