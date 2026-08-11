r"""The engine's mutable memory.

:class:`EngineState` holds four stores that mirror the original kalvin memory
tiers:

- **work_list** — the cogitator queue: incoming entries plus the ungrounded
  signatures and nodes their routing unpacked.
- **ltm** — ratified klines (Long-Term Memory).
- **frame** — the outgoing kline proposals and identity requests K has emitted.
- **stm** — Short-Term Memory: reserved for the expansion strategies' exclusive
  use. Not wired into any logic; maintained independently of the other stores.

State is plain ints (signature + node lists); ``dbg`` is debug-only and dropped
on save. A saved state is a ratified prior injected into an actor at
construction.

The :class:`dialogue.engine.Engine` is stateless about its own emissions; all
per-turn memory lives here, owned by the actor and mutated in place.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from kalvin.kline import (
    KLine,
    is_canon,
    is_identity,
    is_relationship,
    is_terminal,
    is_unknown,
    sig_level,
)
from kalvin.stm import STM

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["EngineState"]


@dataclass
class EngineState:
    """The engine's mutable memory, owned by the actor.

    - **_signifier** — the structural-significance oracle the state's queries
      dispatch through; set at construction.
    - **work_list** — the cogitator queue: incoming entries and the ungrounded
      signatures/nodes unpacked from them. Entries carry no significance band;
      dispatch is structural.
    - **ltm** — ratified klines, keyed by signature.
    - **frame** — klines K has previously emitted, keyed by signature. The
      fast route matches incoming S1/S4 queries against it.
    - **stm** — Short-Term Memory. Reserved for the expansion strategies; not
      wired into any logic and maintained independently of the other stores.
    """

    _signifier: KSignifier
    work_list: list[KLine] = field(default_factory=list)
    ltm: dict[int, list[KLine]] = field(default_factory=dict)
    frame: dict[int, list[KLine]] = field(default_factory=dict)
    stm: STM = field(init=False)
    _dbg_step: int = 0

    def __post_init__(self) -> None:
        self.stm = STM(signifier=self._signifier)

    @property
    def signifier(self) -> KSignifier:
        """The structural-significance oracle this state's queries dispatch through."""
        return self._signifier

    # -- LTM queries -------------------------------------------------
    #
    # The graph-expansion walk (``ExpandFit._expand``) reads LTM through these.
    # Named ``is_in_ltm`` (not ``ltm``) so as not to shadow the ``ltm`` field.

    def find(self, signature: int) -> KLine | None:
        """The last ratified kline under ``signature``, or ``None``."""
        bucket = self.ltm.get(signature)
        return bucket[-1] if bucket else None

    # -- work-list (cogitator queue) ---------------------------------

    def add_work(self, kline: KLine) -> None:
        """Append ``kline`` to the work-list."""
        self.work_list.append(kline)

    def remove_work_at(self, idx: int) -> KLine:
        """Remove and return the work-list entry at ``idx``."""
        return self.work_list.pop(idx)

    def note_ratified(self, kline: KLine) -> bool:
        """Record ``kline`` in LTM. Idempotent on nodes.

        Returns ``True`` when a new entry was added, ``False`` when an
        isomorphic kline (same signature and nodes) was already ratified.
        """
        bucket = self.ltm.setdefault(kline.signature, [])
        if any(existing.nodes == kline.nodes for existing in bucket):
            return False
        bucket.append(kline)
        return True

    def _is_groundable(self, kline: KLine) -> bool:
        """Can ``kline`` be ratified at S1 right now?

        Universal rule: a signature grounds only once every one of its nodes
        is in LTM. An identity is the exception — it is self-referential
        (``{S:[S]}``), so its single node is itself and it grounds
        unconditionally when promoted.
        """
        if is_identity(kline):
            return True
        return all(node in self.ltm for node in kline.nodes)

    def is_in_ltm(self, kline: KLine) -> bool:
        """Is an isomorphic kline (same signature and nodes) in LTM?"""
        return any(
            existing.nodes == kline.nodes
            for existing in self.ltm.get(kline.signature, [])
        )

    def where(self, predicate: Callable[[KLine], bool]) -> list[KLine]:
        """All ratified klines matching ``predicate``."""
        return [
            kline
            for bucket in self.ltm.values()
            for kline in bucket
            if predicate(kline)
        ]

    def ltm_nodes(self, signature: int) -> list[int] | None:
        """The nodes of any ratified kline under ``signature`` with non-empty nodes."""
        for kline in self.ltm.get(signature, []):
            if kline.nodes:
                return list(kline.nodes)
        return None

    def canon_nodes(self, signature: int) -> list[int] | None:
        """The nodes of ``signature``'s canon, in LTM or the work-list."""
        signifier = self._signifier
        for kline in self.ltm.get(signature, []):
            if is_canon(kline, signifier):
                return list(kline.nodes)
        for entry in self.work_list:
            if entry.signature == signature and is_canon(entry, signifier):
                return list(entry.nodes)
        return None

    def is_seen(self, signature: int) -> bool:
        """Has K seen ``signature`` — ratified or pending as an Unknown ask?"""
        if signature in self.ltm:
            return True
        return any(
            entry.signature == signature and is_unknown(entry)
            for entry in self.work_list
        )

    def signature_seen(self, signature: int) -> bool:
        """Has ``signature`` been seen — framed, pending on the work-list, or ratified?"""
        if self.in_frame(KLine(signature, [])):
            return True
        if signature in self.ltm:
            return True
        return any(
            entry.signature == signature and is_unknown(entry)
            for entry in self.work_list
        )

    def pop_identity(self, signature: int) -> None:
        """Drop the first pending Unknown ask for ``signature`` (T answered it)."""
        for i, entry in enumerate(self.work_list):
            if entry.signature == signature and is_unknown(entry):
                self.remove_work_at(i)
                return

    def is_countersignable(self, entry: KLine) -> bool:
        """Is ``entry`` a relationship whose two operands both have canons?"""
        if not is_relationship(entry):
            return False
        return (
            self.canon_nodes(entry.signature) is not None
            and self.canon_nodes(entry.nodes[0]) is not None
        )

    # -- frame (emission memory) ------------------------------------

    def in_frame(self, kline: KLine) -> bool:
        """Is ``kline`` already in play in the frame?

        Terminals are keyed by signature alone: a terminal is one lexical
        item with multiple shapes (the S4 Unknown ask ``X:[]`` and the S1
        Identity ratifications ``X:[X]``, ``X:[COMPOUND, x, y]``), and any shape
        recognises any other — an S4 ask framed by K matches the S1 reply T
        sends back. Non-terminals match on structural significance, as before.
        """
        signifier = self._signifier
        bucket = self.frame.get(kline.signature, [])
        if is_terminal(kline):
            return any(is_terminal(framed) for framed in bucket)
        target = sig_level(kline, signifier)
        return any(
            sig_level(framed, signifier) == target
            for framed in bucket
        )

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

    def similar_fit_candidates(
        self, entry: KLine
    ) -> list[KLine]:
        """Ratified klines sharing at least one but not all node values with ``entry``,
        excluding the entry's own canon (its resolution, not a recombination ingredient)."""
        signifier = self._signifier
        entry_nodes = set(entry.nodes)
        candidates: list[KLine] = []
        for bucket in self.ltm.values():
            for kline in bucket:
                if kline is entry or not kline.nodes or entry.nodes == kline.nodes:
                    continue
                if kline.signature == entry.signature and is_canon(kline, signifier):
                    continue
                kline_nodes = set(kline.nodes)
                if entry_nodes & kline_nodes and len(kline_nodes.difference(entry_nodes)):
                    candidates.append(kline)
        return candidates

    # -- persistence -------------------------------------------------
    #
    # State is plain ints (signature + node lists); ``dbg`` is debug-only and
    # dropped on save. A saved state is a ratified prior injected into an actor
    # at construction. STM is not persisted: it is empty at session start and
    # reserved for the expansion strategies' working memory.

    def to_dict(self) -> dict:
        """A JSON-serialisable snapshot of the model (no ``dbg``)."""
        def _kl(k: KLine) -> list[int]:
            return [k.signature, list(k.nodes)]
        return {
            "work_list": [_kl(k) for k in self.work_list],
            "ltm": {
                str(sig): [_kl(k) for k in bucket]
                for sig, bucket in self.ltm.items()
            },
            "frame": {
                str(sig): [_kl(k) for k in bucket]
                for sig, bucket in self.frame.items()
            },
        }

    @classmethod
    def from_dict(cls, signifier: KSignifier, data: dict) -> EngineState:
        """Rebuild a state from :meth:`to_dict` output."""
        def _kl(pair: list[int]) -> KLine:
            sig, nodes = pair[0], pair[1]
            return KLine(sig, list(nodes))
        return cls(
            signifier,
            work_list=[_kl(p) for p in data.get("work_list", [])],
            ltm={
                int(sig): [_kl(k) for k in bucket]
                for sig, bucket in data.get("ltm", {}).items()
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
    def load(cls, signifier: KSignifier, path: str | Path) -> EngineState:
        """Load a state snapshot from ``path`` (JSON)."""
        return cls.from_dict(signifier, json.loads(Path(path).read_text()))
