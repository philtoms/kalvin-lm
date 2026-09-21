r"""The engine's mutable memory.

:class:`EngineState` holds three stores that mirror the kalvin memory
relations:

- **work_list** — the entries fed to Kalvin via the slow route, plus the
  ungrounded signatures and nodes their routing unpacked. Written by
  attention: whatever routing or cogitation touches lands here until it
  grounds or is asked about.
- **frame** — the outgoing kline proposals and identity requests K has emitted.
- **ltm** — grounded klines: what Kalvin counts on.
- **stm** — working memory: the hop writes (composed correspondences)
  later hops trawl. Unratified evidence; empty at session start.

Reads are continuous, layered access points over these stores — the work
list (attention) first, then the Frame (emissions), then LTM (grounded):
:meth:`find`, :meth:`find_sig`, :meth:`findCanons`, :meth:`where`,
:meth:`sig_nodes`, and :meth:`canon_nodes` all span the layers in that
order (:meth:`where` spans STM too when flagged). Store-specific
predicates (``is_grounded`` = in LTM, ``is_framed`` = in Frame) keep their
single-store meaning.

State is plain nodes (signature + node lists); ``dbg`` is debug-only and dropped
on save. A saved state is a grounded prior injected into an actor at
construction.

The :class:`kalvin.engine.Engine` is stateless about its own emissions; all
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
    canon_key,
    is_ask,
    is_canon,
    is_canon_evidence,
    is_denotation as denotation_shape,
    is_identity,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["EngineState"]


@dataclass
class EngineState:
    """The engine's mutable memory, owned by the actor.

    - **_signifier** — the structural-significance oracle the state's queries
      dispatch through; set at construction.
    - **work_list** — the entries fed to Kalvin via the slow route. Written by attention.
    - **stm** — Working Memory. The hop writes later hops trawl.
    - **frame** — Working Memory. Framed klines, keyed by signature. 
    - **ltm** — Long-Term Memory. Grounded klines, keyed by signature.
    - **word_bits** - The word→bit mapping the persisted node values were encoded under.
    """

    _signifier: KSignifier
    work_list: list[KLine] = field(default_factory=list)
    stm: list[KLine] = field(default_factory=list)
    ltm: dict[KNode, list[KLine]] = field(default_factory=dict)
    frame: dict[KNode, list[KLine]] = field(default_factory=dict)
    refused: set[tuple[KNode, tuple[KNode, ...]]] = field(default_factory=set)
    word_bits: dict[str, int] | None = None
    _dbg_step: int = 0

    @property
    def signifier(self) -> KSignifier:
        """The structural-significance oracle this state's queries dispatch through."""
        return self._signifier

    # -- Layered read access (work list → Frame → LTM) -------------------

    def find(self, signature: KNode) -> KLine | None:
        """The most recent kline under ``signature`` across the layers.

        Searches the work list (attention) first, then the Frame (emissions),
        then LTM (grounded). Within a bucket the last entry (most recent) wins.
        """
        for entry in reversed(self.work_list):
            if entry.signature == signature:
                return entry
        for store in (self.frame, self.ltm):
            bucket = store.get(signature)
            if bucket:
                return bucket[-1]
        return None

    def find_sig(self, signature: KNode) -> list[KLine]:
        """Every kline under ``signature`` across the layers.

        All work-list entries with the signature, then the Frame bucket, then the
        LTM bucket — in attention-first order.
        """
        entries = [e for e in self.work_list if e.signature == signature]
        entries.extend(self.frame.get(signature, ()))
        entries.extend(self.ltm.get(signature, ()))
        return entries

    def find_canon(self, signature: KNode) -> KLine | None:
        for item in self.find_sig(signature):
            if is_canon_evidence(item, self.signifier):
                return item

    def where(
        self, predicate: Callable[[KLine], bool], include_stm: bool = False
    ) -> list[KLine]:
        """All klines matching ``predicate`` across the layers, work list first.

        ``include_stm`` spans STM as the last tier.
        """
        matches = [kline for kline in self.work_list if predicate(kline)]
        for store in (self.frame, self.ltm):
            for bucket in store.values():
                matches.extend(kline for kline in bucket if predicate(kline))
        if include_stm:
            matches.extend(kline for kline in self.stm if predicate(kline))
        return matches

    def sig_nodes(self, signature: KNode) -> list[KNode] | None:
        """The nodes of the first kline under ``signature`` with non-empty
        nodes, searching work list, Frame, then LTM."""
        for kline in self.find_sig(signature):
            if kline.nodes:
                return list(kline.nodes)
        return None

    def canon_nodes(self, signature: KNode) -> list[KNode] | None:
        """The nodes of ``signature``'s canon, searching work list, Frame, then LTM."""
        signifier = self._signifier
        for kline in self.find_sig(signature):
            if is_canon_evidence(kline, signifier):
                return list(kline.nodes)
        return None

    def is_countersignable(self, entry: KLine) -> bool:
        """Is ``entry`` a denotation whose two operands both have canons?"""
        if not self.is_denotation(entry):
            return False
        return (
            self.canon_nodes(entry.signature) is not None
            and self.canon_nodes(entry.nodes[0]) is not None
        )

    def is_denotation(self, kline: KLine) -> bool:
        """Case 4: a 1:1 relationship whose node shares no atom with its
        signature (uncovered — S3)."""
        return denotation_shape(kline, self._signifier)


    # -- work list (attention) ----------------------------------------

    def add_work(self, kline: KLine) -> None:
        """Append ``kline`` to the work list — cogitation is now attending to it."""
        if not any(
            e.signature == kline.signature  and e.nodes == kline.nodes
            for e in self.work_list
        ):
            self.work_list.append(kline)

    def remove_work_at(self, idx: int) -> KLine | None:
        """Remove and return the work-list entry at ``idx``."""
        if idx < len(self.work_list):
            return self.work_list.pop(idx)
        return None

    def remove_work(self, kline: KLine) -> None:
        """Drop every work-list entry matching ``kline`` by signature and nodes."""
        self.work_list = [
            e for e in self.work_list
            if not (e.signature == kline.signature and e.nodes == kline.nodes)
        ]

    # -- stm (working memory) ------------------------------------------

    def add_stm(self, kline: KLine) -> None:
        """Append ``kline`` to STM if absent — a hop write other hops may trawl."""
        if not any(
            e.signature == kline.signature and e.nodes == kline.nodes
            for e in self.stm
        ):
            self.stm.append(kline)

    def extend_stm(self, klines: list[KLine]) -> None:
        """Append every ``klines`` entry to STM via :meth:`add_stm`."""
        for kline in klines:
            self.add_stm(kline)

    def refuse(self, kline: KLine) -> None:
        """Record ``kline`` as rejected at S4 — not to be re-proposed."""
        self.refused.add((kline.signature, tuple(kline.nodes)))

    def is_refused(self, kline: KLine) -> bool:
        return (kline.signature, tuple(kline.nodes)) in self.refused

    # -- frame / ltm ------------------------------------

    def ground(self, kline: KLine, store=None) -> bool:
        """Record ``kline`` in Frame. Idempotent on nodes.

        Returns ``True`` when a new entry was added, ``False`` when an
        isomorphic kline (same signature and nodes) was already grounded.
        """
        if store is None:
            store = self.frame
        bucket = store.setdefault(kline.signature, [])
        if any(existing.nodes == kline.nodes for existing in bucket):
            return False if store is self.ltm else self.ground(kline, self.ltm)

        bucket.append(kline)
        return True

    def is_groundable(self, kline: KLine, store=None) -> bool:
        """Can ``kline`` be grounded at S1 right now?

        Canons and identities are groundable if all of their nodes are already grounded.
        Misfits are only groundable if their signatures are also grounded.
        An unknown (``{S: []}``) never grounds.
        """
        if store is None:
            store = self.frame

        if kline.signature in store:
            if all(node in store for node in kline.nodes):
                return True
        if is_identity(kline):
            return True
        if is_canon(kline, self.signifier):
            return True
        
        return False if store is self.ltm else self.is_groundable(kline, self.ltm)

    def is_answered(self, kline: KLine) -> bool:
        """An ask whose answer is grounded: a content answer — a grounded
        witness at its signature whose content differs — or, for a bare
        word-resolution ask, any grounded witness. Identities never
        answer a compound ask (tautological)."""
        if not is_ask(kline.signature):
            return False
        sig = canon_key(kline.signature)
        for store in (self.frame, self.ltm):
            for k in store.get(sig, []):
                if tuple(k.nodes) == tuple(kline.nodes):
                    continue  # its own riding canon
                if not kline.nodes and k.nodes:
                    return True  # any witness resolves a bare ask
                if self.signifier.signature_of(k.nodes) != sig:
                    return True  # a content answer beyond the signature
        return False

    def is_grounded(self, kline: KLine, store=None) -> bool:
        """Is an isomorphic kline (same signature and nodes) in Frame ?"""
        if store is None:
            store = self.frame
        if any(
            existing.nodes == kline.nodes
            for existing in store.get(kline.signature, [])
        ):
            return True

        return False if store is self.ltm else self.is_grounded(kline, self.ltm)

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
            pair = [_n(k.signature), [_n(n) for n in k.nodes]]
            return pair + [k.acq_depth] if k.acq_depth else pair
        out = {
            "work_list": [_kl(k) for k in self.work_list],
            "ltm": {
                str(int(sig)): [_kl(k) for k in bucket]
                for sig, bucket in self.ltm.items()
            },
            "frame": {
                str(int(sig)): [_kl(k) for k in bucket]
                for sig, bucket in self.frame.items()
            },
        }
        if self.word_bits is not None:
            out["word_bits"] = self.word_bits
        return out

    @classmethod
    def from_dict(cls, signifier: KSignifier, data: dict) -> EngineState:
        """Rebuild a state from :meth:`to_dict` output."""
        def _n(p) -> KNode:
            if isinstance(p, list):
                return KNode(p[0], p[1]) if len(p) > 1 and p[1] else KNode(p[0])
            return KNode(p)
        def _kl(pair) -> KLine:
            sig, nodes = _n(pair[0]), pair[1]
            acq_depth = pair[2] if len(pair) > 2 else 0
            return KLine(sig, [_n(n) for n in nodes], acq_depth=acq_depth)
        return cls(
            signifier,
            word_bits=data.get("word_bits"),
            work_list=[_kl(p) for p in data.get("work_list", [])],
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
