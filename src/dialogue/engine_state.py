r"""The engine's mutable memory.

:class:`EngineState` holds the work-list, K's grounded model, and the emission
frame. It is plain ints (signature + node lists); ``dbg`` is debug-only and
dropped on save. A saved state is a grounded prior injected into an actor at
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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["EngineState"]


@dataclass
class EngineState:
    """The engine's mutable memory, owned by the actor.

    - **_signifier** — the structural-significance oracle the state's queries
      dispatch through; set at construction.
    - **work_list** — pending klines awaiting cogitation. Entries carry no
      significance band; dispatch is structural.
    - **grounded** — K's grounded model, keyed by signature.
    - **frame** — klines K has previously emitted, keyed by signature. The
      fast route matches incoming S1/S4 queries against it.
    """

    _signifier: KSignifier
    work_list: list[KLine] = field(default_factory=list)
    grounded: dict[int, list[KLine]] = field(default_factory=dict)
    frame: dict[int, list[KLine]] = field(default_factory=dict)
    _dbg_step: int = 0

    @property
    def signifier(self) -> KSignifier:
        """The structural-significance oracle this state's queries dispatch through."""
        return self._signifier

    # -- grounded-store queries -------------------------------------
    #
    # The graph-expansion walk (``ExpandFit._expand``) reads the grounded
    # store through these instead of a separate adapter. Named ``is_grounded``
    # (not ``grounded``) so as not to shadow the ``grounded`` dict field.

    def find(self, signature: int) -> KLine | None:
        """The last grounded kline under ``signature``, or ``None``."""
        bucket = self.grounded.get(signature)
        return bucket[-1] if bucket else None

    def _is_groundable(self, kline: KLine) -> bool:
        """Can ``kline`` be grounded at S1 right now?

        An identity whose signature is grounded; a canon whose nodes are all
        grounded; a single-node relationship whose reciprocal is grounded;
        or — the general misfit rule — a misfit whose signature and every
        node are grounded (the relationship is fully supported by what K
        already holds).
        """
        signifier = self._signifier
        if is_identity(kline):
            return kline.signature in self.grounded
        if is_canon(kline, signifier):
            return all(node in self.grounded for node in kline.nodes)
        if len(kline.nodes) == 1 and self.is_grounded(kline):
            return True
        if kline.signature in self.grounded:
            return all(node in self.grounded for node in kline.nodes)

        return False

    def is_grounded(self, kline: KLine) -> bool:
        """Is an isomorphic kline (same signature and nodes) in the grounded store?"""
        return any(
            existing.nodes == kline.nodes
            for existing in self.grounded.get(kline.signature, [])
        )

    def where(self, predicate: Callable[[KLine], bool]) -> list[KLine]:
        """All grounded klines matching ``predicate``."""
        return [
            kline
            for bucket in self.grounded.values()
            for kline in bucket
            if predicate(kline)
        ]

    def grounded_nodes(self, signature: int) -> list[int] | None:
        """The nodes of any grounded kline under ``signature`` with non-empty nodes."""
        for kline in self.grounded.get(signature, []):
            if kline.nodes:
                return list(kline.nodes)
        return None

    def canon_nodes(self, signature: int) -> list[int] | None:
        """The nodes of ``signature``'s canon, in grounded memory or the work-list."""
        signifier = self._signifier
        for kline in self.grounded.get(signature, []):
            if is_canon(kline, signifier):
                return list(kline.nodes)
        for entry in self.work_list:
            if entry.signature == signature and is_canon(entry, signifier):
                return list(entry.nodes)
        return None

    def is_seen(self, signature: int) -> bool:
        """Has K seen ``signature`` — grounded or pending as an Unknown ask?"""
        if signature in self.grounded:
            return True
        return any(
            entry.signature == signature and is_unknown(entry)
            for entry in self.work_list
        )

    def signature_seen(self, signature: int) -> bool:
        """Has ``signature`` been seen — framed, pending on the work-list, or grounded?"""
        if self.in_frame(KLine(signature, [])):
            return True
        if signature in self.grounded:
            return True
        return any(
            entry.signature == signature and is_unknown(entry)
            for entry in self.work_list
        )

    def pop_identity(self, signature: int) -> None:
        """Drop the first pending Unknown ask for ``signature`` (T answered it)."""
        for i, entry in enumerate(self.work_list):
            if entry.signature == signature and is_unknown(entry):
                del self.work_list[i]
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
        Identity groundings ``X:[X]``, ``X:[COMPOUND, x, y]``), and any shape
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
        """Grounded klines sharing at least one but not all node values with ``entry``,
        excluding the entry's own canon (its resolution, not a recombination ingredient)."""
        signifier = self._signifier
        entry_nodes = set(entry.nodes)
        candidates: list[KLine] = []
        for bucket in self.grounded.values():
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
    # dropped on save. A saved state is a grounded prior injected into an actor
    # at construction.

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
    def from_dict(cls, signifier: KSignifier, data: dict) -> EngineState:
        """Rebuild a state from :meth:`to_dict` output."""
        def _kl(pair: list[int]) -> KLine:
            sig, nodes = pair[0], pair[1]
            return KLine(sig, list(nodes))
        return cls(
            signifier,
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
    def load(cls, signifier: KSignifier, path: str | Path) -> EngineState:
        """Load a state snapshot from ``path`` (JSON)."""
        return cls.from_dict(signifier, json.loads(Path(path).read_text()))
