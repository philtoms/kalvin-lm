r"""The engine's mutable memory.

:class:`EngineState` holds three stores that mirror the kalvin memory
relations:

- **stm** — Short-Term Memory: what cogitation is currently attending to —
  incoming entries plus the ungrounded signatures and nodes their routing
  unpacked. Written by attention: whatever routing or cogitation touches
  lands here until it grounds or is asked about.
- **ltm** — grounded klines (Long-Term Memory): what Kalvin counts on.
- **frame** — the outgoing kline proposals and identity requests K has emitted.

State is plain ints (signature + node lists); ``dbg`` is debug-only and dropped
on save. A saved state is a grounded prior injected into an actor at
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
    - **stm** — Short-Term Memory: what cogitation is attending to — incoming
      entries and the ungrounded signatures/nodes unpacked from them. Written
      by attention. Entries carry no significance band; dispatch is structural.
    - **ltm** — Long-Term Memory. Grounded klines, keyed by signature.
    - **frame** — klines K has previously emitted, keyed by signature. The
      fast route matches incoming S1/S4 queries against it.
    """

    _signifier: KSignifier
    stm: list[KLine] = field(default_factory=list)
    ltm: dict[int, list[KLine]] = field(default_factory=dict)
    frame: dict[int, list[KLine]] = field(default_factory=dict)
    refused: set[tuple[int, tuple[int, ...]]] = field(default_factory=set)
    _dbg_step: int = 0

    @property
    def signifier(self) -> KSignifier:
        """The structural-significance oracle this state's queries dispatch through."""
        return self._signifier

    # -- LTM (grounded) memory -------------------------------------------------

    def find(self, signature: int) -> KLine | None:
        """The last grounded kline under ``signature``, or ``None``."""
        bucket = self.ltm.get(signature)
        return bucket[-1] if bucket else None

    def is_grounded(self, kline: KLine) -> bool:
        """Is an isomorphic kline (same signature and nodes) in LTM?"""
        return any(
            existing.nodes == kline.nodes
            for existing in self.ltm.get(kline.signature, [])
        )

    def where(self, predicate: Callable[[KLine], bool]) -> list[KLine]:
        """All grounded klines matching ``predicate``."""
        return [
            kline
            for bucket in self.ltm.values()
            for kline in bucket
            if predicate(kline)
        ]

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

    def ltm_nodes(self, signature: int) -> list[int] | None:
        """The nodes of any grounded kline under ``signature`` with non-empty nodes."""
        for kline in self.ltm.get(signature, []):
            if kline.nodes:
                return list(kline.nodes)
        return None

    def canon_nodes(self, signature: int) -> list[int] | None:
        """The nodes of ``signature``'s canon, in LTM or STM."""
        signifier = self._signifier
        for kline in self.ltm.get(signature, []):
            if is_canon(kline, signifier):
                return list(kline.nodes)
        for entry in self.stm:
            if entry.signature == signature and is_canon(entry, signifier):
                return list(entry.nodes)
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

    def is_seen(self, signature: int) -> bool:
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
    # State is plain ints (signature + node lists); ``dbg`` is debug-only and
    # dropped on save. A saved state is a grounded prior injected into an actor
    # at construction.

    def to_dict(self) -> dict:
        """A JSON-serialisable snapshot of the model (no ``dbg``)."""
        def _kl(k: KLine) -> list[int]:
            return [k.signature, list(k.nodes)]
        return {
            "stm": [_kl(k) for k in self.stm],
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
            stm=[_kl(p) for p in data.get("stm", [])],
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
