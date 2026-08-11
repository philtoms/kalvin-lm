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

from kalvin.kline import KLine, is_canon

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["EngineState"]


@dataclass
class EngineState:
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

    # -- grounded-store queries -------------------------------------
    #
    # The graph-expansion walk (``ExpandFit._expand``) reads the grounded
    # store through these instead of a separate adapter. Named ``is_grounded``
    # (not ``grounded``) so as not to shadow the ``grounded`` dict field.

    def find(self, signature: int) -> KLine | None:
        """The last grounded kline under ``signature``, or ``None``."""
        bucket = self.grounded.get(signature)
        return bucket[-1] if bucket else None

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

    def similar_fit_candidates(
        self, signifier: KSignifier, entry: KLine
    ) -> list[KLine]:
        """Grounded klines sharing at least one but not all node values with ``entry``,
        excluding the entry's own canon (its resolution, not a recombination ingredient)."""
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
    def from_dict(cls, data: dict) -> EngineState:
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
    def load(cls, path: str | Path) -> EngineState:
        """Load a state snapshot from ``path`` (JSON)."""
        return cls.from_dict(json.loads(Path(path).read_text()))
