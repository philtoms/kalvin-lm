"""KLine Graph — Stateless, lazy graph over KLine model data.

Updated to work with the new KLine API (nodes always a list, literal flag).
"""

from __future__ import annotations

from typing import Any

from kalvin.kline import KLine, KSig


# ── QueryNode ───────────────────────────────────────────────────────────────

class QueryNode:
    """A node returned by graph queries."""

    __slots__ = ("sig", "nodes", "path", "dbg_text")

    def __init__(
        self,
        sig: KSig,
        nodes: list[int],
        path: list[KSig],
        dbg_text: str = "",
    ):
        self.sig = sig
        self.nodes = list(nodes)
        self.path = list(path)
        self.dbg_text = dbg_text

    def as_kline(self) -> KLine:
        return KLine(signature=self.sig, nodes=self.nodes, dbg_text=self.dbg_text)

    def to_dict(self) -> dict[str, Any]:
        return {"sig": self.sig, "nodes": self.nodes, "path": self.path}

    def __repr__(self) -> str:
        text = self.dbg_text[:50] if self.dbg_text else ""
        pfx = f"{text} " if text else ""
        return f"QueryNode({pfx}sig={self.sig:#x}, nodes={self.nodes!r}, pathlen={len(self.path)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QueryNode):
            return NotImplemented
        return self.sig == other.sig and self.nodes == other.nodes and self.path == other.path


# ── KLineGraph ──────────────────────────────────────────────────────────────

class KLineGraph:
    """Stateless, lazy graph over KLine model data."""

    def __init__(self, model: Any):
        self._model = model

    def query(self, sig: KSig, d: int = 1) -> list[QueryNode]:
        """BFS query from *sig*, returning nodes up to depth *d*."""
        if d < 1:
            return []

        results: list[QueryNode] = []
        frontier: list[tuple[KLine, list[KSig]]] = []

        for kl in self._model.find_all(sig):
            results.append(self._make_node(kl, [sig]))
            frontier.append((kl, [sig]))

        for _ in range(d - 1):
            next_frontier: list[tuple[KLine, list[KSig]]] = []
            for parent_kl, parent_path in frontier:
                for child_sig in parent_kl.nodes:
                    if child_sig == 0:
                        continue
                    for kl in self._model.find_all(child_sig):
                        child_path = parent_path + [child_sig]
                        results.append(self._make_node(kl, child_path))
                        next_frontier.append((kl, child_path))
            frontier = next_frontier
            if not frontier:
                break

        return results

    def _make_node(self, kl: KLine, path: list[KSig]) -> QueryNode:
        return QueryNode(
            sig=kl.signature,
            nodes=kl.nodes,
            path=path,
            dbg_text=getattr(kl, "dbg_text", ""),
        )

    def __repr__(self) -> str:
        return f"KLineGraph(klines={len(self._model)})"
