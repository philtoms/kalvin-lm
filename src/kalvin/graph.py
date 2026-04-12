"""KLine Graph — Stateless, lazy graph over KLine model data.

Each query performs a fresh BFS expansion from a signature, discovering
edges from the model at traversal time.  The graph holds no internal
state — model changes between calls are always reflected.

Significance levels are inferred from KLine node structure:

    S1  signed        single int node       (countersign / undersign)
    S2  canonized     (signature | nodes) != 0   (canonize)
    S3  connotated    (signature | nodes) == 0   (connotate)
    S4  unsigned      None or empty list     (unsigned leaf)

Depth semantics
~~~~~~~~~~~~~~~
``d`` controls how many BFS levels to materialise.  The root KLines are
always included; each additional level expands the children of the
previous level.

    d=1  →  root + children      (1 expansion step)
    d=2  →  root + children + grandchildren
    d=N  →  root + … + level N

No cycle-checking is performed: if the model contains cycles the same
KLines will appear multiple times in the result.

Usage
~~~~~
    g = KLineGraph(model)

    # basic query — root + 2 expansion levels
    for node in g.query(signature, d=3):
        print(node.level, node.path)

    # re-query from any result node
    child_nodes = g.query(result_node.sig, d=2)

    # model changes are visible immediately
    model.add(KLine(some_sig, some_nodes))
    g.query(same_sig, d=2)   # ← sees the new entry
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kalvin.abstract import KLine, KModel, KSig

if TYPE_CHECKING:
    pass

# ── Significance-level constants ────────────────────────────────────────────

S1 = "S1"   # signed       — single int node
S2 = "S2"   # canonized    — (signature | nodes) != 0
S3 = "S3"   # connotated   — (signature | nodes) == 0
S4 = "S4"   # unsigned     — None or empty list


def infer_level(kline: KLine) -> str:
    """Infer significance level from KLine node structure."""
    if kline.nodes is None:
        return S4
    if isinstance(kline.nodes, int):
        return S1
    if isinstance(kline.nodes, list):
        if not kline.nodes:
            return S4
        combined = kline.signature
        for node in kline.nodes:
            combined |= node
        return S2 if combined != 0 else S3
    return S4


# ── QueryNode ───────────────────────────────────────────────────────────────

class QueryNode:
    """A node returned by :py:meth:`KLineGraph.query`.

    Carries its own traversal path and significance level so it can be
    used directly as the input to a subsequent query.

    Attributes:
        sig:      KLine signature  (use as ``query(node.sig, …)``)
        level:    Significance level  (S1 / S2 / S3 / S4)
        nodes:    Child node values from the KLine
        path:     Traversal path from the original query root to this node
        dbg_text: Optional debug string from the source KLine
    """

    __slots__ = ("sig", "level", "nodes", "path", "dbg_text")

    def __init__(
        self,
        sig: KSig,
        level: str,
        nodes: Any,
        path: list[KSig],
        dbg_text: str = "",
    ):
        self.sig = sig
        self.level = level
        self.nodes = nodes
        self.path = list(path)
        self.dbg_text = dbg_text

    # -- conversions --

    def as_kline(self) -> KLine:
        """Reconstruct a KLine from this node."""
        return KLine(signature=self.sig, nodes=self.nodes, dbg_text=self.dbg_text)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict representation for inspection / serialisation."""
        return {
            "sig": self.sig,
            "level": self.level,
            "nodes": self.nodes,
            "path": self.path,
        }

    # -- dunder --

    def __repr__(self) -> str:
        text = self.dbg_text[:50] if self.dbg_text else ""
        pfx = f"{text} " if text else ""
        return (
            f"QueryNode({pfx}sig={self.sig:#x}, "
            f"level={self.level}, nodes={self.nodes!r}, "
            f"pathlen={len(self.path)})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QueryNode):
            return NotImplemented
        return (
            self.sig == other.sig
            and self.level == other.level
            and self.nodes == other.nodes
            and self.path == other.path
        )


# ── KLineGraph ──────────────────────────────────────────────────────────────

class KLineGraph:
    """Stateless, lazy graph over KLine model data.

    Parameters
    ----------
    model:
        The underlying KModel.  The graph holds a reference but maintains
        **no internal state** — every :py:meth:`query` call re-reads the
        model, so mutations between calls are always visible.

    Edge discovery is purely lazy: it happens inside :py:meth:`query`
    at traversal time, not at construction time.
    """

    def __init__(self, model: KModel):
        self._model = model

    # -- public API ----------------------------------------------------------

    def query(self, sig: KSig, d: int = 1) -> list[QueryNode]:
        """BFS query from *sig*, returning nodes up to depth *d*.

        Parameters
        ----------
        sig:
            Signature to start from.
        d:
            Number of BFS levels to materialise (≥ 1).

            * ``d=1`` — root KLines only.
            * ``d=2`` — root KLines + 1 expansion step.
            * ``d=N`` — root KLines + *N−1* expansion steps.

        Returns
        -------
            list[QueryNode] in BFS order (root first, then level by level).

        Examples
        --------
        Given model ``[{C: [D]}, {D: E}, {D: [2, 3]}, {E: D}]``::

            g.query(C, 1)  # → [C]
            g.query(C, 2)  # → [C, D, D]         (expand C's children)
            g.query(C, 3)  # → [C, D, D, E]     (expand D's children)
            g.query(C, 4)  # → [C, D, D, E, D, D]  (cycle, no check)
        """
        if d < 1:
            return []

        results: list[QueryNode] = []
        # frontier entries: (KLine, traversal_path)
        frontier: list[tuple[KLine, list[KSig]]] = []

        # ── Root: seed ────────────────────────────────────────────────
        for kl in self._model.find_signed_klines(sig):
            results.append(self._make_node(kl, [sig]))
            frontier.append((kl, [sig]))

        # ── Expansion steps 1 … d-1 ───────────────────────────────────
        for _ in range(d - 1):
            next_frontier: list[tuple[KLine, list[KSig]]] = []
            for parent_kl, parent_path in frontier:
                for child_sig in parent_kl.as_node_list():
                    if child_sig == 0:
                        continue
                    for kl in self._model.find_signed_klines(child_sig):
                        child_path = parent_path + [child_sig]
                        results.append(self._make_node(kl, child_path))
                        next_frontier.append((kl, child_path))
            frontier = next_frontier
            if not frontier:
                break

        return results

    # -- internals ----------------------------------------------------------

    def _make_node(self, kl: KLine, path: list[KSig]) -> QueryNode:
        """Build a QueryNode from a KLine and its traversal path."""
        return QueryNode(
            sig=kl.signature,
            level=infer_level(kl),
            nodes=kl.nodes,
            path=path,
            dbg_text=getattr(kl, "dbg_text", ""),
        )

    def __repr__(self) -> str:
        return f"KLineGraph(klines={len(self._model)})"
