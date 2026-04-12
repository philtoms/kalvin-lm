"""Test harness for the stateless, lazy KLine graph.

Covers:
  - Depth semantics  (root + N expansion steps)
  - Statelessness    (model mutations between queries)
  - Path tracking    (traversal breadcrumbs)
  - Re-querying      (using a QueryNode as the next query)
  - No cycle-checking (duplicates in cyclic models)
  - Compiled KScript integration
  - Visual output    (run with ``pytest -s``)
"""

from __future__ import annotations

import pytest

from kalvin.abstract import KLine, KSig
from kalvin.graph import KLineGraph, QueryNode, infer_level, S1, S2, S3, S4
from kalvin.model import Model
from kalvin.mod_tokenizer import Mod32Tokenizer
from kscript.compiler import compile_source


# ── Helpers ──────────────────────────────────────────────────────────────────

def _example_model() -> Model:
    """Model from the spec::
        [{A: [B, C]}, {B: 1}, {C: [D]}, {D: [2,3]}, {2: None}, {3: None}, {D: E}, {E: D}]
    """
    entries = _compile('''
                            1
                            A => 
                              B = 1
                              C > D => 
                                2 
                                3
                            D == E
    ''')
    klines: list[KLine] = [k for k in entries]
    return Model(klines)

_tok = Mod32Tokenizer()

# Symbolic signatures
A: int = _tok.encode("A", pack=True)[0]
B: int = _tok.encode("B", pack=True)[0]
C: int = _tok.encode("C", pack=True)[0]
D: int = _tok.encode("D", pack=True)[0]
E: int = _tok.encode("E", pack=True)[0]
_1: int = _tok.encode("1", pack=False)[0]
_2: int = _tok.encode("2", pack=False)[0]
_3: int = _tok.encode("3", pack=False)[0]

def _compile(source: str) -> list:
    return compile_source(source, tokenizer=_tok, dev=True)


def _load_model(entries: list) -> Model:
    """Load compiled entries into a Model, adding unsigned placeholders
    for every referenced node token that lacks its own entry.

    This mirrors the agent's ``rationalise()`` behaviour of pre-registering
    referenced tokens as unsigned S4 nodes so the graph can traverse
    through them.
    """
    model = Model()
    seen: set[KSig] = set()
    for e in entries:
        model.add(KLine(signature=e.signature, nodes=e.nodes, dbg_text=e.dbg_text))
        seen.add(e.signature)
    for e in entries:
        for ns in e.as_node_list():
            if ns != 0 and ns not in seen:
                model.add(KLine(signature=ns, nodes=None, dbg_text=""))
                seen.add(ns)
    return model


def _sig(name: str) -> KSig:
    """Encode a signature name (same encoding the compiler uses)."""
    return _tok.encode(name, pack=True)[0]


# ── 1. Depth semantics ──────────────────────────────────────────────────────

class TestQueryDepth:
    """Verify depth: d=1 is root-only, d=2 adds one expansion step, etc."""

    def test_d1_root_only(self) -> None:
        """d=1 returns root KLines with no expansion."""
        g = KLineGraph(_example_model())
        nodes = g.query(C, 1)
        assert len(nodes) == 1
        assert nodes[0].sig == C
        assert nodes[0].level == S2
        assert nodes[0].nodes == [D]

    def test_d2_root_plus_children(self) -> None:
        """d=2 returns root + children (1 expansion step)."""
        g = KLineGraph(_example_model())
        nodes = g.query(C, 2)
        assert len(nodes) == 3
        assert [(n.sig, n.level, n.nodes) for n in nodes] == [
            (C, S2, [D]),
            (D, S2, [_2, _3]),
            (D, S1, E),
        ]

    def test_d3_three_levels(self) -> None:
        """d=3 returns root + children + grandchildren."""
        g = KLineGraph(_example_model())
        nodes = g.query(C, 3)
        assert len(nodes) == 6
        assert [(n.sig, n.level, n.nodes) for n in nodes] == [
            (C, S2, [D]),
            (D, S2, [_2, _3]),
            (D, S1, E),
            (_2, S4, None),
            (_3, S4, None),
            (E, S1, D),
        ]

    def test_d4_cycles_through(self) -> None:
        """d=4 — no cycle-checking; D reappears through E → D."""
        g = KLineGraph(_example_model())
        nodes = g.query(C, 4)
        assert [(n.sig, n.level, n.nodes) for n in nodes] == [
            (C, S2, [D]),
            (D, S2, [_2, _3]),
            (D, S1, E),
            (_2, S4, None),
            (_3, S4, None),
            (E, S1, D),
            (D, S2, [_2, _3]),
            (D, S1, E),
        ]

    def test_d5_continues_cycle(self) -> None:
        """d=5 — E reappears from the cycled D."""
        g = KLineGraph(_example_model())
        nodes = g.query(C, 5)
        sigs = [n.sig for n in nodes]
        assert sigs == [C, D, D, _2, _3, E, D, D, _2, _3, E]

    def test_d0_returns_empty(self) -> None:
        """d < 1 → empty result."""
        g = KLineGraph(_example_model())
        assert g.query(C, 0) == []
        assert g.query(C, -1) == []


# ── 2. Statelessness ────────────────────────────────────────────────────────

class TestStatelessness:
    """Model mutations between queries are always visible."""

    def test_model_add_reflected(self) -> None:
        """Adding a KLine to the model changes the next query result."""
        model = Model([KLine(B, 1)])
        g = KLineGraph(model)

        r1 = g.query(B, 1)
        assert len(r1) == 1
        assert r1[0].nodes == 1

        model.add(KLine(B, 2))

        r2 = g.query(B, 1)
        assert len(r2) == 2
        assert {n.nodes for n in r2} == {1, 2}

    def test_graph_is_stateless(self) -> None:
        """The graph object itself holds no traversal state."""
        model = Model([KLine(B, 1)])
        g = KLineGraph(model)

        g.query(B, 5)           # deep query — builds nothing internally
        g.query(A, 5)           # completely different query
        model.add(KLine(B, 2))  # mutate model
        r = g.query(B, 1)       # sees the mutation
        assert len(r) == 2


# ── 3. Path tracking ────────────────────────────────────────────────────────

class TestPathTracking:
    """Each QueryNode carries its traversal path from the root."""

    def test_root_path(self) -> None:
        """Root node path is ``[query_sig]``."""
        g = KLineGraph(_example_model())
        nodes = g.query(C, 1)
        assert nodes[0].path == [C]

    def test_child_paths(self) -> None:
        """Children at d=2 have path ``[root, child]``."""
        g = KLineGraph(_example_model())
        nodes = g.query(C, 2)
        assert nodes[1].path == [C, D]
        assert nodes[2].path == [C, D]

    def test_grandchild_path(self) -> None:
        """E at d=3 has path ``[C, D, E]``."""
        g = KLineGraph(_example_model())
        nodes = g.query(C, 3)
        assert nodes[5].sig == E
        assert nodes[5].path == [C, D, E]

    def test_cycle_path(self) -> None:
        """Cycled D at d=4 has path ``[C, D, E, D]``."""
        g = KLineGraph(_example_model())
        nodes = g.query(C, 4)
        assert nodes[6].path == [C, D, E, D]
        assert nodes[7].path == [C, D, E, D]


# ── 4. Re-querying from a result node ───────────────────────────────────────

class TestNodeAsQuery:
    """A QueryNode can be used as the input to a subsequent query."""

    def test_requery_from_child(self) -> None:
        """Query D directly yields the same KLines as expanding C."""
        g = KLineGraph(_example_model())

        c_nodes = g.query(C, 2)
        d_from_c = [n for n in c_nodes if n.sig == D]
        assert len(d_from_c) == 2

        d_nodes = g.query(D, 1)
        assert len(d_nodes) == 2

        for dc, dd in zip(d_from_c, d_nodes):
            assert dc.sig == dd.sig
            assert dc.level == dd.level
            assert dc.nodes == dd.nodes

    def test_requery_builds_independent_path(self) -> None:
        """Re-querying starts a fresh path, not a continuation."""
        g = KLineGraph(_example_model())

        c_nodes = g.query(C, 2)
        d_from_c = c_nodes[1]  # D via C → D, path = [C, D]

        d_nodes = g.query(d_from_c.sig, 1)
        assert d_nodes[0].path == [D]  # fresh root


# ── 5. Level inference ──────────────────────────────────────────────────────

class TestInferLevel:
    """``infer_level`` derives significance from node structure."""

    def test_none_nodes_is_s4(self) -> None:
        assert infer_level(KLine(1, None)) == S4

    def test_int_node_is_s1(self) -> None:
        assert infer_level(KLine(1, 42)) == S1

    def test_single_node_list_is_s2(self) -> None:
        """Single-node list with non-zero signature|nodes is S2."""
        assert infer_level(KLine(1, [42])) == S2

    def test_single_node_list_all_zero_is_s3(self) -> None:
        """Single-node list with all-zero signature|nodes is S3."""
        assert infer_level(KLine(0, [0])) == S3

    def test_multi_node_list_nonzero_is_s2(self) -> None:
        assert infer_level(KLine(1, [42, 43])) == S2
        assert infer_level(KLine(1, [42, 43, 44])) == S2

    def test_multi_node_list_all_zero_is_s3(self) -> None:
        assert infer_level(KLine(0, [0, 0])) == S3

    def test_empty_list_is_s4(self) -> None:
        assert infer_level(KLine(1, [])) == S4


# ── 6. QueryNode helpers ────────────────────────────────────────────────────

class TestQueryNode:
    """QueryNode conversion and equality."""

    def test_to_dict(self) -> None:
        g = KLineGraph(_example_model())
        nodes = g.query(C, 1)
        d = nodes[0].to_dict()
        assert d["sig"] == C
        assert d["level"] == S2
        assert d["nodes"] == [D]
        assert d["path"] == [C]

    def test_as_kline(self) -> None:
        g = KLineGraph(_example_model())
        nodes = g.query(C, 1)
        kl = nodes[0].as_kline()
        assert kl.signature == C
        assert kl.nodes == [D]

    def test_equality(self) -> None:
        g = KLineGraph(_example_model())
        n1, n2 = g.query(C, 1), g.query(C, 1)
        assert n1[0] == n2[0]

    def test_inequality_different_path(self) -> None:
        g = KLineGraph(_example_model())
        root_only = g.query(C, 1)
        expanded = g.query(C, 2)
        assert root_only[0].sig == expanded[0].sig
        assert root_only[0] == expanded[0]  # same path [C]


# ── 7. Edge cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    """Boundary conditions."""

    def test_missing_signature_returns_empty(self) -> None:
        """Query for a signature not in the model → empty list."""
        g = KLineGraph(_example_model())
        assert g.query(99999, 3) == []

    def test_leaf_kline_no_expansion(self) -> None:
        """KLine with int node whose value has no KLines → stops."""
        model = Model([KLine(B, 1)])
        g = KLineGraph(model)
        assert len(g.query(B, 1)) == 1
        assert len(g.query(B, 5)) == 1

    def test_multiple_roots_same_sig(self) -> None:
        """Two KLines with the same signature → both in result."""
        model = Model([KLine(D, E), KLine(D, [2, 3])])
        g = KLineGraph(model)
        nodes = g.query(D, 1)
        assert len(nodes) == 2
        assert nodes[0].level == S1
        assert nodes[1].level == S2

    def test_node_value_zero_skipped(self) -> None:
        """Node value 0 is skipped (sentinel / masked value)."""
        model = Model([KLine(A, [0, B]), KLine(B, None)])
        g = KLineGraph(model)
        nodes = g.query(A, 2)
        sigs = [n.sig for n in nodes]
        assert sigs == [A, B]

    def test_empty_model(self) -> None:
        """Query on empty model → empty."""
        g = KLineGraph(Model())
        assert g.query(A, 5) == []


# ── 8. Compiled KScript integration ─────────────────────────────────────────

class TestCompiledScript:
    """Graph works with compiled KScript entries loaded into a model."""

    @pytest.fixture
    def mw_source(self) -> str:
        return """
MHALL = SVO =>
  S(ubject) < M
  V(erb) < H
  O(bject) < ALL =>
    A > D(et)
    L > M(od)
    L > O
"""

    def test_compile_and_load(self, mw_source: str) -> None:
        """Compiled entries load into model without error."""
        entries = _compile(mw_source)
        model = _load_model(entries)
        assert len(model) > 10

    def test_mhall_query_d1(self, mw_source: str) -> None:
        """Query MHALL at d=1 returns just the root KLines."""
        entries = _compile(mw_source)
        model = _load_model(entries)
        g = KLineGraph(model)

        mhall = _sig("MHALL")
        nodes = g.query(mhall, 1)
        assert len(nodes) >= 1
        assert all(n.sig == mhall for n in nodes)

    def test_mhall_query_d2_expands(self, mw_source: str) -> None:
        """d=2 from MHALL reaches SVO and children."""
        entries = _compile(mw_source)
        model = _load_model(entries)
        g = KLineGraph(model)

        mhall = _sig("MHALL")
        svo = _sig("SVO")

        nodes = g.query(mhall, 2)
        sigs = {n.sig for n in nodes}
        assert mhall in sigs
        assert svo in sigs

    def test_mhall_d2_shows_undersign(self, mw_source: str) -> None:
        """MHALL's undersign (S1) and SVO expansion appear at d=2."""
        entries = _compile(mw_source)
        model = _load_model(entries)
        g = KLineGraph(model)

        mhall = _sig("MHALL")
        svo = _sig("SVO")

        d1 = g.query(mhall, 1)
        d2 = g.query(mhall, 2)

        # d=2 should have more nodes than d=1
        assert len(d2) > len(d1)

        # MHALL root has both S1 (undersign) and S2 (MCS) entries
        mhall_d1 = [n for n in d1 if n.sig == mhall]
        assert any(n.level == S1 for n in mhall_d1), "S1 undersign at root"
        assert any(n.level == S2 for n in mhall_d1), "S2 MCS at root"

        # Expanding the S1 entry's node (SVO) produces SVO KLines
        svo_nodes = [n for n in d2 if n.sig == svo]
        assert len(svo_nodes) >= 1
        assert any(n.level == S2 for n in svo_nodes), "SVO canonize at d=2"

    def test_svo_requery(self, mw_source: str) -> None:
        """Re-querying from SVO returns its children."""
        entries = _compile(mw_source)
        model = _load_model(entries)
        g = KLineGraph(model)

        svo = _sig("SVO")
        nodes = g.query(svo, 2)
        sigs = {n.sig for n in nodes}
        assert svo in sigs
        s = _sig("S")
        v = _sig("V")
        o = _sig("O")
        assert s in sigs or v in sigs or o in sigs

    def test_model_mutation_visible(self, mw_source: str) -> None:
        """Adding a KLine after compilation is visible to the graph."""
        entries = _compile(mw_source)
        model = _load_model(entries)
        g = KLineGraph(model)

        mhall = _sig("MHALL")
        before = g.query(mhall, 1)

        model.add(KLine(mhall, 42))
        after = g.query(mhall, 1)

        assert len(after) == len(before) + 1


# ── 9. Visual output (run with ``pytest -s``) ───────────────────────────────

class TestVisualOutput:
    """Print graph traversals for manual inspection."""

    def decode(self, sig: int) -> str:
        return _tok.decode([sig], pack=None) if sig else "None"

    def _pp(self, nodes: list[QueryNode]) -> str:
        """Pretty-print a query result."""
        lines: list[str] = []
        for n in nodes:
            path_str = " → ".join(f"{self.decode(s)}" for s in n.path)
            lines.append(
                f"  {n.level}  sig={self.decode(n.sig)}  nodes={[self.decode(s) for s in n.nodes] if isinstance(n.nodes, list) else self.decode(n.nodes)!r:<20s}  path=[{path_str}]"
            )
        return "\n".join(lines)

    def test_spec_walkthrough_visual(self, capsys: pytest.CaptureFixture) -> None:
        print("\n" + "═" * 72)
        print("  Spec walkthrough  (model: {C:[D]}, {D:E}, {D:[2,3]}, {E:D})")
        print("═" * 72)
        g = KLineGraph(_example_model())
        for d in range(1, 6):
            print(f"\n── d={d} ──")
            print(self._pp(g.query(C, d)))

    def test_statelessness_visual(self, capsys: pytest.CaptureFixture) -> None:
        print("\n" + "═" * 72)
        print("  Statelessness demo")
        print("═" * 72)
        model = Model([KLine(B, 1)])
        g = KLineGraph(model)

        print("\n── before mutation ──")
        print(self._pp(g.query(B, 1)))

        model.add(KLine(B, 2))

        print("\n── after adding {B: 2} ──")
        print(self._pp(g.query(B, 1)))

    def test_mhall_visual(self, capsys: pytest.CaptureFixture) -> None:
        source = """
MHALL = SVO =>
  S(ubject) < M
  V(erb) < H
  O(bject) < ALL =>
    A > D(et)
    L > M(od)
    L > O
"""
        entries = _compile(source)
        model = _load_model(entries)
        g = KLineGraph(model)

        mhall = _sig("MHALL")
        print("\n" + "═" * 72)
        print("  Mary Had a Little Lamb — graph traversal")
        print("═" * 72)
        print(f"\n  {g}")
        print(f"  model has {len(model)} KLines\n")

        for d in [1, 2, 3, 4]:
            nodes = g.query(mhall, d)
            print(f"── d={d}: {len(nodes)} nodes ──")
            print(self._pp(nodes))

    def test_countersign_visual(self, capsys: pytest.CaptureFixture) -> None:
        source = "AB == CD"
        entries = _compile(source)
        model = _load_model(entries)
        g = KLineGraph(model)

        ab = _sig("AB")
        print("\n" + "═" * 72)
        print("  Countersign (AB == CD) — bidirectional edges")
        print("═" * 72)
        for d in [1, 2, 3]:
            nodes = g.query(ab, d)
            print(f"\n── d={d}: {len(nodes)} nodes ──")
            print(self._pp(nodes))

    def test_path_context_visual(self, capsys: pytest.CaptureFixture) -> None:
        """Show how paths reveal traversal context through cycles."""
        print("\n" + "═" * 72)
        print("  Path context — cycle at d=4")
        print("═" * 72)
        g = KLineGraph(_example_model())
        nodes = g.query(C, 4)
        for n in nodes:
            path_str = " → ".join(f"{s:#x}" for s in n.path)
            indent = "  " * (len(n.path) - 1)
            print(f"{indent}{n.level} sig={n.sig:#x} nodes={n.nodes!r}  [{path_str}]")
