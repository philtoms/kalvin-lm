"""One-to-one arity: `=`, `>`, `<` relate exactly one node each.

A denotation or connotation with more than one node — same-line items,
block constructs, or a mix — is a parse error. Chains (``A < B < C``)
and `=>` scaffolding remain the multi-relation forms; canon (``=>``)
keeps multiple nodes.
"""

from __future__ import annotations

import pytest

from ks.compiler import compile_source
from ks.parser import ParseError


def _decoded(source: str) -> list[str]:
    """The compiled klines' decoded forms (op + sig:[nodes])."""
    return [
        f"{kv.kline.dbg.op} | {kv.kline.dbg.decoded}"
        for kv in compile_source(source, dev=True)
    ]


class TestSameLine:
    @pytest.mark.parametrize("op", ["=", ">", "<"])
    def test_second_node_is_a_parse_error(self, op):
        with pytest.raises(ParseError):
            compile_source(f"A {op} B C\n", dev=True)

    def test_error_names_the_relation(self):
        with pytest.raises(ParseError, match="one node"):
            compile_source("A = B C\n", dev=True)

    def test_single_node_still_compiles(self):
        assert _decoded("A = B\n") == ["DENOTES | A:[B]", "UNKNOWN | B:[]"]
        assert _decoded("A > B\n") == ["CONNOTES | AB:[B]", "UNKNOWN | B:[]"]
        assert _decoded("A < B\n") == ["CONNOTES | BA:[A]"]


class TestBlockForm:
    def test_two_block_nodes_are_a_parse_error(self):
        with pytest.raises(ParseError):
            compile_source("A <\n   B\n   C\n", dev=True)
        with pytest.raises(ParseError):
            compile_source("A =\n   B\n   C\n", dev=True)

    def test_item_plus_block_node_is_a_parse_error(self):
        with pytest.raises(ParseError):
            compile_source("A < B\n   C\n", dev=True)

    def test_single_block_node_compiles(self):
        assert _decoded("A <\n   B\n") == ["CONNOTES | BA:[A]"]


class TestMultiRelationFormsStayLegal:
    def test_chain_is_two_one_to_one_connotations(self):
        assert _decoded("A < B < C\n") == [
            "CONNOTES | BA:[A]",
            "CONNOTES | CB:[B]",
        ]

    def test_denotation_chain_still_compiles(self):
        assert _decoded("A = B = C\n") == [
            "DENOTES | A:[B]",
            "DENOTES | B:[C]",
            "UNKNOWN | C:[]",
        ]

    def test_canon_stays_multi_node(self):
        # `=>` is composition, not connotation/denotation.
        assert _decoded("S => P Q\n") == [
            "CANONICALISES | S:[P, Q]",
            "UNKNOWN | Q:[]",
        ]
