"""Tests for KScript v2 compiler with new parser AST."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from kalvin.abstract import KLine
from kalvin.mod_tokenizer import Mod64Tokenizer, Mod32Tokenizer
from kalvin.significance import Int32Significance
from kscript.lexer import Lexer
from kscript.parser import Parser, ParseError, Signature, Literal, Construct, Script, KScriptFile, Block, PrimaryConstruct
from kscript.compiler import Compiler, CompiledEntry, compile_source
from kscript.decompiler import Decompiler, DecompiledEntry
from kscript.token import TokenType

# Shared significance instance for tests
_sig = Int32Significance()

# Shared tokenizer for tests
_tokenizer = Mod64Tokenizer()


def compile_test_source(source: str) -> list[CompiledEntry]:
    """Helper to compile source string to entries."""
    tokens = Lexer(source).tokenize()
    kscript_file = Parser(tokens).parse()
    return Compiler(_tokenizer, dev=True).compile(kscript_file)


def entries_to_dict(entries: list[CompiledEntry]) -> dict[str, list[str] | str | None]:
    """Convert entries to dict for easier comparison (decoded).

    WARNING: Only use when sigs are unique. For duplicate sigs, use
    entries_to_multidict() or decode entries directly by index.
    """
    result = {}
    for e in entries:
        sig, nodes = e.decode(_tokenizer)
        result[sig] = nodes
    return result


def entries_to_multidict(entries: list[CompiledEntry]) -> dict[str, list[list[str] | str | None]]:
    """Convert entries to multi-dict preserving all values per sig.

    Use when the same sig may appear multiple times with different node values.
    Returns {sig: [nodes1, nodes2, ...]} for all occurrences.
    """
    result: dict[str, list[list[str] | str | None]] = {}
    for e in entries:
        sig, nodes = e.decode(_tokenizer)
        if sig not in result:
            result[sig] = []
        result[sig].append(nodes)
    return result


def decoded_entries(entries: list[CompiledEntry]) -> list[tuple[str, list[str] | str | None]]:
    """Convert entries to list of (sig, nodes) tuples, preserving order and duplicates."""
    return [e.decode(_tokenizer) for e in entries]


# =============================================================================
# 1. Parser AST Tests (New AST Structure)
# =============================================================================

class TestParserAST:
    """Tests for the new parser AST structure."""

    def test_chain_construct(self) -> None:
        """Chain construct: A => B C <= D"""
        source = "A => B C <= D"
        tokens = Lexer(source).tokenize()
        kscript_file = Parser(tokens).parse()

        assert len(kscript_file.scripts) == 1
        script = kscript_file.scripts[0]
        assert len(script.constructs) >= 1

        # First construct should be a chain with CANONIZE_FWD
        construct = script.constructs[0]
        assert isinstance(construct.inner, list)
        assert len(construct.inner) == 1
        assert construct.inner[0].sig.id == "A"
        assert construct.chain_op == TokenType.CANONIZE_FWD

        # Right side should be a chain with CANONIZE_BWD
        right = construct.chain_right
        assert right is not None
        assert isinstance(right.inner, list)
        assert len(right.inner) >= 1
        assert right.chain_op == TokenType.CANONIZE_BWD

    def test_block_construct(self) -> None:
        """Block construct with indented subscripts."""
        source = "A =>\n  B\n  C"
        tokens = Lexer(source).tokenize()
        kscript_file = Parser(tokens).parse()

        script = kscript_file.scripts[0]
        construct = script.constructs[0]

        # Should have chain_op CANONIZE_FWD
        assert construct.chain_op == TokenType.CANONIZE_FWD
        assert construct.chain_right is not None

        # Right side should be a block with 1 construct containing 2 primaries
        right = construct.chain_right
        assert isinstance(right.inner, Block)
        assert len(right.inner.constructs) == 1
        # The single construct has 2 primaries (B and C at same indent)
        assert len(right.inner.constructs[0].inner) == 2

    def test_implicit_opening(self) -> None:
        """Implicit opening: A B <= CD"""
        source = "A B <= CD"
        tokens = Lexer(source).tokenize()
        kscript_file = Parser(tokens).parse()

        script = kscript_file.scripts[0]
        # Should parse without error
        assert len(script.constructs) >= 1

    def test_single_line_comment(self) -> None:
        """Single-line comment (...) is consumed."""
        source = "A (inline comment) => B"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        from kscript.token import TokenType
        assert TokenType.COMMENT in types
        # Comment doesn't break parsing
        kscript_file = Parser(tokens).parse()
        assert len(kscript_file.scripts[0].constructs) >= 1

    def test_multi_line_comment(self) -> None:
        """Multi-line comment (...) spans newlines."""
        source = "A => B\n(this is\na\ncomment)\nC => D"
        tokens = Lexer(source).tokenize()
        comments = [t for t in tokens if t.type.name == "COMMENT"]
        assert len(comments) == 1
        assert "this is" in comments[0].value
        assert "\n" in comments[0].value
        # Comment doesn't break parsing
        kscript_file = Parser(tokens).parse()
        script = kscript_file.scripts[0]
        sigs = [c.inner[0].sig.id for c in script.constructs]
        assert "A" in sigs
        assert "C" in sigs

    def test_nested_parens_in_comment(self) -> None:
        """Nested parens in multi-line comment are handled."""
        source = "A => B\n(outer (inner)\nstill outer)\nC => D"
        tokens = Lexer(source).tokenize()
        comments = [t for t in tokens if t.type.name == "COMMENT"]
        assert len(comments) == 1
        assert "inner" in comments[0].value
# =============================================================================

class TestEagerEmit:
    """Tests for eager emit compilation."""

    def test_eager_emit_canonize(self) -> None:
        """Eager emit for canonize: A => B C emits immediately."""
        entries = compile_test_source("A => B C")
        d = entries_to_dict(entries)
        assert d["A"] == ["B", "C"]

    def test_bwd_triggers_additional_emit(self) -> None:
        """BWD triggers additional emit: A => B C <= D."""
        entries = compile_test_source("A => B C <= D")
        md = entries_to_multidict(entries)

        # First emit: {A|S2: [B, C]}
        assert ["B", "C"] in md.get("A", [])
        # Second emit (BWD): {D|S2: [B, C]}
        assert ["B", "C"] in md.get("D", [])


# =============================================================================
# 3. MCS Expansion Tests
# =============================================================================

class TestMCSExpansion:
    """Tests for MCS expansion."""

    def test_mcs_simple_expansion(self) -> None:
        """MCS expansion: ABC emits canonization + identities."""
        entries = compile_test_source("ABC")

        # Should have 5 entries: MCS canonization + 3 identities + 1 unsigned
        assert len(entries) == 5

        md = entries_to_multidict(entries)

        # MCS canonization: {ABC: [A, B, C]}
        assert ["A", "B", "C"] in md["ABC"]

        # MCS unsigned entries
        assert md["A"][0] == None
        assert md["B"][0] == None
        assert md["C"][0] == None

    def test_mcs_in_construct_position(self) -> None:
        """MCS in construct position."""
        entries = compile_test_source("ABC => X")

        # Should have: 3 unsigned identities (first) + MCS canonization + construct + entity
        assert len(entries) >= 5

        # First 3 entries should be unsigned identities (chars emitted before compound)
        for i in range(0, 3):
            sig, nodes = entries[i].decode(_tokenizer)
            assert sig in "ABC"
            # Identities now reference themselves
            assert nodes is None

        # Fourth entry should be MCS canonization
        sig, nodes = entries[3].decode(_tokenizer)
        assert sig == "ABC"
        assert nodes == ["A", "B", "C"]

        # Fifth entry should be the construct
        sig, nodes = entries[4].decode(_tokenizer)
        assert sig == "ABC"
        assert nodes == ["X"]

    def test_no_mcs_for_single_char(self) -> None:
        """No MCS for single-char owner."""
        entries = compile_test_source("A => X")

        d = entries_to_dict(entries)
        # Only construct entry, no MCS
        assert d["A"] == ["X"]

    def test_mcs_with_countersign(self) -> None:
        """MCS expansion with countersign construct."""
        entries = compile_test_source("ABC == X")

        # Should have: 3 identities + MCS canonization + 2 countersign entries
        assert len(entries) == 6

        # Entries 0-2: Component identities (chars emitted before compound)
        for i in range(0, 3):
            sig, nodes = entries[i].decode(_tokenizer)
            assert sig in "ABC"
            # Unsigned Identities now reference themselves
            assert nodes is None

        # Entry 3: MCS canonization {ABC: [A, B, C]}
        sig, nodes = entries[3].decode(_tokenizer)
        assert sig == "ABC"
        assert nodes == ["A", "B", "C"]

        # Entries 4-5: Countersign bidirectional
        sig, nodes = entries[4].decode(_tokenizer)
        assert sig == "ABC" and nodes == "X"
        sig, nodes = entries[5].decode(_tokenizer)
        assert sig == "X" and nodes == "ABC"


# =============================================================================
# 4. Significance Level Tests
# =============================================================================

class TestSignificanceLevels:
    """Tests for significance level inference from node structure."""

    def test_countersign_significance(self) -> None:
        """Countersign produces int node (S1 when decompiled)."""
        entries = compile_test_source("A == B")

        # Check that countersign entries have int nodes
        for e in entries:
            if isinstance(e.nodes, int):
                return
        assert False, "No int-node entry found"

    def test_canonize_significance(self) -> None:
        """Canonize produces multi-node list."""
        entries = compile_test_source("AB => C D")

        # Check for multi-node list entry
        for e in entries:
            if isinstance(e.nodes, list) and len(e.nodes) > 1:
                return
        assert False, "No multi-node entry found"

    def test_connotate_significance(self) -> None:
        """Connotate produces single-node list."""
        entries = compile_test_source("A > B")

        # Check for single-node list entry
        for e in entries:
            if isinstance(e.nodes, list) and len(e.nodes) == 1:
                return
        assert False, "No single-node entry found"

    def test_undersign_significance(self) -> None:
        """Undersign: S1 unidirectional."""
        entries = compile_test_source("A = B")

        d = entries_to_dict(entries)
        assert d["A"] == "B"


# =============================================================================
# 5. Subscript Tests
# =============================================================================

class TestSubscripts:
    """Tests for subscript processing."""

    def test_subscript_attaches_to_last_cln(self) -> None:
        """Subscript attaches to last CLN."""
        source = "A =>\n  B\n  C = D"
        entries = compile_test_source(source)

        d = entries_to_dict(entries)
        # A => [B, C]
        assert d["A"] == ["B", "C"]
        # C = D (subscript)
        assert d["C"] == "D"

    def test_nested_subscript(self) -> None:
        """Nested subscript."""
        source = "A =>\n  B =>\n    C\n    D"
        entries = compile_test_source(source)

        d = entries_to_dict(entries)
        # Should have entries for A, B, C, D
        assert "A" in d
        assert "B" in d


# =============================================================================
# 6. Complex Examples
# =============================================================================

class TestComplexExamples:
    """Tests for complex examples from exploration."""

    def test_ab_arrow_a_b(self) -> None:
        """Test: AB => A B"""
        entries = compile_test_source("AB => A B")
        d = entries_to_dict(entries)

        # MCS for AB
        assert d["AB"] == ["A", "B"]
        # Entities
        assert d["A"] is None
        assert d["B"] is None

    def test_ab_double_equal_cd(self) -> None:
        """Test: AB == CD"""
        entries = compile_test_source("AB == CD")
        md = entries_to_multidict(entries)

        # MCS for AB: {AB: [A, B]}
        assert ["A", "B"] in md.get("AB", [])
        # Countersign: {AB: CD} and {CD: AB}
        assert "CD" in md.get("AB", [])
        assert "AB" in md.get("CD", [])

    def test_a_b_le_arrow_cd(self) -> None:
        """Test: A B <= CD"""
        entries = compile_test_source("A B <= CD")
        md = entries_to_multidict(entries)

        # CD binds ALL CLNs [A, B]
        assert ["A", "B"] in md.get("CD", [])
        # MCS for CD: unsigned identities reference themselves
        assert md.get("C", [""])[0] == None
        assert md.get("D", [""])[0] == None

    def test_ab_gt_c(self) -> None:
        """Test: AB > C"""
        entries = compile_test_source("AB > C")
        md = entries_to_multidict(entries)

        # MCS for AB: {AB: [A, B]}
        assert ["A", "B"] in md.get("AB", [])
        # Connotate: {AB|S3: [C]}
        assert ["C"] in md.get("AB", [])

    def test_c_lt_ab(self) -> None:
        """Test: C < AB"""
        entries = compile_test_source("C < AB")
        md = entries_to_multidict(entries)

        # MCS for AB: {AB: [A, B]}
        assert ["A", "B"] in md.get("AB", [])
        # unsigned identities reference themselves
        assert md.get("A", [""])[0] == None
        assert md.get("B", [""])[0] == None
        # BWD connotate: {AB|S3: [C]}
        assert ["C"] in md.get("AB", [])


# =============================================================================
# 7b. S2 Literal Tests
# =============================================================================

class TestS2Literals:
    """Tests for S2 literal support in constructions.

    Grammar: construct ::= block | literal | primary_construct+ (...)?

    - Literals are bare constructs (unsigned identities) with NO chain ops.
    - In inline chains (A => 1), the literal is a separate construct consumed
      as the single right-hand item.
    - In blocks, literals and primary_constructs are siblings (mixed flattening).
    - Literals CANNOT own chain ops: '1 => A' and 'A => 1 => B' are errors.
    """

    # --- Inline chain with literal right side ---

    def test_canonize_fwd_single_literal(self) -> None:
        """A => 1: literal is the right construct -> {A: [1]}."""
        entries = compile_test_source("A => 1")
        d = entries_to_dict(entries)
        assert d["A"] == "1"
        assert d["1"] is None

    def test_canonize_fwd_literal_consumes_one(self) -> None:
        """A => 1 2: literal(1) is the right construct, 2 is separate.

        Inline literal constructs consume only themselves.
        """
        entries = compile_test_source("A => 1 2")
        md = entries_to_multidict(entries)
        # A => 1 (literal construct consumed as right)
        assert "1" in md.get("A", [])
        # 2 is a separate bare literal identity
        assert None in md.get("2", [])

    def test_canonize_fwd_sig_then_literal(self) -> None:
        """A => B 1: primary_construct[B] is the right, 1 is separate.

        Inline primary_construct+ stops when it hits a literal.
        """
        entries = compile_test_source("A => B 1")
        md = entries_to_multidict(entries)
        assert ["B"] in md.get("A", [])
        assert None in md.get("B", [])
        assert None in md.get("1", [])

    def test_canonize_fwd_literal_then_sig(self) -> None:
        """A => 1 B: literal(1) is the right, B is separate.

        Inline literal is a standalone construct alternative.
        """
        entries = compile_test_source("A => 1 B")
        md = entries_to_multidict(entries)
        assert "1" in md.get("A", [])
        assert None in md.get("1", [])
        assert None in md.get("B", [])

    # --- Block: literals as siblings with constructs ---

    def test_block_mixed_literal_and_sig(self) -> None:
        """A => block(1, B): literals and sigs are siblings in blocks.

        This is the key use case: block flattening produces mixed items.
        """
        source = "A =>\n  1\n  B"
        entries = compile_test_source(source)
        d = entries_to_dict(entries)
        assert d["A"] == ["1", "B"]
        assert d["1"] is None
        assert d["B"] is None

    def test_block_all_literals(self) -> None:
        """A => block(1, 2, 3): all-literal block.

        All-literal items are decoded as a single string.
        """
        source = "A =>\n  1\n  2\n  3"
        entries = compile_test_source(source)
        d = entries_to_dict(entries)
        assert d["A"] == "123"
        assert d["1"] is None
        assert d["2"] is None
        assert d["3"] is None

    def test_block_mixed_literal_literal_sig(self) -> None:
        """A => block(1, 2, B): consecutive literals group, sigs separate.

        Mixed items: consecutive literals decode as grouped strings,
        signatures decode individually.
        """
        source = "A =>\n  1\n  2\n  B"
        entries = compile_test_source(source)
        d = entries_to_dict(entries)
        assert d["A"] == ["12", "B"]

    def test_block_nested(self) -> None:
        """Nested blocks with literals."""
        source = "A =>\n  B =>\n    1\n    C"
        entries = compile_test_source(source)
        d = entries_to_dict(entries)
        assert d["A"] == ["B"]
        assert d["B"] == ["1", "C"]

    # --- BWD / CONNOTATE_BWD with literal owner ---

    def test_canonize_bwd_literal_owner(self) -> None:
        """A <= 1: BWD with literal as right owner -> {1: [A]}."""
        entries = compile_test_source("A <= 1")
        md = entries_to_multidict(entries)
        assert ["A"] in md.get("1", [])

    def test_connotate_bwd_literal_owner(self) -> None:
        """A < 1: CONNOTATE_BWD with literal owner -> {1: [A]}."""
        entries = compile_test_source("A < 1")
        md = entries_to_multidict(entries)
        assert ["A"] in md.get("1", [])

    def test_canonize_bwd_multiple_left_literal_owner(self) -> None:
        """A B <= 1: multiple left primaries, literal owner -> {1: [A, B]}."""
        entries = compile_test_source("A B <= 1")
        md = entries_to_multidict(entries)
        assert ["A", "B"] in md.get("1", [])

    # --- Bare literal identities ---

    def test_bare_literal_identity(self) -> None:
        """Bare literal '1' emits unsigned identity."""
        entries = compile_test_source("1")
        assert len(entries) == 1
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "1"
        assert nodes is None

    def test_bare_literal_multiple_identities(self) -> None:
        """Multiple bare literals each emit unsigned identities."""
        entries = compile_test_source("1 2 3")
        assert len(entries) == 3
        sigs = [e.decode(_tokenizer)[0] for e in entries]
        assert sigs == ["1", "2", "3"]

    # --- Literal cannot own chain ops (error cases) ---

    def test_literal_cannot_own_canonize_fwd(self) -> None:
        """1 => A: literal cannot own => chain -> ParseError."""
        tokens = Lexer("1 => A").tokenize()
        with pytest.raises(ParseError):
            Parser(tokens).parse()

    def test_literal_cannot_own_canonize_bwd(self) -> None:
        """1 <= A: literal cannot own <= chain -> ParseError."""
        tokens = Lexer("1 <= A").tokenize()
        with pytest.raises(ParseError):
            Parser(tokens).parse()

    def test_literal_cannot_own_connotate_bwd(self) -> None:
        """1 < A: literal cannot own < chain -> ParseError."""
        tokens = Lexer("1 < A").tokenize()
        with pytest.raises(ParseError):
            Parser(tokens).parse()

    def test_cannot_chain_through_literal(self) -> None:
        """A => 1 => B: cannot chain through literal -> ParseError."""
        tokens = Lexer("A => 1 => B").tokenize()
        with pytest.raises(ParseError):
            Parser(tokens).parse()

    # --- Pre-existing: literal as inline op node ---

    def test_literal_with_undersign(self) -> None:
        """A = 1: undersign with literal node (pre-existing behavior)."""
        entries = compile_test_source("A = 1")
        d = entries_to_dict(entries)
        assert d["A"] == "1"

    def test_literal_with_countersign(self) -> None:
        """A == 1: countersign with literal node (pre-existing behavior).

        Note: reverse direction not emitted for literal nodes.
        """
        entries = compile_test_source("A == 1")
        md = entries_to_multidict(entries)
        assert "1" in md.get("A", [])

    def test_literal_with_connotate_fwd(self) -> None:
        """A > 1: connotate with literal node (pre-existing behavior)."""
        entries = compile_test_source("A > 1")
        d = entries_to_dict(entries)
        assert d["A"] == "1"

    def test_literal_with_quoted_string(self) -> None:
        """A => \"hello\": quoted string as literal right construct."""
        entries = compile_test_source('A => "hello"')
        d = entries_to_dict(entries)
        assert d["A"] == '"hello"'


# =============================================================================
# 8. Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Tests for backward compatibility with existing behavior."""

    def test_unsigned(self) -> None:
        """Test compiling unsigned script."""
        entries = compile_test_source("A")
        assert len(entries) == 1
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "A"
        assert nodes is None

    def test_countersign(self) -> None:
        """Test compiling countersign (bidirectional)."""
        entries = compile_test_source("A == B")
        d = entries_to_dict(entries)
        assert d["A"] == "B"
        assert d["B"] == "A"

    def test_undersign(self) -> None:
        """Test compiling undersign."""
        entries = compile_test_source("A = B")
        assert len(entries) == 1
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "A" and nodes == "B"

    def test_connotate_fwd(self) -> None:
        """Test compiling forward connotate."""
        entries = compile_test_source("A > B")
        assert len(entries) == 1
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "A" and nodes == ["B"]

    def test_connotate_bwd(self) -> None:
        """Test compiling backward connotate."""
        entries = compile_test_source("A < B")
        # Emits: {B: [A]} (connotate) and {B: None} 
        md = entries_to_multidict(entries)
        assert ["A"] in md.get("B", [])

    def test_canonize_fwd(self) -> None:
        """Test compiling forward canonize."""
        entries = compile_test_source("AB => C D")
        d = entries_to_dict(entries)
        assert d["AB"] == ["C", "D"]

    def test_literals(self) -> None:
        """Test compiling string and number literals."""
        d = entries_to_dict(compile_test_source(r'A = "hello"'))
        assert d["A"] == '"hello"'
        d = entries_to_dict(compile_test_source("A = 42"))
        assert d["A"] == "42"

    def test_multiple_constructs(self) -> None:
        """Test multiple constructs with immediate binding."""
        entries = compile_test_source("A => B => C")
        # Check S2 construct entries
        md = entries_to_multidict(entries)
        assert ["B"] in md.get("A", [])
        assert ["C"] in md.get("B", [])

    def test_subscript_as_nodes(self) -> None:
        """Test subscript signatures as nodes."""
        source = "A =>\n  B\n  C"
        entries = compile_test_source(source)
        d = entries_to_dict(entries)
        assert d["A"] == ["B", "C"]
        assert d["B"] is None
        assert d["C"] is None


# =============================================================================
# 8. Decompiler Tests
# =============================================================================

class TestDecompiler:
    """Tests for decompiling KLines back to KScript source."""

    def _roundtrip(self, source: str) -> list[DecompiledEntry]:
        """Helper: compile source, then decompile back."""
        entries = compile_test_source(source)
        decompiler = Decompiler(_tokenizer)
        return decompiler.decompile(entries)

    def _find_entry(self, entries: list, sig: str, level: str | None = None) -> dict | None:
        """Find an entry by signature name, optionally filtered by significance."""
        for e in entries:
            if e.sig == sig:
                if level is None or e.level == level:
                    return e.to_dict()
        return None

    def test_decompile_unsigned(self) -> None:
        """Decompile unsigned: A -> A"""
        result = self._roundtrip("A")
        entry = self._find_entry(result, "A")
        assert entry is not None
        assert entry["level"] == "S4"
        assert entry["nodes"] is None

    def test_decompile_undersign(self) -> None:
        """Decompile undersign: A = B -> A = B"""
        result = self._roundtrip("A = B")
        entry = self._find_entry(result, "A")
        assert entry is not None
        assert entry["level"] == "S1"
        assert entry["nodes"] == "B"

    def test_decompile_countersign(self) -> None:
        """Decompile countersign: A == B -> A == B"""
        result = self._roundtrip("A == B")
        entry = self._find_entry(result, "A")
        assert entry is not None
        assert entry["level"] == "S1"
        assert entry["nodes"] == "B"

    def test_decompile_connotate(self) -> None:
        """Decompile connotate: A > B -> A > B"""
        result = self._roundtrip("A > B")
        entry = self._find_entry(result, "A")
        assert entry is not None
        assert entry["level"] == "S2"
        assert entry["nodes"] == "B"

    def test_decompile_canonize_single(self) -> None:
        """Decompile canonize with single node: A => B."""
        result = self._roundtrip("A => B")
        entry = self._find_entry(result, "A")
        assert entry is not None
        # Single-node list with non-zero signature|nodes → S2
        assert entry["level"] == "S2"
        assert entry["nodes"] == "B"

    def test_decompile_canonize_multi(self) -> None:
        """Decompile canonize with multiple nodes: A => B C -> list of nodes."""
        result = self._roundtrip("A => B C")
        entry = self._find_entry(result, "A")
        assert entry is not None
        assert entry["level"] == "S2"
        assert entry["nodes"] == ["B", "C"]

    def test_decompile_mcs_preserves_name(self) -> None:
        """MCS signatures preserve original name using MCS entry nodes.

        This is the key test: ABC encoded with mod32 loses order/collapses,
        but the MCS entry stores [A, B, C] as nodes which we use for the name.
        """
        result = self._roundtrip("ABC")
        # Should reconstruct "ABC" from MCS nodes, not decode the packed token
        entry = self._find_entry(result, "ABC")
        assert entry is not None

    def test_decompile_mcs_in_construct(self) -> None:
        """MCS in construct position preserves name."""
        result = self._roundtrip("ABC => X")
        # Find the construct entry (single-node, non-zero sig|nodes → S2), not the MCS entry
        entry = self._find_entry(result, "ABC", level="S2")
        assert entry is not None
        assert entry["nodes"] == "X"

    def test_decompile_literal_nodes(self) -> None:
        """Decompile with literal string nodes."""
        result = self._roundtrip('A = "hello"')
        entry = self._find_entry(result, "A")
        assert entry is not None
        # String literals are stored with quotes
        assert entry["nodes"] == '"hello"'

    def test_decompile_multi_construct(self) -> None:
        """Decompile multiple constructs."""
        result = self._roundtrip("A => B => C")
        entry_a = self._find_entry(result, "A")
        assert entry_a is not None
        # Single-node chain, non-zero sig|nodes → S2
        assert entry_a["level"] == "S2"


class TestDecompilerMCS:
    """Tests specifically for MCS handling in decompiler."""

    def _find_entry(self, entries: list, sig: str, level: str | None = None) -> dict | None:
        """Find an entry by signature name, optionally filtered by significance."""
        for e in entries:
            if e.sig == sig:
                if level is None or e.level == level:
                    return e.to_dict()
        return None

    def _has_sig(self, entries: list, sig: str) -> bool:
        """Check if signature exists in entries."""
        return any(e.sig == sig for e in entries)

    def test_mcs_name_recovery(self) -> None:
        """MCS entry provides name for packed token.

        With mod32, 'AB' and 'BA' may encode to the same token (OR of bits).
        The decompiler must use MCS nodes to recover the original name.
        """
        # Compile AB which creates MCS entry {AB: [A, B]}
        entries = compile_test_source("AB")

        # Find the MCS entry
        mcs_entry = None
        for e in entries:
            sig, nodes = e.decode(_tokenizer)
            if nodes == ["A", "B"]:
                mcs_entry = e
                break

        assert mcs_entry is not None, "MCS entry not found"

        # Decompile should recover "AB" from the MCS nodes
        decompiler = Decompiler(_tokenizer)
        result = decompiler.decompile(entries)

        # Check that AB entry exists with MCS significance
        entry = self._find_entry(result, "AB")
        assert entry is not None
        assert entry["level"] == "MCS"
        assert entry["nodes"] == ["A", "B"]

    def test_mcs_with_mod32_tokenizer(self) -> None:
        """Test MCS handling with Mod32Tokenizer specifically.

        Mod32 has more collisions than Mod64, making MCS recovery critical.
        """
        tokenizer32 = Mod32Tokenizer()

        source = "ABC => X"
        tokens = Lexer(source).tokenize()
        kscript_file = Parser(tokens).parse()
        entries = Compiler(tokenizer32, dev=True).compile(kscript_file)

        decompiler = Decompiler(tokenizer32)
        result = decompiler.decompile(entries)

        # Should recover "ABC" from MCS nodes and have the construct
        # Find the construct entry (single-node, non-zero sig|nodes → S2)
        entry = self._find_entry(result, "ABC", level="S2")
        assert entry is not None
        # The construct entry should have nodes "X"
        assert entry["nodes"] == "X"

    def test_mcs_countersign_roundtrip(self) -> None:
        """MCS with countersign: AB == CD"""
        result = compile_test_source("AB == CD")

        decompiler = Decompiler(_tokenizer)
        decompiled = decompiler.decompile(result)

        # Both MCS names should be recovered
        assert self._has_sig(decompiled, "AB")
        assert self._has_sig(decompiled, "CD")
        # Check for countersign (S1) entries (not MCS entries)
        entry_ab = self._find_entry(decompiled, "AB", level="S1")
        assert entry_ab is not None
        assert entry_ab["level"] == "S1"


class TestDecompilerEdgeCases:
    """Edge case tests for decompiler."""

    def _roundtrip(self, source: str) -> list:
        """Helper: compile source, then decompile back."""
        entries = compile_test_source(source)
        decompiler = Decompiler(_tokenizer)
        return decompiler.decompile(entries)

    def _find_entry(self, entries: list[DecompiledEntry], sig: str, level: str | None = None) -> dict | None:
        """Find an entry by signature name, optionally filtered by level."""
        for e in entries:
            if e.sig == sig:
                if level is None or e.level == level:
                    return e.to_dict()
        return None

    def _has_sig(self, entries: list, sig: str) -> bool:
        """Check if signature exists in entries."""
        return any(e.sig == sig for e in entries)

    def test_empty_input(self) -> None:
        """Decompile empty input."""
        decompiler = Decompiler(_tokenizer)
        result = decompiler.decompile([])
        assert result == []

    def test_single_char_no_mcs(self) -> None:
        """Single char signatures don't create MCS entries."""
        entries = compile_test_source("A")
        # Should only have unsigned entry, no MCS
        assert len(entries) == 1
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "A"
        assert nodes is None

    def test_subscript_block(self) -> None:
        """Decompile subscript block."""
        source = "A =>\n  B\n  C"
        result = self._roundtrip(source)
        # Check that A has nodes B and C
        entry = self._find_entry(result, "A")
        assert entry is not None
        assert entry["level"] == "S2"
        assert entry["nodes"] == ["B", "C"]
        # Check that B and C exist as entries
        assert self._has_sig(result, "B")
        assert self._has_sig(result, "C")

    def test_complex_nested_script(self) -> None:
        """Compile and decompile complex nested KScript."""
        source = '''MHALL == SVO =>
  S(ubject) = M
  V = H
  O = ALL =>
    A = D
    L = M < MOD => A B
    L > O < BS =>
      B = "baby"
      S = "sheep"
'''
        result = self._roundtrip(source)
        # Check key signatures are present
        assert self._has_sig(result, "MHALL")
        # SVO should be the countersign partner
        assert self._has_sig(result, "SVO")
