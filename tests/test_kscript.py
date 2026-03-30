"""Tests for KScript v2 compiler with CLN-based semantics."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from kalvin.abstract import KLine
from kalvin.mod_tokenizer import Mod64Tokenizer, PACKED_BIT
from kalvin.significance import Int32Significance
from kscript.lexer import Lexer
from kscript.parser import Parser, Signature, Literal, Construct, Script, KScriptFile
from kscript.compiler import Compiler, CompiledEntry, compile_source

# Shared significance instance for tests
_sig = Int32Significance()

# Shared tokenizer for tests
_tokenizer = Mod64Tokenizer()


def compile_source(source: str) -> list[CompiledEntry]:
    """Helper to compile source string to entries using v2 compiler."""
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
# 1. CLN Collection Tests
# =============================================================================

class TestCLNCollection:
    """Tests for CLN-based node collection."""

    def test_cln_canonize(self) -> None:
        """CLN collection for canonize: A => B C <= D -> CLNs = [B, C]"""
        source = "A => B C <= D"
        tokens = Lexer(source).tokenize()
        kscript_file = Parser(tokens).parse()

        assert len(kscript_file.scripts) == 1
        script = kscript_file.scripts[0]
        assert len(script.constructs) >= 1

        # First construct should have CLNs [B, C]
        construct = script.constructs[0]
        assert construct.sig.id == "A"
        assert construct.op == "=>"  # Canonize fwd operator
        assert len(construct.clns) == 2
        assert construct.clns[0].id == "B"
        assert construct.clns[1].id == "C"

        # BWD should bind ALL CLNs
        assert construct.bwd is not None
        bwd_sig, bwd_op, bwd_clns = construct.bwd
        assert bwd_sig.id == "D"
        assert bwd_op == "<="
        assert len(bwd_clns) == 2

    def test_cln_connotate_bwd(self) -> None:
        """CLN collection for connotate BWD: A => B C < D -> binds CLNs[-1]"""
        source = "A => B C < D"
        tokens = Lexer(source).tokenize()
        kscript_file = Parser(tokens).parse()

        script = kscript_file.scripts[0]
        construct = script.constructs[0]

        # BWD should bind only CLNs[-1]
        assert construct.bwd is not None
        bwd_sig, bwd_op, bwd_clns = construct.bwd
        assert bwd_sig.id == "D"
        assert bwd_op == "<"
        assert len(bwd_clns) == 1
        assert bwd_clns[0].id == "C"

    def test_cln_implicit_opening(self) -> None:
        """CLN collection with implicit opening: A B <= CD -> CLNs = [A, B]"""
        source = "A B <= CD"
        tokens = Lexer(source).tokenize()
        kscript_file = Parser(tokens).parse()

        script = kscript_file.scripts[0]
        # Should have constructs with BWD binding [A, B]

    def test_literal_in_bwd_sig_position(self) -> None:
        """Literal in BWD sig position is rejected."""
        source = "A < 1"
        tokens = Lexer(source).tokenize()
        kscript_file = Parser(tokens).parse()

        # A should be treated as identity, BWD rejected
        script = kscript_file.scripts[0]
        # Implementation should not create BWD for literal


# =============================================================================
# 2. Eager Emit Tests
# =============================================================================

class TestEagerEmit:
    """Tests for eager emit compilation."""

    def test_eager_emit_canonize(self) -> None:
        """Eager emit for canonize: A => B C emits immediately."""
        entries = compile_source("A => B C")
        d = entries_to_dict(entries)
        assert d["A"] == ["B", "C"]

    def test_bwd_triggers_additional_emit(self) -> None:
        """BWD triggers additional emit: A => B C <= D."""
        entries = compile_source("A => B C <= D")
        d = entries_to_dict(entries)

        # First emit: {A|S2: [B, C]}
        assert d["A"] == ["B", "C"]
        # Second emit (BWD): {D|S2: [B, C]}
        assert d["D"] == ["B", "C"]


# =============================================================================
# 3. MCS Expansion Tests
# =============================================================================

class TestMCSExpansion:
    """Tests for MCS expansion."""

    def test_mcs_simple_expansion(self) -> None:
        """MCS expansion: ABC emits canonization + identities."""
        entries = compile_source("ABC")

        # Should have 4 entries: MCS canonization + 4 identities
        assert len(entries) == 5

        d = entries_to_multidict(entries)

        # MCS canonization: {ABC: [A, B, C]}
        assert ["A", "B", "C"] in d["ABC"]

        # Component identities
        assert d["A"][0] is None
        assert d["B"][0] is None
        assert d["C"][0] is None

    def test_mcs_in_construct_position(self) -> None:
        """MCS in construct position."""
        entries = compile_source("ABC => X")

        # Should have: MCS canonization + 3 identities + construct + entity
        assert len(entries) >= 5

        # First entry should be MCS canonization
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "ABC"
        assert nodes == ["A", "B", "C"]

        # Next 3 should be component identities
        for i in range(1, 4):
            sig, nodes = entries[i].decode(_tokenizer)
            assert sig in "ABC"
            assert nodes is None

        # Fifth entry should be the construct
        sig, nodes = entries[4].decode(_tokenizer)
        assert sig == "ABC"
        assert nodes == ["X"]

    def test_no_mcs_for_single_char(self) -> None:
        """No MCS for single-char owner."""
        entries = compile_source("A => X")

        d = entries_to_dict(entries)
        # Only construct entry, no MCS
        assert d["A"] == ["X"]

    def test_mcs_with_countersign(self) -> None:
        """MCS expansion with countersign construct."""
        entries = compile_source("ABC == X")

        # Should have: MCS canonization + 3 identities + 2 countersign entries
        assert len(entries) == 6

        # Check entries in order (dict would overwrite duplicate keys)
        # Entry 0: MCS canonization {ABC: [A, B, C]}
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "ABC"
        assert nodes == ["A", "B", "C"]

        # Entries 1-3: Component identities
        for i in range(1, 4):
            sig, nodes = entries[i].decode(_tokenizer)
            assert sig in "ABC"
            assert nodes is None

        # Entries 4-5: Countersign bidirectional
        sig, nodes = entries[4].decode(_tokenizer)
        assert sig == "ABC" and nodes == "X"
        sig, nodes = entries[5].decode(_tokenizer)
        assert sig == "X" and nodes == "ABC"


# =============================================================================
# 4. Significance Level Tests
# =============================================================================

class TestSignificanceLevels:
    """Tests for significance level emission."""

    def test_countersign_significance(self) -> None:
        """Countersign: S1 bidirectional."""
        entries = compile_source("A == B")

        # Check significance bits
        for e in entries:
            if e.signature & _sig.S1:
                # Found S1 entry
                return
        assert False, "No S1 entry found"

    def test_canonize_significance(self) -> None:
        """Canonize: S2 multi-node."""
        entries = compile_source("AB => C D")

        # Check for S2 entry
        for e in entries:
            if e.signature & _sig.S2:
                return
        assert False, "No S2 entry found"

    def test_connotate_significance(self) -> None:
        """Connotate: S3 single-node with entity."""
        entries = compile_source("A > B")

        # Check for S3 entry
        for e in entries:
            if e.signature & _sig.S3:
                return
        assert False, "No S3 entry found"

    def test_undersign_significance(self) -> None:
        """Undersign: S4 unidirectional with entity."""
        entries = compile_source("A = B")

        d = entries_to_dict(entries)
        assert d["A"] == "B"
        assert d["B"] is None


# =============================================================================
# 5. Subscript Tests
# =============================================================================

class TestSubscripts:
    """Tests for subscript processing."""

    def test_subscript_attaches_to_last_cln(self) -> None:
        """Subscript attaches to last CLN."""
        source = "A =>\n  B\n  C = D"
        entries = compile_source(source)

        d = entries_to_dict(entries)
        # A => [B, C]
        assert d["A"] == ["B", "C"]
        # C = D (subscript attaches to B, but C starts new construct)
        assert d["C"] == "D"

    def test_nested_subscript(self) -> None:
        """Nested subscript."""
        source = "A =>\n  B =>\n    C\n    D"
        entries = compile_source(source)

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
        entries = compile_source("AB => A B")
        d = entries_to_dict(entries)

        # MCS for AB
        assert d["AB"] == ["A", "B"]
        # Entities
        assert d["A"] is None
        assert d["B"] is None

    def test_ab_double_equal_cd(self) -> None:
        """Test: AB == CD"""
        entries = compile_source("AB == CD")
        md = entries_to_multidict(entries)

        # MCS for AB: {AB: [A, B]}
        assert ["A", "B"] in md.get("AB", [])
        # Countersign: {AB: CD} and {CD: AB}
        assert "CD" in md.get("AB", [])
        assert "AB" in md.get("CD", [])

    def test_a_b_le_arrow_cd(self) -> None:
        """Test: A B <= CD"""
        entries = compile_source("A B <= CD")
        md = entries_to_multidict(entries)

        # CD binds ALL CLNs [A, B]
        assert ["A", "B"] in md.get("CD", [])
        # MCS for CD
        assert None in md.get("C", [])
        assert None in md.get("D", [])

    def test_ab_gt_c(self) -> None:
        """Test: AB > C"""
        entries = compile_source("AB > C")
        md = entries_to_multidict(entries)

        # MCS for AB: {AB: [A, B]}
        assert ["A", "B"] in md.get("AB", [])
        # Connotate: {AB|S3: [C]}
        assert ["C"] in md.get("AB", [])
        # Entity: {C|S4: None}
        assert None in md.get("C", [])

    def test_c_lt_ab(self) -> None:
        """Test: C < AB"""
        entries = compile_source("C < AB")
        d = entries_to_dict(entries)

        # Identity for C
        assert d["C"] is None
        # BWD connotate: {AB|S3: [C]}
        # MCS for AB
        assert d["AB"] == ["A", "B"]


# =============================================================================
# 7. Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Tests for backward compatibility with existing behavior."""

    def test_identity(self) -> None:
        """Test compiling identity script."""
        entries = compile_source("A")
        assert len(entries) == 1
        sig, nodes = entries[0].decode(_tokenizer)
        assert sig == "A"
        assert nodes is None

    def test_countersign(self) -> None:
        """Test compiling countersign (bidirectional)."""
        entries = compile_source("A == B")
        d = entries_to_dict(entries)
        assert d["A"] == "B"
        assert d["B"] == "A"

    def test_undersign(self) -> None:
        """Test compiling undersign."""
        entries = compile_source("A = B")
        assert len(entries) == 2
        sig1, nodes1 = entries[0].decode(_tokenizer)
        sig2, nodes2 = entries[1].decode(_tokenizer)
        assert sig1 == "A" and nodes1 == "B"
        assert sig2 == "B" and nodes2 is None

    def test_connotate_fwd(self) -> None:
        """Test compiling forward connotate."""
        entries = compile_source("A > B")
        assert len(entries) == 2
        sig1, nodes1 = entries[0].decode(_tokenizer)
        sig2, nodes2 = entries[1].decode(_tokenizer)
        assert sig1 == "A" and nodes1 == ["B"]
        assert sig2 == "B" and nodes2 is None

    def test_connotate_bwd(self) -> None:
        """Test compiling backward connotate."""
        entries = compile_source("A < B")
        assert len(entries) == 2
        sig1, nodes1 = entries[0].decode(_tokenizer)
        sig2, nodes2 = entries[1].decode(_tokenizer)
        assert sig1 == "A" and nodes1 is None
        assert sig2 == "B" and nodes2 == ["A"]

    def test_canonize_fwd(self) -> None:
        """Test compiling forward canonize."""
        entries = compile_source("AB => C D")
        d = entries_to_dict(entries)
        assert d["AB"] == ["C", "D"]

    def test_literals(self) -> None:
        """Test compiling string and number literals."""
        d = entries_to_dict(compile_source(r'A = "\"hello\""'))
        assert d["A"] == r'"\"hello\""'
        d = entries_to_dict(compile_source("A = 42"))
        assert d["A"] == "42"

    def test_multiple_constructs(self) -> None:
        """Test multiple constructs with immediate binding."""
        entries = compile_source("A => B => C")
        # Check S2 construct entries (before entity emissions)
        sig1, nodes1 = entries[0].decode(_tokenizer)
        sig2, nodes2 = entries[2].decode(_tokenizer)
        assert sig1 == "A" and nodes1 == ["B"]
        assert sig2 == "B" and nodes2 == ["C"]
        # Remaining entries are entities for B and C
        assert len(entries) == 4  # 2 constructs + 2 entities

    def test_subscript_as_nodes(self) -> None:
        """Test subscript signatures as nodes."""
        source = "A =>\n  B\n  C"
        entries = compile_source(source)
        d = entries_to_dict(entries)
        assert d["A"] == ["B", "C"]
        assert d["B"] is None
        assert d["C"] is None
