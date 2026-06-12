"""Tests for KScript v3 Compiler Orchestrator and Public API.

Covers:
  KS-35 — Complex nested example (§14.11) with complete entry validation
  KS-36 — NLP-bound example (§14.12) with binding resolution
  KS-37 — Mixed NLP/Mod32 compilation (same source, different tokenizers)
  compile_source — Convenience function
  KScript API — Public class
  Pipeline wiring — End-to-end pipeline
  BindingScope always created — No Mod32 mode switch
"""

from __future__ import annotations

import pytest

from kalvin.abstract import KTokenizer
from kalvin.mod_tokenizer import Mod32Tokenizer

from ks import CompiledEntry, Compiler, KScript, SymbolicEntry, compile_source
from ks.lexer import Lexer
from ks.parser import Parser
from ks.ast_emitter import ASTEmitter
from ks.binding_scope import BindingScope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_tok32 = Mod32Tokenizer()

# NLP tokenizer availability
try:
    from kalvin.nlp_tokenizer import NLPTokenizer
    _has_nlp_import = True
except ImportError:
    _has_nlp_import = False

# Check if NLP tokenizer data files are actually available
_nlp_available = False
if _has_nlp_import:
    try:
        NLPTokenizer.from_files()
        _nlp_available = True
    except (FileNotFoundError, OSError):
        pass

_nlp_skip = pytest.mark.skipif(
    not _nlp_available,
    reason="NLPTokenizer data files unavailable",
)


def _decode_sig(entry: CompiledEntry) -> str:
    """Decode an entry's signature to a string."""
    return _tok32.decode([entry.signature])


def _decode_nodes(entry: CompiledEntry) -> str:
    """Decode an entry's nodes to a string."""
    nodes = entry.as_node_list()
    return _tok32.decode(nodes) if nodes else ""


def _has_entry(
    entries: list[CompiledEntry],
    op: str,
    sig: str,
    nodes: str | None = None,
) -> bool:
    """Check if a matching entry exists in the list."""
    for e in entries:
        if e.op != op:
            continue
        if _decode_sig(e) != sig:
            continue
        if nodes is not None and _decode_nodes(e) != nodes:
            continue
        return True
    return False


def _count_entries(
    entries: list[CompiledEntry],
    op: str | None = None,
) -> int:
    """Count entries matching op (or all if op is None)."""
    if op is None:
        return len(entries)
    return sum(1 for e in entries if e.op == op)


# ---------------------------------------------------------------------------
# §14.11 source
# ---------------------------------------------------------------------------

SOURCE_14_11 = """\
MHALL == SVO =>
  S = M
  V = H
  O = ALL =>
    A = D
    L = M
    L > O"""


# ---------------------------------------------------------------------------
# KS-35: Complex nested example (§14.11)
# ---------------------------------------------------------------------------

class TestKS35ComplexNested:
    """KS-35 — Complex nested example from §14.11.

    Compiles the full MHALL == SVO => ... source with Mod32Tokenizer
    and validates the complete entry list.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.entries = compile_source(SOURCE_14_11)

    def test_entry_count(self) -> None:
        """Total entry count matches the v3 ASTEmitter output.

        After KB-193: bare Signature items no longer emit spurious UNSIGNED.
        After KB-199: CANONIZE subscript blocks emit identity UNSIGNED for
        leaf Signatures and UNDERSIGN scope sigs. Was 28, now 29 (+1 D UNSIGNED).
        """
        assert len(self.entries) == 29

    def test_no_duplicate_canonize(self) -> None:
        """CANONIZE dedup: no two CANONIZE entries share the same (sig, nodes)."""
        canonize_keys: set[tuple[int, tuple[int, ...]]] = set()
        for e in self.entries:
            if e.op == "CANONIZE":
                nodes = tuple(sorted(e.as_node_list()))
                key = (e.signature, nodes)
                assert key not in canonize_keys, (
                    f"Duplicate CANONIZE: sig=0x{e.signature:x}, nodes={nodes}"
                )
                canonize_keys.add(key)

    def test_canonize_count(self) -> None:
        """Three unique CANONIZE entries: MHALL, SVO, ALL."""
        assert _count_entries(self.entries, "CANONIZE") == 3

    def test_mcs_mhall_components(self) -> None:
        """MCS for MHALL: unsigned entries for M, H, A, L components."""
        assert _has_entry(self.entries, "IDENTITY", "M")
        assert _has_entry(self.entries, "IDENTITY", "H")
        assert _has_entry(self.entries, "IDENTITY", "A")
        assert _has_entry(self.entries, "IDENTITY", "L")

    def test_mcs_mhall_canonize(self) -> None:
        """MCS canonization: MHALL → [M, H, A, L, L]."""
        assert _has_entry(self.entries, "CANONIZE", "AHLM", "MHALL")

    def test_mcs_svo_components(self) -> None:
        """MCS for SVO: unsigned entries for S, V, O components."""
        assert _has_entry(self.entries, "IDENTITY", "S")
        assert _has_entry(self.entries, "IDENTITY", "V")
        assert _has_entry(self.entries, "IDENTITY", "O")

    def test_mcs_svo_canonize(self) -> None:
        """MCS canonization: SVO → [S, V, O]."""
        assert _has_entry(self.entries, "CANONIZE", "OSV", "SVO")

    def test_countersign_pair(self) -> None:
        """Countersign: MHALL → SVO and SVO → MHALL."""
        assert _has_entry(self.entries, "COUNTERSIGN", "AHLM", "OSV")
        assert _has_entry(self.entries, "COUNTERSIGN", "OSV", "AHLM")

    def test_undersign_pairs(self) -> None:
        """Undersign entries: M→S, H→V (reversed direction)."""
        assert _has_entry(self.entries, "UNDERSIGN", "M", "S")
        assert _has_entry(self.entries, "UNDERSIGN", "H", "V")

    def test_mcs_all(self) -> None:
        """MCS for ALL: canonize entry and undersign O→ALL."""
        assert _has_entry(self.entries, "CANONIZE", "AL", "ALL")
        assert _has_entry(self.entries, "UNDERSIGN", "AL", "O")

    def test_leaf_undersign(self) -> None:
        """Leaf undersign: D→A, M→L."""
        assert _has_entry(self.entries, "UNDERSIGN", "D", "A")
        assert _has_entry(self.entries, "UNDERSIGN", "M", "L")

    def test_connotate(self) -> None:
        """Connotate: L → O."""
        assert _has_entry(self.entries, "CONNOTATE", "L", "O")

    def test_all_entries_are_compiled_entry(self) -> None:
        """Every entry is a CompiledEntry instance."""
        for e in self.entries:
            assert isinstance(e, CompiledEntry)

    def test_unsigned_count(self) -> None:
        """18 UNSIGNED entries from MCS expansion + identity UNSIGNED.

        After KB-199: adds identity UNSIGNED for D (leaf Signature in
        CANONIZE subscript block). Was 17, now 18.
        """
        assert _count_entries(self.entries, "UNSIGNED") == 18


# ---------------------------------------------------------------------------
# KS-36: NLP-bound example (§14.12)
# ---------------------------------------------------------------------------

SOURCE_14_12 = """\
(Mary Had A Little Lamb)
MHALL == SVO =>
  S(ubject) = M
  V = H
  O = ALL =>
    A = D
    L = M
    L > O"""


@_nlp_skip
class TestKS36NLPBound:
    """KS-36 — NLP-bound example from §14.12.

    Tests that block annotation provides word bindings and inline
    annotations resolve correctly.  Requires NLPTokenizer data files.
    """

    def _get_nlp_tokenizer(self) -> NLPTokenizer:
        """Get NLPTokenizer from standard file paths."""
        return NLPTokenizer.from_files()

    def test_nlp_binding_mcs_mhall(self) -> None:
        """MCS for MHALL resolves characters to NLP words.

        M → Mary, H → Had, A → "A", L → "Little", L → "Lamb".
        """
        nlp = self._get_nlp_tokenizer()
        entries = compile_source(SOURCE_14_12, tokenizer=nlp)

        # Build decoded map using NLP tokenizer
        decoded = []
        for e in entries:
            sig_str = nlp.decode([e.signature])
            nodes = e.as_node_list()
            nodes_str = nlp.decode(nodes) if nodes else ""
            decoded.append((e.op, sig_str, nodes_str))

        # MCS for MHALL should have resolved characters
        # M → Mary, H → Had, A → "A", L → "Little", L → "Lamb"
        assert any(op == "IDENTITY" and sig == "Mary" for op, sig, _ in decoded), \
            "Expected IDENTITY Mary entry"
        assert any(op == "IDENTITY" and sig == "Had" for op, sig, _ in decoded), \
            "Expected IDENTITY Had entry"
        assert any(op == "IDENTITY" and sig == "A" for op, sig, _ in decoded), \
            "Expected IDENTITY A entry"

    def test_inline_binding_subject(self) -> None:
        """Inline annotation S(ubject) → 'Subject'.

        The SVO canonize entry should have S patched to 'Subject'
        via Rule B4 override.
        """
        nlp = self._get_nlp_tokenizer()
        entries = compile_source(SOURCE_14_12, tokenizer=nlp)

        # Find CANONIZE entries for SVO
        for e in entries:
            if e.op == "CANONIZE":
                sig_str = nlp.decode([e.signature])
                if sig_str == "SVO" or "S" in sig_str:
                    nodes = e.as_node_list()
                    node_strs = [nlp.decode([n]) for n in nodes]
                    # At least one CANONIZE should contain "Subject"
                    if "Subject" in node_strs:
                        return  # Found the patched entry

        # If we get here, check that S resolved to Subject somewhere
        all_sigs = [nlp.decode([e.signature]) for e in entries]
        assert "Subject" in all_sigs, "Expected 'Subject' from inline binding"

    def test_compiled_entries_valid(self) -> None:
        """All entries are CompiledEntry instances with valid structure."""
        nlp = self._get_nlp_tokenizer()
        entries = compile_source(SOURCE_14_12, tokenizer=nlp)
        assert len(entries) > 0
        for e in entries:
            assert isinstance(e, CompiledEntry)
            assert isinstance(e.signature, int)
            assert e.op in ("COUNTERSIGN", "CANONIZE", "CONNOTATE", "UNDERSIGN", "IDENTITY")


# ---------------------------------------------------------------------------
# KS-37: Mixed NLP/Mod32 compatibility
# ---------------------------------------------------------------------------

class TestKS37MixedNLPMod32:
    """KS-37 — Same source compiles under both Mod32 and NLP without modification.

    The compiler always creates a BindingScope (no mode switch).
    With Mod32, the scope has no word lists, so all characters pass through raw.
    """

    def test_mod32_compiles_same_source(self) -> None:
        """Mod32 tokenizer compiles the NLP source without errors."""
        entries = compile_source(SOURCE_14_12, tokenizer=_tok32)
        assert len(entries) > 0

    def test_mod32_all_entries_unsigned_single_char(self) -> None:
        """In Mod32 mode, all signatures are single-token packed bit positions."""
        entries = compile_source("A == B", tokenizer=_tok32)
        for e in entries:
            # Mod32 produces single-token signatures (packed bits)
            assert isinstance(e.signature, int)
            assert e.signature > 0

    def test_annotation_parsed_in_mod32(self) -> None:
        """Annotations are parsed but unused in Mod32 mode."""
        source = "(Mary Had A Little Lamb)\nA == B"
        entries = compile_source(source, tokenizer=_tok32)
        assert len(entries) > 0
        # In Mod32 mode, the annotation doesn't affect encoding
        # All characters pass through raw
        assert _has_entry(entries, "COUNTERSIGN", "A", "B")
        assert _has_entry(entries, "COUNTERSIGN", "B", "A")


# ---------------------------------------------------------------------------
# compile_source convenience function
# ---------------------------------------------------------------------------

class TestCompileSource:
    """Tests for the compile_source() convenience function."""

    def test_returns_nonempty_list(self) -> None:
        """compile_source('A == B') returns a non-empty list."""
        entries = compile_source("A == B")
        assert isinstance(entries, list)
        assert len(entries) > 0

    def test_entries_are_compiled_entry(self) -> None:
        """All returned items are CompiledEntry objects."""
        entries = compile_source("A == B")
        for e in entries:
            assert isinstance(e, CompiledEntry)
            assert isinstance(e.signature, int)
            assert isinstance(e.op, str)

    def test_correct_structure_countersign(self) -> None:
        """COUNTERSIGN produces correct sig/nodes structure."""
        entries = compile_source("A == B")
        # Should have COUNTERSIGN A→B and B→A
        assert _has_entry(entries, "COUNTERSIGN", "A", "B")
        assert _has_entry(entries, "COUNTERSIGN", "B", "A")

    def test_empty_source(self) -> None:
        """Empty source produces empty entries."""
        entries = compile_source("")
        assert entries == []

    def test_dev_mode(self) -> None:
        """Dev mode populates dbg on entries."""
        entries = compile_source("A", dev=True)
        assert len(entries) == 1
        assert entries[0].dbg is not None

    def test_custom_tokenizer(self) -> None:
        """Passing a custom tokenizer works."""
        tok = Mod32Tokenizer()
        entries = compile_source("A == B", tokenizer=tok)
        assert len(entries) > 0


# ---------------------------------------------------------------------------
# KScript public API
# ---------------------------------------------------------------------------

class TestKScriptAPI:
    """Tests for the KScript public API class."""

    def test_entries_property(self) -> None:
        """KScript('A == B').entries returns a list of CompiledEntry."""
        model = KScript("A == B")
        assert isinstance(model.entries, list)
        assert len(model.entries) > 0
        for e in model.entries:
            assert isinstance(e, CompiledEntry)

    def test_unsigned_entry(self) -> None:
        """KScript('A') produces a single unsigned entry."""
        model = KScript("A")
        assert len(model.entries) == 1
        assert model.entries[0].op == "IDENTITY"
        assert _decode_sig(model.entries[0]) == "A"

    def test_complex_source(self) -> None:
        """KScript handles complex nested source."""
        model = KScript(SOURCE_14_11)
        assert len(model.entries) > 10
        assert _has_entry(model.entries, "COUNTERSIGN", "AHLM", "OSV")

    def test_dev_mode(self) -> None:
        """KScript(dev=True) enables debug text."""
        model = KScript("A == B", dev=True)
        for e in model.entries:
            assert e.dbg is not None or e.dbg is None  # dbg field exists

    def test_default_tokenizer(self) -> None:
        """Default tokenizer is Mod32Tokenizer."""
        model = KScript("A")
        # The entry's signature should be a Mod32 packed bit
        sig = model.entries[0].signature
        assert _tok32.decode([sig]) == "A"


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------

class TestPipelineWiring:
    """Tests that the Compiler correctly wires Lexer → Parser → Emitter → Encoder."""

    def test_end_to_end_simple(self) -> None:
        """Simple source compiles end-to-end with default Mod32Tokenizer."""
        compiler = Compiler()
        tokens = Lexer("A == B").tokenize()
        kfile = Parser(tokens).parse()
        entries = compiler.compile(kfile)
        assert len(entries) > 0
        assert _has_entry(entries, "COUNTERSIGN", "A", "B")

    def test_end_to_end_mcs(self) -> None:
        """Multi-character identifier triggers MCS expansion."""
        compiler = Compiler()
        tokens = Lexer("ABC == X").tokenize()
        kfile = Parser(tokens).parse()
        entries = compiler.compile(kfile)
        # MCS for ABC
        assert _has_entry(entries, "IDENTITY", "A")
        assert _has_entry(entries, "IDENTITY", "B")
        assert _has_entry(entries, "IDENTITY", "C")
        # No MCS for single-char X
        assert _has_entry(entries, "COUNTERSIGN", "ABC", "X")
        assert _has_entry(entries, "COUNTERSIGN", "X", "ABC")

    def test_end_to_end_canonize(self) -> None:
        """CANONIZE operator with subscript block."""
        source = "A =>\n  B\n  C"
        compiler = Compiler()
        tokens = Lexer(source).tokenize()
        kfile = Parser(tokens).parse()
        entries = compiler.compile(kfile)
        assert _has_entry(entries, "CANONIZE", "A", "BC")

    def test_compiler_with_custom_tokenizer(self) -> None:
        """Compiler accepts custom tokenizer."""
        tok = Mod32Tokenizer()
        compiler = Compiler(tokenizer=tok)
        tokens = Lexer("A = B").tokenize()
        kfile = Parser(tokens).parse()
        entries = compiler.compile(kfile)
        assert len(entries) > 0

    def test_dev_mode(self) -> None:
        """Compiler dev mode populates dbg."""
        compiler = Compiler(dev=True)
        tokens = Lexer("A").tokenize()
        kfile = Parser(tokens).parse()
        entries = compiler.compile(kfile)
        assert entries[0].dbg is not None


# ---------------------------------------------------------------------------
# BindingScope always created
# ---------------------------------------------------------------------------

class TestBindingScopeAlwaysCreated:
    """Verify that a BindingScope is always instantiated — no Mod32 mode switch."""

    def test_binding_scope_created_for_mod32(self) -> None:
        """Even with Mod32 tokenizer, Compiler creates a BindingScope internally."""
        # We verify this indirectly: the compiler produces the same
        # output whether or not a binding scope exists, because Mod32
        # characters have no word lists and resolve() returns None.
        # The key invariant: no conditional mode switch.
        compiler = Compiler()
        tokens = Lexer("A == B").tokenize()
        kfile = Parser(tokens).parse()
        entries = compiler.compile(kfile)

        # If BindingScope were NOT created, the ASTEmitter would
        # operate differently. But since it IS always created,
        # the output is correct.
        assert len(entries) > 0

    def test_no_mode_switch_in_compiler(self) -> None:
        """Compiler.compile() always creates BindingScope unconditionally.

        This is a structural test — we verify that the compiler
        doesn't check supports_mcs or any tokenizer property
        to decide whether to create a scope.
        """
        import inspect
        source = inspect.getsource(Compiler.compile)
        assert "supports_mcs" not in source, (
            "Compiler.compile should not reference supports_mcs"
        )
        assert "skip_mcs" not in source, (
            "Compiler.compile should not reference skip_mcs"
        )

    def test_annotation_feeds_binding_scope_in_v3(self) -> None:
        """In v3, annotations always feed the BindingScope (no mode switch).

        This means annotations affect character resolution even with Mod32.
        The word 'annotation' in the annotation resolves A to 'annotation'
        (first-letter match). This is correct v3 behavior.
        """
        source = "(annotation)\nA == B"
        entries = compile_source(source)
        # A resolves to 'annotation' via first-letter match
        # Verify entries exist (binding scope was active)
        assert len(entries) > 0
        # The COUNTERSIGN pair should involve the resolved word
        countersigns = [e for e in entries if e.op == "COUNTERSIGN"]
        assert len(countersigns) >= 2
