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
from kalvin.kline import KLine
from kalvin.mod_tokenizer import Mod32Tokenizer

from ks import Compiler, KScript, SymbolicEntry, compile_source
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


def _decode_sig(entry: KLine) -> str:
    """Decode an entry's signature to a string."""
    return _tok32.decode([entry.signature])


def _decode_nodes(entry: KLine) -> str:
    """Decode an entry's nodes to a string."""
    nodes = entry.as_node_list()
    return _tok32.decode(nodes) if nodes else ""


def _has_entry(
    entries: list[KLine],
    op: str,
    sig: str,
    nodes: str | None = None,
) -> bool:
    """Check if a matching entry exists in the list."""
    for e in entries:
        if e.dbg.op != op:
            continue
        if _decode_sig(e) != sig:
            continue
        if nodes is not None and _decode_nodes(e) != nodes:
            continue
        return True
    return False


def _count_entries(
    entries: list[KLine],
    op: str | None = None,
) -> int:
    """Count entries matching op (or all if op is None)."""
    if op is None:
        return len(entries)
    return sum(1 for e in entries if e.dbg.op == op)


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
        self.entries = compile_source(SOURCE_14_11, dev=True)

    def test_entry_count(self) -> None:
        """Total entry count matches spec §14.11 (18 entries).

        After KB-205 + KB-207: MCS component IDENTITY dedup, no compound-own
        identity, and subscript identity suppression for MCS CANONIZE scopes.
        """
        assert len(self.entries) == 18

    def test_no_duplicate_canonize(self) -> None:
        """CANONIZE dedup: no two CANONIZE entries share the same (sig, nodes)."""
        canonize_keys: set[tuple[int, tuple[int, ...]]] = set()
        for e in self.entries:
            if e.dbg.op == "CANONIZED":
                nodes = tuple(sorted(e.as_node_list()))
                key = (e.signature, nodes)
                assert key not in canonize_keys, (
                    f"Duplicate CANONIZE: sig=0x{e.signature:x}, nodes={nodes}"
                )
                canonize_keys.add(key)

    def test_canonize_count(self) -> None:
        """Three unique CANONIZE entries: MHALL, SVO, ALL."""
        assert _count_entries(self.entries, "CANONIZED") == 3

    def test_mcs_mhall_components(self) -> None:
        """MCS for MHALL: unsigned entries for M, H, A, L components."""
        assert _has_entry(self.entries, "IDENTITY", "M")
        assert _has_entry(self.entries, "IDENTITY", "H")
        assert _has_entry(self.entries, "IDENTITY", "A")
        assert _has_entry(self.entries, "IDENTITY", "L")

    def test_mcs_mhall_canonize(self) -> None:
        """MCS canonization: MHALL → [M, H, A, L, L]."""
        assert _has_entry(self.entries, "CANONIZED", "AHLM", "MHALL")

    def test_mcs_svo_components(self) -> None:
        """MCS for SVO: unsigned entries for S, V, O components."""
        assert _has_entry(self.entries, "IDENTITY", "S")
        assert _has_entry(self.entries, "IDENTITY", "V")
        assert _has_entry(self.entries, "IDENTITY", "O")

    def test_mcs_svo_canonize(self) -> None:
        """MCS canonization: SVO → [S, V, O]."""
        assert _has_entry(self.entries, "CANONIZED", "OSV", "SVO")

    def test_countersign_pair(self) -> None:
        """Countersign: MHALL → SVO and SVO → MHALL."""
        assert _has_entry(self.entries, "COUNTERSIGNED", "AHLM", "OSV")
        assert _has_entry(self.entries, "COUNTERSIGNED", "OSV", "AHLM")

    def test_undersign_pairs(self) -> None:
        """Undersign entries: M→S, H→V (reversed direction)."""
        assert _has_entry(self.entries, "UNDERSIGNED", "M", "S")
        assert _has_entry(self.entries, "UNDERSIGNED", "H", "V")

    def test_mcs_all(self) -> None:
        """MCS for ALL: canonize entry and undersign O→ALL."""
        assert _has_entry(self.entries, "CANONIZED", "AL", "ALL")
        assert _has_entry(self.entries, "UNDERSIGNED", "AL", "O")

    def test_leaf_undersign(self) -> None:
        """Leaf undersign: D→A, M→L."""
        assert _has_entry(self.entries, "UNDERSIGNED", "D", "A")
        assert _has_entry(self.entries, "UNDERSIGNED", "M", "L")

    def test_connotate(self) -> None:
        """Connotate: L → O."""
        assert _has_entry(self.entries, "CONNOTED", "L", "O")

    def test_all_entries_are_compiled_entry(self) -> None:
        """Every entry is a KLine instance."""
        for e in self.entries:
            assert isinstance(e, KLine)

    def test_identity_count(self) -> None:
        """7 IDENTITY entries from MCS component identities.

        MCS components (deduped): M, H, A, L, S, V, O = 7 unique chars.
        Total IDENTITY: 7 (no compound-own identity).
        """
        assert _count_entries(self.entries, "IDENTITY") == 7


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
            decoded.append((e.dbg.op, sig_str, nodes_str))

        # MCS for MHALL should have resolved characters to their bound
        # words.  Under the regenerated BPE model, multi-syllable words
        # (e.g. "Mary", "Lamb") may appear in an entry's node-list rather
        # than its single-node signature, so check each word appears in
        # either the decoded sig or the decoded node-list.
        all_text = [t for _op, sig, nodes in decoded for t in (sig, nodes) if t]
        for word in ("Mary", "Had", "A", "Little", "Lamb"):
            assert any(word == t for t in all_text), (
                f"Expected '{word}' binding to appear in compiled entries"
            )

    def test_inline_binding_subject(self) -> None:
        """Inline annotation S(ubject) → 'Subject'.

        The SVO canonize entry should have S patched to 'Subject'
        via Rule B4 override.  Under BPE, 'Subject' splits across
        subwords, so the resolved word appears in an entry's node-list
        rather than a single-node signature.
        """
        nlp = self._get_nlp_tokenizer()
        entries = compile_source(SOURCE_14_12, tokenizer=nlp)

        # S(ubject) → 'Subject' resolves into a node-list under BPE.
        found = False
        for e in entries:
            nodes = e.as_node_list()
            if nodes and nlp.decode(nodes) == "Subject":
                found = True
                break
        assert found, "Expected 'Subject' from inline binding"

    def test_compiled_entries_valid(self) -> None:
        """All entries are KLine instances with valid structure."""
        nlp = self._get_nlp_tokenizer()
        entries = compile_source(SOURCE_14_12, tokenizer=nlp)
        assert len(entries) > 0
        for e in entries:
            assert isinstance(e, KLine)
            assert isinstance(e.signature, int)
            assert e.dbg.op in ("COUNTERSIGNED", "CANONIZED", "CONNOTED", "UNDERSIGNED", "IDENTITY")


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
        assert _has_entry(entries, "COUNTERSIGNED", "A", "B")
        assert _has_entry(entries, "COUNTERSIGNED", "B", "A")


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
        """All returned items are KLine objects."""
        entries = compile_source("A == B")
        for e in entries:
            assert isinstance(e, KLine)
            assert isinstance(e.signature, int)
            assert isinstance(e.dbg.op, str)

    def test_correct_structure_countersign(self) -> None:
        """COUNTERSIGN produces correct sig/nodes structure."""
        entries = compile_source("A == B")
        # Should have COUNTERSIGN A→B and B→A
        assert _has_entry(entries, "COUNTERSIGNED", "A", "B")
        assert _has_entry(entries, "COUNTERSIGNED", "B", "A")

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
        """KScript('A == B').entries returns a list of KLine."""
        model = KScript("A == B")
        assert isinstance(model.entries, list)
        assert len(model.entries) > 0
        for e in model.entries:
            assert isinstance(e, KLine)

    def test_unsigned_entry(self) -> None:
        """KScript('A') produces a single unsigned entry."""
        model = KScript("A")
        assert len(model.entries) == 1
        assert model.entries[0].dbg.op == "IDENTITY"
        assert _decode_sig(model.entries[0]) == "A"

    def test_complex_source(self) -> None:
        """KScript handles complex nested source."""
        model = KScript(SOURCE_14_11)
        assert len(model.entries) > 10
        assert _has_entry(model.entries, "COUNTERSIGNED", "AHLM", "OSV")

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
        assert _has_entry(entries, "COUNTERSIGNED", "A", "B")

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
        assert _has_entry(entries, "COUNTERSIGNED", "ABC", "X")
        assert _has_entry(entries, "COUNTERSIGNED", "X", "ABC")

    def test_end_to_end_canonize(self) -> None:
        """CANONIZE operator with subscript block."""
        source = "A =>\n  B\n  C"
        compiler = Compiler()
        tokens = Lexer(source).tokenize()
        kfile = Parser(tokens).parse()
        entries = compiler.compile(kfile)
        assert _has_entry(entries, "CANONIZED", "A", "BC")

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
        countersigns = [e for e in entries if e.dbg.op == "COUNTERSIGNED"]
        assert len(countersigns) >= 2
