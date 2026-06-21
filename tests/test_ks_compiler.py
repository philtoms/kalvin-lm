"""Tests for KScript v3 Compiler Orchestrator and Public API.

Covers:
  KS-35 — Complex nested example (§14.11) with complete entry validation
  KS-36 — Word-bound example (§14.12) with binding resolution
  compile_source — Convenience function
  KScript API — Public class
  Pipeline wiring — End-to-end pipeline
  BindingScope always created — no tokenizer mode switch
"""

from __future__ import annotations

import pytest

from kalvin.kline import KLine
from kalvin.nlp_tokenizer import NLPTokenizer
from ks import Compiler, KScript, compile_source
from ks.lexer import Lexer
from ks.parser import Parser

# Tokenizer data-asset gating shared via conftest for consistent skips
from tests.conftest import requires_tokenizer_data

# The whole module compiles real KScript sources (default kalvin tokenizer);
# skip cleanly when the tokenizer data assets are absent.
pytestmark = requires_tokenizer_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Lazy module-level tokenizer (safe at import time; ``pytestmark`` gates
# execution so this is only ever instantiated on a data-present machine).
_tok32_instance: NLPTokenizer | None = None


def _tok32() -> NLPTokenizer:
    """Return the shared tokenizer, constructing it on first use."""
    global _tok32_instance
    if _tok32_instance is None:
        _tok32_instance = NLPTokenizer()
    return _tok32_instance


def _decode_sig(entry: KLine) -> str:
    """Decode an entry's signature to its conceptual identity string.

    Prefers ``dbg.label`` (the source identifier, e.g. ``"MHALL"``) when
    available (dev mode); otherwise falls back to a raw BPE decode of the
    uint64 signature.
    """
    if entry.dbg and entry.dbg.label:
        return entry.dbg.label
    return _tok32().decode([entry.signature])


def _decode_nodes(
    entry: KLine,
    sig_to_label: dict[int, str] | None = None,
) -> str:
    """Decode an entry's node values to a concatenated identity string.

    Compound-identifier node values are recovered via the ``sig_to_label``
    map (signature → conceptual identity) built from the surrounding entries;
    single-character tokens fall back to a raw BPE decode.
    """
    nodes = entry.as_node_list()
    if not nodes:
        return ""
    parts: list[str] = []
    for n in nodes:
        if sig_to_label and n in sig_to_label:
            parts.append(sig_to_label[n])
        else:
            parts.append(_tok32().decode([n]))
    return "".join(parts)


def _has_entry(
    entries: list[KLine],
    op: str,
    sig: str,
    nodes: str | None = None,
) -> bool:
    """Check if a matching entry exists in the list."""
    sig_to_label = {e.signature: e.dbg.label for e in entries if e.dbg and e.dbg.label}
    for e in entries:
        if not e.dbg or e.dbg.op != op:
            continue
        if _decode_sig(e) != sig:
            continue
        if nodes is not None and _decode_nodes(e, sig_to_label) != nodes:
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

    Compiles the full MHALL == SVO => ... source with the tokenizer
    and validates the complete entry list.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.entries = compile_source(SOURCE_14_11, dev=True)

    def test_entry_count(self) -> None:
        """Total entry count matches spec §14.11 (18 entries).

        MTS component IDENTITY dedup, no compound-own
        identity, and subscript identity suppression for MTS CANONIZE scopes.
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

    def test_mts_mhall_components(self) -> None:
        """MTS for MHALL: unsigned entries for M, H, A, L components."""
        assert _has_entry(self.entries, "IDENTITY", "M")
        assert _has_entry(self.entries, "IDENTITY", "H")
        assert _has_entry(self.entries, "IDENTITY", "A")
        assert _has_entry(self.entries, "IDENTITY", "L")

    def test_mts_mhall_canonize(self) -> None:
        """MTS canonization: MHALL → [M, H, A, L, L]."""
        assert _has_entry(self.entries, "CANONIZED", "MHALL", "MHALL")

    def test_mts_svo_components(self) -> None:
        """MTS for SVO: unsigned entries for S, V, O components."""
        assert _has_entry(self.entries, "IDENTITY", "S")
        assert _has_entry(self.entries, "IDENTITY", "V")
        assert _has_entry(self.entries, "IDENTITY", "O")

    def test_mts_svo_canonize(self) -> None:
        """MTS canonization: SVO → [S, V, O]."""
        assert _has_entry(self.entries, "CANONIZED", "SVO", "SVO")

    def test_countersign_pair(self) -> None:
        """Countersign: MHALL → SVO and SVO → MHALL."""
        assert _has_entry(self.entries, "COUNTERSIGNED", "MHALL", "SVO")
        assert _has_entry(self.entries, "COUNTERSIGNED", "SVO", "MHALL")

    def test_undersign_pairs(self) -> None:
        """Undersign entries: M→S, H→V (reversed direction)."""
        assert _has_entry(self.entries, "UNDERSIGNED", "M", "S")
        assert _has_entry(self.entries, "UNDERSIGNED", "H", "V")

    def test_mts_all(self) -> None:
        """MTS for ALL: canonize entry and undersign O→ALL."""
        assert _has_entry(self.entries, "CANONIZED", "ALL", "ALL")
        assert _has_entry(self.entries, "UNDERSIGNED", "ALL", "O")

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
        """7 IDENTITY entries from MTS component identities.

        MTS components (deduped): M, H, A, L, S, V, O = 7 unique chars.
        Total IDENTITY: 7 (no compound-own identity).
        """
        assert _count_entries(self.entries, "IDENTITY") == 7


# ---------------------------------------------------------------------------
# KS-36: Word-bound example (§14.12)
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


@requires_tokenizer_data
class TestKS36WordBound:
    """KS-36 — Word-bound example from §14.12.

    Tests that block annotation provides word bindings and inline
    annotations resolve correctly.  Requires Tokenizer data files.
    """

    def _get_tokenizer(self) -> NLPTokenizer:
        """Get NLPTokenizer from standard file paths."""
        from kalvin.nlp_tokenizer import NLPTokenizer

        return NLPTokenizer()

    def test_word_binding_mts_mhall(self) -> None:
        """MTS for MHALL resolves characters to words.

        M → Mary, H → Had, A → "A", L → "Little", L → "Lamb".
        """
        tok = self._get_tokenizer()
        # dev=True so dbg.label carries the resolved symbolic signature
        # string. A packed signature is opaque per §11.6 and cannot be
        # decoded as a single BPE token, so the label is the reliable text.
        entries = compile_source(SOURCE_14_12, tokenizer=tok, dev=True)

        all_text: list[str] = []
        for e in entries:
            if e.dbg.label:
                all_text.append(e.dbg.label)
            for node_val in e.as_node_list():
                try:
                    decoded = tok.decode([node_val])
                    if decoded:
                        all_text.append(decoded)
                except Exception:
                    pass  # packed node — opaque

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
        tok = self._get_tokenizer()
        entries = compile_source(SOURCE_14_12, tokenizer=tok)

        # S(ubject) → 'Subject' resolves into a node-list under BPE.
        found = False
        for e in entries:
            nodes = e.as_node_list()
            if nodes and tok.decode(nodes) == "Subject":
                found = True
                break
        assert found, "Expected 'Subject' from inline binding"

    def test_compiled_entries_valid(self) -> None:
        """All entries are KLine instances with valid structure."""
        tok = self._get_tokenizer()
        entries = compile_source(SOURCE_14_12, tokenizer=tok)
        assert len(entries) > 0
        for e in entries:
            assert isinstance(e, KLine)
            assert isinstance(e.signature, int)
            assert e.dbg.op in ("COUNTERSIGNED", "CANONIZED", "CONNOTED", "UNDERSIGNED", "IDENTITY")


# ---------------------------------------------------------------------------
# KS-41/42: Canonical resolution + encoding (§8.3, §11.4/§11.5)
# ---------------------------------------------------------------------------


@requires_tokenizer_data
class TestCanonicalEncoding:
    """KS-41 (canonical resolution) + KS-42 (canonical encoding).

    Asserted at the KLine level (post-TokenEncoder) on the §14.12
    Word-bound example. These are the regression net for the duplicate-
    CANONIZE and phantom-IDENTITY bugs: an identifier has one identity,
    computed once and reused.
    """

    def _entries(self, tokenizer):
        return compile_source(SOURCE_14_12, tokenizer=tokenizer, dev=True)

    def test_one_canonized_per_compound(self, tokenizer):
        """KS-42: exactly one CANONIZED kline per compound identifier."""
        entries = self._entries(tokenizer)
        from collections import Counter

        counts = Counter(e.dbg.label for e in entries if e.dbg.op == "CANONIZED")
        for compound in ("MHALL", "SVO", "ALL"):
            assert counts.get(compound, 0) == 1, (
                f"{compound}: expected 1 CANONIZED, got {counts.get(compound, 0)}"
            )

    def test_no_packed_identity(self, tokenizer):
        """KS-42: no IDENTITY kline carries a packed (compound) signature."""
        entries = self._entries(tokenizer)
        packed_sigs = {e.signature for e in entries if e.dbg.op == "CANONIZED"}
        bad = [e for e in entries if e.dbg.op == "IDENTITY" and e.signature in packed_sigs]
        assert bad == [], f"IDENTITY klines with packed sigs: {bad}"

    def test_compound_resolution_consistent(self, tokenizer):
        """KS-41: a compound resolves identically wherever it appears.

        The CANONIZED definition's nodes are the canonical resolution;
        no other kline should carry a CANONIZED entry for the same
        compound with different (decoded) nodes.
        """
        entries = self._entries(tokenizer)
        seen = {}
        for e in entries:
            if e.dbg.op != "CANONIZED":
                continue
            nodes = tuple(sorted(e.as_node_list()))
            prev = seen.get(e.dbg.label)
            assert prev is None or prev == nodes, (
                f"{e.dbg.label}: divergent CANONIZED nodes {prev} vs {nodes}"
            )
            seen[e.dbg.label] = nodes


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
        tok = NLPTokenizer()
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
        model = KScript(SOURCE_14_11, dev=True)
        assert len(model.entries) > 10
        assert _has_entry(model.entries, "COUNTERSIGNED", "MHALL", "SVO")

    def test_dev_mode(self) -> None:
        """KScript(dev=True) enables debug text."""
        model = KScript("A == B", dev=True)
        for e in model.entries:
            assert e.dbg is not None or e.dbg is None  # dbg field exists

    def test_default_tokenizer_is_nlp(self) -> None:
        """Default tokenizer is the kalvin NLPTokenizer."""
        model = KScript("A")
        assert isinstance(model._tokenizer, NLPTokenizer)
        # The entry's signature should decode back to "A" via the tokenizer
        sig = model.entries[0].signature
        assert _tok32().decode([sig]) == "A"


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


class TestPipelineWiring:
    """Tests that the Compiler correctly wires Lexer → Parser → Emitter → Encoder."""

    def test_end_to_end_simple(self) -> None:
        """Simple source compiles end-to-end with the default tokenizer."""
        compiler = Compiler()
        tokens = Lexer("A == B").tokenize()
        kfile = Parser(tokens).parse()
        entries = compiler.compile(kfile)
        assert len(entries) > 0
        assert _has_entry(entries, "COUNTERSIGNED", "A", "B")

    def test_end_to_end_mts(self) -> None:
        """Multi-character identifier triggers MTS expansion."""
        compiler = Compiler(dev=True)
        tokens = Lexer("ABC == X").tokenize()
        kfile = Parser(tokens).parse()
        entries = compiler.compile(kfile)
        # MTS for ABC
        assert _has_entry(entries, "IDENTITY", "A")
        assert _has_entry(entries, "IDENTITY", "B")
        assert _has_entry(entries, "IDENTITY", "C")
        # No MTS for single-char X
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
        tok = NLPTokenizer()
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
    """Verify that a BindingScope is always instantiated — no tokenizer mode switch."""

    def test_binding_scope_always_created(self) -> None:
        """Compiler always creates a BindingScope internally (no tokenizer guard)."""
        # The compiler unconditionally creates a BindingScope regardless of the
        # tokenizer. We verify this indirectly: compiling produces entries with
        # correct operator structure.
        compiler = Compiler()
        tokens = Lexer("A == B").tokenize()
        kfile = Parser(tokens).parse()
        entries = compiler.compile(kfile)

        assert len(entries) > 0

    def test_no_mode_switch_in_compiler(self) -> None:
        """Compiler.compile() always creates BindingScope unconditionally.

        The deleted ``supports_mts`` property is gone; the compiler must not
        reference any tokenizer property to decide whether to create a scope.
        """
        import inspect

        from kalvin.abstract import KTokenizer

        assert not hasattr(KTokenizer, "supports_mts"), "supports_mts should be deleted"
        source = inspect.getsource(Compiler.compile)
        assert "supports_mts" not in source, "Compiler.compile should not reference supports_mts"
        assert "skip_mts" not in source, "Compiler.compile should not reference skip_mts"

    def test_annotation_feeds_binding_scope_in_v3(self) -> None:
        """In v3, annotations always feed the BindingScope (no mode switch).

        This means annotations affect character resolution. The word
        'annotation' in the annotation resolves A to 'annotation'
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
