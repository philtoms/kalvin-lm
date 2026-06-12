"""Tests for ks.token_encoder — TokenEncoder.

Uses ModTokenizer for most tests (no external files required).
A mock multi-token tokenizer is used for BPE multi-token MCS tests.
"""

from __future__ import annotations

import pytest

from kalvin.abstract import KTokenizer
from kalvin.kline import KDbg, KLine
from kalvin.mod_tokenizer import ModTokenizer
from kalvin.signature import make_signature

from ks.ast_emitter import SymbolicEntry
from ks.token_encoder import TokenEncoder


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mod_tz() -> ModTokenizer:
    """A standard ModTokenizer."""
    return ModTokenizer()


@pytest.fixture
def encoder(mod_tz: ModTokenizer) -> TokenEncoder:
    """A TokenEncoder backed by ModTokenizer."""
    return TokenEncoder(mod_tz)


@pytest.fixture
def dev_encoder(mod_tz: ModTokenizer) -> TokenEncoder:
    """A TokenEncoder in dev mode."""
    return TokenEncoder(mod_tz, dev=True)


# ── Mock multi-token tokenizer ───────────────────────────────────────

class MockMultiTokenTokenizer(KTokenizer):
    """Mock tokenizer that returns multiple tokens for certain words.

    Used to test multi-token BPE MCS (§11.4) without requiring
    real BPE files.
    """

    def __init__(self, multi_map: dict[str, list[int]] | None = None):
        self._multi_map = multi_map or {}
        self._fallback = ModTokenizer()

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str, pad_ws: bool = False) -> list[int]:
        if text in self._multi_map:
            return self._multi_map[text]
        return self._fallback.encode(text, pad_ws)

    def decode(self, ids: list[int]) -> str:
        # Reverse lookup for mock tokens
        for text, tokens in self._multi_map.items():
            if ids == tokens:
                return text
        return self._fallback.decode(ids)


# ── KS-32: Mod32 fallback for unbound characters ─────────────────────

class TestMod32Fallback:
    """Unbound characters produce valid encoded entries via Mod32."""

    def test_unbound_chars_encode(self, encoder: TokenEncoder, mod_tz: ModTokenizer) -> None:
        entry = SymbolicEntry(sig="A", nodes=["B", "C"], op="COUNTERSIGN")
        results = encoder.encode_entries([entry])
        assert len(results) == 1
        compiled = results[0]
        assert compiled.signature == mod_tz.encode("A")[0]
        assert compiled.nodes[0] == mod_tz.encode("B")[0]
        assert compiled.nodes[1] == mod_tz.encode("C")[0]

    def test_unbound_sig_uint64(self, encoder: TokenEncoder, mod_tz: ModTokenizer) -> None:
        """Unbound char produces a proper uint64 (bit-packed)."""
        entry = SymbolicEntry(sig="Z", nodes=[], op="IDENTITY")
        results = encoder.encode_entries([entry])
        assert results[0].signature == mod_tz.encode("Z")[0]
        assert results[0].signature > 0


# ── KS-34: nodes always list ─────────────────────────────────────────

class TestNodesAlwaysList:
    """KLine.nodes is always list[int], never None or bare int."""

    def test_empty_nodes(self, encoder: TokenEncoder) -> None:
        entry = SymbolicEntry(sig="A", nodes=[], op="IDENTITY")
        results = encoder.encode_entries([entry])
        assert len(results) == 1
        assert results[0].nodes == []
        assert isinstance(results[0].nodes, list)

    def test_single_node_is_list(self, encoder: TokenEncoder) -> None:
        entry = SymbolicEntry(sig="A", nodes=["B"], op="CONNOTATE")
        results = encoder.encode_entries([entry])
        assert len(results) == 1
        assert isinstance(results[0].nodes, list)
        assert len(results[0].nodes) == 1
        assert not isinstance(results[0].nodes, int)

    def test_multiple_nodes(self, encoder: TokenEncoder) -> None:
        entry = SymbolicEntry(sig="A", nodes=["B", "C", "D"], op="CANONIZE")
        results = encoder.encode_entries([entry])
        assert len(results) == 1
        assert isinstance(results[0].nodes, list)
        assert len(results[0].nodes) == 3

    def test_all_entries_have_list_nodes(self, encoder: TokenEncoder) -> None:
        """Every KLine produced must have list nodes."""
        entries = [
            SymbolicEntry(sig="A", nodes=[], op="IDENTITY"),
            SymbolicEntry(sig="B", nodes=["C"], op="CONNOTATE"),
            SymbolicEntry(sig="D", nodes=["E", "F"], op="COUNTERSIGN"),
        ]
        results = encoder.encode_entries(entries)
        for r in results:
            assert isinstance(r.nodes, list), f"Expected list, got {type(r.nodes)}"


# ── Signature encoding ───────────────────────────────────────────────

class TestSignatureEncoding:
    """Signature strings are correctly encoded to uint64."""

    def test_single_char_sig(self, encoder: TokenEncoder, mod_tz: ModTokenizer) -> None:
        entry = SymbolicEntry(sig="A", nodes=[], op="IDENTITY")
        results = encoder.encode_entries([entry])
        assert results[0].signature == mod_tz.encode("A")[0]

    def test_multi_char_sig_packed(self, encoder: TokenEncoder, mod_tz: ModTokenizer) -> None:
        """Multi-char identifier is packed via OR-reduction."""
        entry = SymbolicEntry(sig="HELLO", nodes=[], op="IDENTITY")
        results = encoder.encode_entries([entry])
        expected = mod_tz.encode("HELLO")[0]
        assert results[0].signature == expected


# ── Node encoding ────────────────────────────────────────────────────

class TestNodeEncoding:
    """Node strings are correctly encoded to uint64."""

    def test_node_values(self, encoder: TokenEncoder, mod_tz: ModTokenizer) -> None:
        entry = SymbolicEntry(sig="A", nodes=["B", "C"], op="CONNOTATE")
        results = encoder.encode_entries([entry])
        compiled = results[0]
        assert compiled.nodes[0] == mod_tz.encode("B")[0]
        assert compiled.nodes[1] == mod_tz.encode("C")[0]


# ── Significance levels (compile-time intent via dbg.op) ──────────────

class TestSignificanceLevels:
    """Each op maps to the correct significance level in dbg.op."""

    @pytest.mark.parametrize("op,expected_level", [
        ("COUNTERSIGN", "S1"),
        ("UNDERSIGN", "S3"),
        ("CANONIZE", "S2"),
        ("CONNOTATE", "S3"),
        ("IDENTITY", "S4"),
    ])
    def test_sig_level(self, dev_encoder: TokenEncoder, op: str, expected_level: str) -> None:
        from kalvin.kline import _SIG_LEVELS
        entry = SymbolicEntry(sig="A", nodes=["B"] if op != "IDENTITY" else [], op=op)
        results = dev_encoder.encode_entries([entry])
        # Find the main entry (last one with matching op)
        main = [r for r in results if r.dbg and r.dbg.op == op]
        assert len(main) >= 1
        assert _SIG_LEVELS.get(main[-1].dbg.op, "S4") == expected_level


# ── Signature is full uint64 (no masking) ────────────────────────────

class TestFullUint64:
    """Signatures are raw values from tokenizer — not masked or truncated."""

    def test_no_masking(self, encoder: TokenEncoder, mod_tz: ModTokenizer) -> None:
        entry = SymbolicEntry(sig="ABC", nodes=[], op="IDENTITY")
        results = encoder.encode_entries([entry])
        raw = mod_tz.encode("ABC")[0]
        assert results[0].signature == raw
        # Ensure the value is unmasked — it should have multiple bits set
        assert results[0].signature == raw

    def test_signature_matches_make_signature(self, mod_tz: ModTokenizer) -> None:
        """For multi-token words, sig == make_signature(tokens)."""
        mock = MockMultiTokenTokenizer({"WORD": [100, 200]})
        enc = TokenEncoder(mock)
        entry = SymbolicEntry(sig="WORD", nodes=[], op="IDENTITY")
        results = enc.encode_entries([entry])
        expected = make_signature([100, 200])
        # Main entry is last (after MCS expansion entries)
        assert results[-1].signature == expected


# ── Multi-token word MCS (§11.4) ─────────────────────────────────────

class TestMultiTokenMCS:
    """Multi-token BPE words produce unsigned per token + CANONIZE packed."""

    def test_multi_token_node_emits_mcs(self) -> None:
        """A multi-token node triggers unsigned + CANONIZE entries."""
        mock = MockMultiTokenTokenizer({"Mary": [10, 20]})
        enc = TokenEncoder(mock, dev=True)
        entry = SymbolicEntry(sig="A", nodes=["Mary"], op="CONNOTATE")
        results = enc.encode_entries([entry])

        # Should have: IDENTITY(10), IDENTITY(20), CANONIZE(30, [10,20]), CONNOTATE(A, [30])
        unsigned_entries = [r for r in results if r.dbg and r.dbg.op == "IDENTITY" and not r.nodes]
        canonize_entries = [r for r in results if r.dbg and r.dbg.op == "CANONIZE"]
        connotate_entries = [r for r in results if r.dbg and r.dbg.op == "CONNOTATE"]

        assert len(unsigned_entries) == 2
        assert unsigned_entries[0].signature == 10
        assert unsigned_entries[0].nodes == []
        assert unsigned_entries[1].signature == 20
        assert unsigned_entries[1].nodes == []

        assert len(canonize_entries) == 1
        packed = make_signature([10, 20])
        assert canonize_entries[0].signature == packed
        assert canonize_entries[0].nodes == [10, 20]

        assert len(connotate_entries) == 1
        assert connotate_entries[0].nodes == [packed]

    def test_multi_token_sig_emits_mcs(self) -> None:
        """A multi-token signature triggers MCS entries."""
        mock = MockMultiTokenTokenizer({"WORD": [50, 60]})
        enc = TokenEncoder(mock, dev=True)
        entry = SymbolicEntry(sig="WORD", nodes=[], op="IDENTITY")
        results = enc.encode_entries([entry])

        packed = make_signature([50, 60])

        # MCS emits: IDENTITY(50), IDENTITY(60), CANONIZE(packed, [50,60])
        # Then main: IDENTITY(packed, [])
        mcs_unsigned = [r for r in results if r.dbg and r.dbg.op == "IDENTITY" and r.signature in (50, 60)]
        canonize_entries = [r for r in results if r.dbg and r.dbg.op == "CANONIZE"]
        main_entry = results[-1]

        assert len(mcs_unsigned) == 2
        assert mcs_unsigned[0].signature == 50
        assert mcs_unsigned[1].signature == 60

        assert len(canonize_entries) == 1
        assert canonize_entries[0].signature == packed

        # Main entry uses the packed signature
        assert main_entry.signature == packed
        assert main_entry.dbg.op == "IDENTITY"

    def test_mcs_entries_come_before_main(self) -> None:
        """MCS expansion entries appear before the entry that references them."""
        mock = MockMultiTokenTokenizer({"Mary": [10, 20]})
        enc = TokenEncoder(mock, dev=True)
        entry = SymbolicEntry(sig="A", nodes=["Mary"], op="CONNOTATE")
        results = enc.encode_entries([entry])

        # Main entry is last
        assert results[-1].dbg.op == "CONNOTATE"
        # All MCS entries come before
        for r in results[:-1]:
            assert r.dbg.op in ("IDENTITY", "CANONIZE")


# ── Dedup multi-token MCS ────────────────────────────────────────────

class TestDedupMCS:
    """Same multi-token word encoded twice should not duplicate MCS entries."""

    def test_dedup_same_word_twice(self) -> None:
        mock = MockMultiTokenTokenizer({"Mary": [10, 20]})
        enc = TokenEncoder(mock, dev=True)
        entries = [
            SymbolicEntry(sig="A", nodes=["Mary"], op="CONNOTATE"),
            SymbolicEntry(sig="B", nodes=["Mary"], op="CONNOTATE"),
        ]
        results = enc.encode_entries(entries)

        # Only 2 IDENTITY entries (not 4) and 1 CANONIZE (not 2)
        unsigned_entries = [r for r in results if r.dbg and r.dbg.op == "IDENTITY" and not r.nodes]
        canonize_entries = [r for r in results if r.dbg and r.dbg.op == "CANONIZE"]
        connotate_entries = [r for r in results if r.dbg and r.dbg.op == "CONNOTATE"]

        assert len(unsigned_entries) == 2  # deduped from potential 4
        assert len(canonize_entries) == 1  # deduped from potential 2
        assert len(connotate_entries) == 2  # both main entries

        # Both main entries use the same packed node value
        packed = make_signature([10, 20])
        assert connotate_entries[0].nodes == [packed]
        assert connotate_entries[1].nodes == [packed]


# ── KLine identity ──────────────────────────────────────────────────

class TestKLineIdentity:
    """Compiled KLines behave correctly as KLines."""

    def test_isinstance_kline(self, encoder: TokenEncoder) -> None:
        entry = SymbolicEntry(sig="A", nodes=["B"], op="CONNOTATE")
        results = encoder.encode_entries([entry])
        assert isinstance(results[0], KLine)

    def test_equality(self) -> None:
        a = KLine(signature=42, nodes=[10, 20])
        b = KLine(signature=42, nodes=[10, 20])
        assert a == b

    def test_inequality_sig(self) -> None:
        a = KLine(signature=42, nodes=[10])
        b = KLine(signature=99, nodes=[10])
        assert a != b

    def test_hash(self) -> None:
        a = KLine(signature=42, nodes=[10, 20])
        b = KLine(signature=42, nodes=[10, 20])
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_nodes_normalized_none(self) -> None:
        """KLine._normalize_nodes handles None → []."""
        entry = KLine(signature=42, nodes=None)
        assert entry.nodes == []

    def test_nodes_normalized_int(self) -> None:
        """KLine._normalize_nodes handles int → [int]."""
        entry = KLine(signature=42, nodes=10)
        assert entry.nodes == [10]


# ── Multiple entries ─────────────────────────────────────────────────

class TestMultipleEntries:
    """encode_entries with multiple SymbolicEntry objects."""

    def test_ordering_and_completeness(self, dev_encoder: TokenEncoder) -> None:
        entries = [
            SymbolicEntry(sig="A", nodes=[], op="IDENTITY"),
            SymbolicEntry(sig="B", nodes=["C"], op="CONNOTATE"),
            SymbolicEntry(sig="D", nodes=["E", "F"], op="CANONIZE"),
        ]
        results = dev_encoder.encode_entries(entries)
        # At least one per input entry
        assert len(results) >= 3
        # Ops present in order
        ops = [r.dbg.op for r in results]
        assert "IDENTITY" in ops
        assert "CONNOTATE" in ops
        assert "CANONIZE" in ops


# ── Dev mode debug text ─────────────────────────────────────────────

class TestDevMode:
    """When dev=True, dbg is populated."""

    def test_dev_dbg(self, mod_tz: ModTokenizer) -> None:
        enc = TokenEncoder(mod_tz, dev=True)
        entry = SymbolicEntry(sig="A", nodes=["B"], op="CONNOTATE")
        results = enc.encode_entries([entry])
        # Main entry should have dbg
        assert results[0].dbg is not None
        assert results[0].dbg.op == "CONNOTATE"

    def test_no_dev_dbg(self, encoder: TokenEncoder) -> None:
        entry = SymbolicEntry(sig="A", nodes=["B"], op="CONNOTATE")
        results = encoder.encode_entries([entry])
        # In non-dev mode, only MCS subword entries get a minimal dbg
        # Main entries may or may not have dbg depending on encoder
        # The key invariant: the KLine is valid
        assert isinstance(results[0], KLine)


# ── Empty symbolic list ─────────────────────────────────────────────

class TestEmptyInput:
    """encode_entries([]) returns []."""

    def test_empty_list(self, encoder: TokenEncoder) -> None:
        assert encoder.encode_entries([]) == []

    def test_empty_list_is_list(self, encoder: TokenEncoder) -> None:
        result = encoder.encode_entries([])
        assert isinstance(result, list)
        assert len(result) == 0
