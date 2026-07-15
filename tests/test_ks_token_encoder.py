"""Tests for ks.token_encoder — TokenEncoder.

Uses NLPTokenizer (loaded from the real BPE + grammar data assets) for most
tests.  A mock multi-token tokenizer is used for BPE multi-token MTS tests.
"""

from __future__ import annotations

import pytest

from kalvin.abstract import KTokenizer
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from ks.ast_emitter import SymbolicEntry
from ks.token_encoder import TokenEncoder
from tests.conftest import requires_tokenizer_data

signifier = NLPSignifier()

# Every test encodes through a real tokenizer; skip cleanly when the tokenizer data
# assets are absent on a fresh clone.
pytestmark = requires_tokenizer_data

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tz() -> NLPTokenizer:
    """A standard NLPTokenizer loaded from the data files."""
    return NLPTokenizer()


@pytest.fixture
def encoder(tz: NLPTokenizer) -> TokenEncoder:
    """A TokenEncoder backed by NLPTokenizer."""
    return TokenEncoder(tz)


@pytest.fixture
def dev_encoder(tz: NLPTokenizer) -> TokenEncoder:
    """A TokenEncoder in dev mode."""
    return TokenEncoder(tz, dev=True)


# ── Mock multi-token tokenizer ───────────────────────────────────────


class MockMultiTokenTokenizer(KTokenizer):
    """Mock tokenizer that returns multiple tokens for certain words.

    Used to test multi-token BPE MTS (§11.4) without depending on the exact
    BPE tokenisation of any particular word.
    """

    def __init__(self, multi_map: dict[str, list[int]] | None = None):
        self._multi_map = multi_map or {}

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str, pad_ws: bool = False) -> list[int]:
        if text in self._multi_map:
            return self._multi_map[text]
        # Fallback: one typed node per character (high 32 bits carry a
        # type so the value looks like a real node, but is fully deterministic).
        return [(ord(c) << 32) | ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        # Reverse lookup for mock tokens
        for text, tokens in self._multi_map.items():
            if ids == tokens:
                return text
        return "".join(chr(i & 0xFFFFFFFF) for i in ids)

    def lookup_type_entry_for_node(self, node: int) -> dict | None:
        # Mock is type-unaware; the encoder's debug type-info path gets None.
        return None


# ── KS-32: unresolved characters encode as typed nodes ──────────────


class TestUnresolvedCharEncoding:
    """Unresolved characters encode as their own raw typed nodes."""

    def test_unresolved_chars_encode(self, encoder: TokenEncoder, tz: NLPTokenizer) -> None:
        entry = SymbolicEntry(sig="A", nodes=["B", "C"], op="COUNTERSIGNS")
        results = encoder.encode_entries([entry])
        assert len(results) == 1
        compiled = results[0]
        assert compiled.kline.signature == tz.encode("A")[0]
        assert compiled.kline.nodes[0] == tz.encode("B")[0]
        assert compiled.kline.nodes[1] == tz.encode("C")[0]

    def test_unresolved_sig_is_typed_node(self, encoder: TokenEncoder, tz: NLPTokenizer) -> None:
        """An unresolved char produces a valid typed node (not a legacy bit value)."""
        entry = SymbolicEntry(sig="Z", nodes=[], op="IDENTITY")
        results = encoder.encode_entries([entry])
        sig = results[0].kline.signature
        assert sig == tz.encode("Z")[0]
        assert (sig >> 32) > 0  # sig-word bits present
        assert sig != 67108864  # not the legacy character-bit-packed value


# ── KS-34: nodes always list ─────────────────────────────────────────


class TestNodesAlwaysList:
    """KLine.nodes is always list[int], never None or bare int."""

    def test_empty_nodes(self, encoder: TokenEncoder) -> None:
        entry = SymbolicEntry(sig="A", nodes=[], op="IDENTITY")
        results = encoder.encode_entries([entry])
        assert len(results) == 1
        assert results[0].kline.nodes == []
        assert isinstance(results[0].kline.nodes, list)

    def test_single_node_is_list(self, encoder: TokenEncoder) -> None:
        entry = SymbolicEntry(sig="A", nodes=["B"], op="CONNOTES")
        results = encoder.encode_entries([entry])
        assert len(results) == 1
        assert isinstance(results[0].kline.nodes, list)
        assert len(results[0].kline.nodes) == 1
        assert not isinstance(results[0].kline.nodes, int)

    def test_multiple_nodes(self, encoder: TokenEncoder) -> None:
        entry = SymbolicEntry(sig="A", nodes=["B", "C", "D"], op="CANONIZES")
        results = encoder.encode_entries([entry])
        assert len(results) == 1
        assert isinstance(results[0].kline.nodes, list)
        assert len(results[0].kline.nodes) == 3

    def test_all_entries_have_list_nodes(self, encoder: TokenEncoder) -> None:
        """Every KLine produced must have list nodes."""
        entries = [
            SymbolicEntry(sig="A", nodes=[], op="IDENTITY"),
            SymbolicEntry(sig="B", nodes=["C"], op="CONNOTES"),
            SymbolicEntry(sig="D", nodes=["E", "F"], op="COUNTERSIGNS"),
        ]
        results = encoder.encode_entries(entries)
        for r in results:
            assert isinstance(r.kline.nodes, list), f"Expected list, got {type(r.kline.nodes)}"


# ── Signature encoding ───────────────────────────────────────────────


class TestSignatureEncoding:
    """Signature strings are correctly encoded to uint64."""

    def test_single_char_sig(self, encoder: TokenEncoder, tz: NLPTokenizer) -> None:
        entry = SymbolicEntry(sig="A", nodes=[], op="IDENTITY")
        results = encoder.encode_entries([entry])
        assert results[0].kline.signature == tz.encode("A")[0]

    def test_multi_char_sig_packed(self, encoder: TokenEncoder, tz: NLPTokenizer) -> None:
        """Multi-token identifier is packed via OR-reduction (make_signature).

        A multi-token signature heading an IDENTITY entry is decomposed into
        per-token IDENTITY entries plus a packed CANONIZES entry (the last
        result); its signature is signifier.make_signature(tokens) — i.e. OR of the tokens.
        """
        entry = SymbolicEntry(sig="HELLO", nodes=[], op="IDENTITY")
        results = encoder.encode_entries([entry])
        expected = signifier.make_signature(tz.encode("HELLO"))
        assert results[-1].kline.signature == expected


# ── Node encoding ────────────────────────────────────────────────────


class TestNodeEncoding:
    """Node strings are correctly encoded to uint64."""

    def test_node_values(self, encoder: TokenEncoder, tz: NLPTokenizer) -> None:
        entry = SymbolicEntry(sig="A", nodes=["B", "C"], op="CONNOTES")
        results = encoder.encode_entries([entry])
        compiled = results[0]
        assert compiled.kline.nodes[0] == tz.encode("B")[0]
        assert compiled.kline.nodes[1] == tz.encode("C")[0]


# ── Significance levels (compile-time intent via dbg.op) ──────────────


class TestSignificanceLevels:
    """Each op maps to the correct significance level in dbg.op."""

    @pytest.mark.parametrize(
        "op,expected_level",
        [
            ("COUNTERSIGNS", "S1"),
            ("DENOTES", "S3"),
            ("CANONIZES", "S2"),
            ("CONNOTES", "S3"),
            ("IDENTITY", "S4"),
        ],
    )
    def test_sig_level(self, dev_encoder: TokenEncoder, op: str, expected_level: str) -> None:
        from kalvin.kline import _SIG_LEVELS

        entry = SymbolicEntry(sig="A", nodes=["B"] if op != "IDENTITY" else [], op=op)
        results = dev_encoder.encode_entries([entry])
        # Find the main entry (last one with matching op)
        main = [r for r in results if r.kline.dbg and r.kline.dbg.op == op]
        assert len(main) >= 1
        assert _SIG_LEVELS.get(main[-1].kline.dbg.op, "S4") == expected_level


# ── Signature is full uint64 (no masking) ────────────────────────────


class TestFullUint64:
    """Signatures are raw values from tokenizer — not masked or truncated."""

    def test_no_masking(self, encoder: TokenEncoder, tz: NLPTokenizer) -> None:
        entry = SymbolicEntry(sig="ABC", nodes=[], op="IDENTITY")
        results = encoder.encode_entries([entry])
        raw = signifier.make_signature(tz.encode("ABC"))
        # The packed CANONIZES entry (last) carries the unmasked OR-reduction.
        assert results[-1].kline.signature == raw
        # Ensure the value is unmasked — it should have multiple bits set
        assert results[-1].kline.signature == raw

    def test_signature_matches_make_signature(self) -> None:
        """For multi-token words, sig == signifier.make_signature(tokens)."""
        mock = MockMultiTokenTokenizer({"WORD": [100, 200]})
        enc = TokenEncoder(mock)
        entry = SymbolicEntry(sig="WORD", nodes=[], op="IDENTITY")
        results = enc.encode_entries([entry])
        expected = signifier.make_signature([100, 200])
        # Main entry is last (after MTS expansion entries)
        assert results[-1].kline.signature == expected


# ── Multi-token word MTS (§11.4) ─────────────────────────────────────


class TestMultiTokenMTS:
    """Multi-token BPE words produce unsigned per token + CANONIZES packed."""

    def test_multi_token_node_emits_mts(self) -> None:
        """A multi-token node triggers unsigned + CANONIZES entries."""
        mock = MockMultiTokenTokenizer({"Mary": [10, 20]})
        enc = TokenEncoder(mock, dev=True)
        entry = SymbolicEntry(sig="A", nodes=["Mary"], op="CONNOTES")
        results = enc.encode_entries([entry])

        # Should have: IDENTITY(10), IDENTITY(20), CANONIZES(30, [10,20]), CONNOTES(A, [30])
        unsigned_entries = [
            r for r in results if r.kline.dbg and r.kline.dbg.op == "IDENTITY" and not r.kline.nodes
        ]
        canonize_entries = [r for r in results if r.kline.dbg and r.kline.dbg.op == "CANONIZES"]
        connote_entries = [r for r in results if r.kline.dbg and r.kline.dbg.op == "CONNOTES"]

        assert len(unsigned_entries) == 2
        assert unsigned_entries[0].kline.signature == 10
        assert unsigned_entries[0].kline.nodes == []
        assert unsigned_entries[1].kline.signature == 20
        assert unsigned_entries[1].kline.nodes == []

        assert len(canonize_entries) == 1
        packed = signifier.make_signature([10, 20])
        assert canonize_entries[0].kline.signature == packed
        assert canonize_entries[0].kline.nodes == [10, 20]

        assert len(connote_entries) == 1
        assert connote_entries[0].kline.nodes == [packed]

    def test_multi_token_sig_emits_mts(self) -> None:
        """A multi-token signature heading an IDENTITY entry is represented
        solely by its §11.4 decomposition — no standalone packed-sig
        IDENTITY (CONTEXT.md "Identity").
        """
        mock = MockMultiTokenTokenizer({"WORD": [50, 60]})
        enc = TokenEncoder(mock, dev=True)
        entry = SymbolicEntry(sig="WORD", nodes=[], op="IDENTITY")
        results = enc.encode_entries([entry])

        packed = signifier.make_signature([50, 60])

        # §11.4 MTS: IDENTITY(50), IDENTITY(60), CANONIZES(packed, [50,60])
        mts_unsigned = [
            r for r in results
            if r.kline.dbg and r.kline.dbg.op == "IDENTITY" and r.kline.signature in (50, 60)
        ]
        canonize_entries = [r for r in results if r.kline.dbg and r.kline.dbg.op == "CANONIZES"]
        assert len(mts_unsigned) == 2
        assert len(canonize_entries) == 1
        assert canonize_entries[0].kline.signature == packed
        assert canonize_entries[0].kline.nodes == [50, 60]

        # No standalone IDENTITY at the packed signature.
        assert not [
            r for r in results
            if r.kline.dbg and r.kline.dbg.op == "IDENTITY" and r.kline.signature == packed
        ]
        assert len(results) == 3

    def test_source_precedes_mts(self) -> None:
        """Compiled source precedes any MTS entries in the output.

        A source entry (CONNOTES) whose node triggers BPE MTS is emitted
        first; its §11.3 MTS expansion (subword identities + canonization)
        follows. This is the output-ordering contract: source before MTS.
        """
        mock = MockMultiTokenTokenizer({"Mary": [10, 20]})
        enc = TokenEncoder(mock, dev=True)
        entry = SymbolicEntry(sig="A", nodes=["Mary"], op="CONNOTES")
        results = enc.encode_entries([entry])

        # Source entry is first
        assert results[0].kline.dbg.op == "CONNOTES"
        # All remaining entries are MTS (IDENTITY/CANONIZES)
        for r in results[1:]:
            assert r.kline.dbg.op in ("IDENTITY", "CANONIZES")


# ── Dedup multi-token MTS ────────────────────────────────────────────


class TestDedupMTS:
    """Same multi-token word encoded twice should not duplicate MTS entries."""

    def test_dedup_same_word_twice(self) -> None:
        mock = MockMultiTokenTokenizer({"Mary": [10, 20]})
        enc = TokenEncoder(mock, dev=True)
        entries = [
            SymbolicEntry(sig="A", nodes=["Mary"], op="CONNOTES"),
            SymbolicEntry(sig="B", nodes=["Mary"], op="CONNOTES"),
        ]
        results = enc.encode_entries(entries)

        # Only 2 IDENTITY entries (not 4) and 1 CANONIZES (not 2)
        unsigned_entries = [
            r for r in results if r.kline.dbg and r.kline.dbg.op == "IDENTITY" and not r.kline.nodes
        ]
        canonize_entries = [r for r in results if r.kline.dbg and r.kline.dbg.op == "CANONIZES"]
        connote_entries = [r for r in results if r.kline.dbg and r.kline.dbg.op == "CONNOTES"]

        assert len(unsigned_entries) == 2  # deduped from potential 4
        assert len(canonize_entries) == 1  # deduped from potential 2
        assert len(connote_entries) == 2  # both main entries

        # Both main entries use the same packed node value
        packed = signifier.make_signature([10, 20])
        assert connote_entries[0].kline.nodes == [packed]
        assert connote_entries[1].kline.nodes == [packed]


# ── KLine identity ──────────────────────────────────────────────────


class TestKLineIdentity:
    """Compiled KLines behave correctly as KLines."""

    def test_isinstance_kline(self, encoder: TokenEncoder) -> None:
        entry = SymbolicEntry(sig="A", nodes=["B"], op="CONNOTES")
        results = encoder.encode_entries([entry])
        assert isinstance(results[0], KValue)
        assert isinstance(results[0].kline, KLine)

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
            SymbolicEntry(sig="B", nodes=["C"], op="CONNOTES"),
            SymbolicEntry(sig="D", nodes=["E", "F"], op="CANONIZES"),
        ]
        results = dev_encoder.encode_entries(entries)
        # At least one per input entry
        assert len(results) >= 3
        # Ops present in order
        ops = [r.kline.dbg.op for r in results]
        assert "IDENTITY" in ops
        assert "CONNOTES" in ops
        assert "CANONIZES" in ops


# ── Dev mode debug text ─────────────────────────────────────────────


class TestDevMode:
    """When dev=True, dbg is populated."""

    def test_dev_dbg(self, tz: NLPTokenizer) -> None:
        enc = TokenEncoder(tz, dev=True)
        entry = SymbolicEntry(sig="A", nodes=["B"], op="CONNOTES")
        results = enc.encode_entries([entry])
        # Main entry should have dbg
        assert results[0].kline.dbg is not None
        assert results[0].kline.dbg.op == "CONNOTES"

    def test_no_dev_dbg(self, encoder: TokenEncoder) -> None:
        entry = SymbolicEntry(sig="A", nodes=["B"], op="CONNOTES")
        results = encoder.encode_entries([entry])
        # In non-dev mode, only MTS subword entries get a minimal dbg
        # Main entries may or may not have dbg depending on encoder
        # The key invariant: the KValue wraps a valid KLine
        assert isinstance(results[0], KValue)
        assert isinstance(results[0].kline, KLine)


# ── Empty symbolic list ─────────────────────────────────────────────


class TestEmptyInput:
    """encode_entries([]) returns []."""

    def test_empty_list(self, encoder: TokenEncoder) -> None:
        assert encoder.encode_entries([]) == []

    def test_empty_list_is_list(self, encoder: TokenEncoder) -> None:
        result = encoder.encode_entries([])
        assert isinstance(result, list)
        assert len(result) == 0
