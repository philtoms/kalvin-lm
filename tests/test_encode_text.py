"""Tests for encode_text.py — verifies the script's core logic
against the current KAgent API (tokenize → KLine → rationalise)."""

import importlib
import sys
import tempfile
from pathlib import Path

import pytest

from kalvin.agent import KAgent
from kalvin.events import EventBus
from kalvin.expand import SIG_S4
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from tests.conftest import requires_tokenizer_data

signifier = NLPSignifier()

# Encoding through KAgent uses the kalvin tokenizer; skip cleanly when the
# data assets are absent on a fresh clone.
pytestmark = requires_tokenizer_data

# ── Import helper ─────────────────────────────────────────────────────
# The script lives in scripts/ which is not a package.  We import it
# by adding its parent to sys.path and using importlib.

_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")


@pytest.fixture(autouse=True)
def _ensure_scripts_on_path():
    """Ensure scripts/ is importable for every test in this module."""
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    yield


@pytest.fixture()
def encode_text():
    """Import and return the encode_text module."""
    return importlib.import_module("encode_text")


# ── Tests ─────────────────────────────────────────────────────────────


class TestEncodeSentenceRationalises:
    """Encoding a sentence through the tokenize → KLine → rationalise
    pattern should grow the agent's frame."""

    def test_encode_sentence_rationalises(self):
        agent = KAgent(adapter=EventBus())
        initial = agent.frame_size()

        sentence = "Hello world"
        nodes = agent.tokenizer.encode(sentence)
        kline = KLine(signature=signifier.make_signature(nodes), nodes=nodes)
        agent.rationalise(KValue(kline, SIG_S4))

        assert agent.frame_size() > initial


class TestEncodeEmptyString:
    """An empty string produces no nodes; rationalise should handle it
    gracefully (no crash)."""

    def test_encode_empty_string(self):
        agent = KAgent(adapter=EventBus())
        initial = agent.frame_size()

        nodes = agent.tokenizer.encode("")
        assert nodes == []

        kline = KLine(signature=signifier.make_signature(nodes), nodes=nodes)
        result = agent.rationalise(KValue(kline, SIG_S4))

        # Empty kline → S4 (frame event), size grows by 1
        assert agent.frame_size() == initial + 1
        assert result is True  # S4 is significant


class TestSplitIntoSentences:
    """Verify that split_into_sentences correctly splits on
    sentence-ending punctuation."""

    def test_period_split(self, encode_text):
        result = encode_text.split_into_sentences("Hello. World.")
        assert result == ["Hello.", "World."]

    def test_exclamation_split(self, encode_text):
        result = encode_text.split_into_sentences("Wow! Great.")
        assert result == ["Wow!", "Great."]

    def test_question_split(self, encode_text):
        result = encode_text.split_into_sentences("Why? Because.")
        assert result == ["Why?", "Because."]

    def test_single_sentence(self, encode_text):
        result = encode_text.split_into_sentences("Just one sentence")
        assert result == ["Just one sentence"]

    def test_empty_string(self, encode_text):
        result = encode_text.split_into_sentences("")
        assert result == []

    def test_whitespace_only(self, encode_text):
        result = encode_text.split_into_sentences("   ")
        assert result == []


class TestEncodeMultipleSentences:
    """Encode a multi-sentence string and verify the frame grows by
    the expected count."""

    def test_encode_multiple_sentences(self):
        # Use the kalvin NLPTokenizer (the sole production tokenizer).
        agent = KAgent(adapter=EventBus(), tokenizer=NLPTokenizer())
        initial = agent.frame_size()

        text = "The cat sat. The dog ran."
        sentences = [
            s.strip() for s in __import__("re").split(r"(?<=[.!?])\s+", text.strip()) if s.strip()
        ]
        assert len(sentences) >= 2

        for sentence in sentences:
            nodes = agent.tokenizer.encode(sentence)
            kline = KLine(signature=signifier.make_signature(nodes), nodes=nodes)
            agent.rationalise(KValue(kline, SIG_S4))

        # Signatures are OR-reductions of node values, so similar sentences
        # (which share characters/tokens) overlap. rationalise() therefore
        # adds only the first novel signature to the frame and queues the
        # overlapping ones for cogitation — so encoding multiple sentences
        # grows the frame by at least one entry rather than one per sentence.
        assert agent.frame_size() > initial


class TestAgentLoadSaveRoundtrip:
    """Create agent, encode text, save via AgentCodec, load via
    AgentCodec, verify model size matches."""

    def test_roundtrip_after_encoding(self):
        agent = KAgent(adapter=EventBus())

        # Encode a sentence
        nodes = agent.tokenizer.encode("Test sentence for roundtrip")
        kline = KLine(signature=signifier.make_signature(nodes), nodes=nodes)
        agent.rationalise(KValue(kline, SIG_S4))

        size_before = agent.frame_size()
        assert size_before > 0

        # Save to binary
        data = agent.to_bytes()
        assert len(data) > 0

        # Load back
        loaded = KAgent.from_bytes(data)
        assert loaded.frame_size() == size_before

    def test_roundtrip_json_after_encoding(self):
        agent = KAgent(adapter=EventBus())

        # Encode a sentence
        nodes = agent.tokenizer.encode("JSON roundtrip test")
        kline = KLine(signature=signifier.make_signature(nodes), nodes=nodes)
        agent.rationalise(KValue(kline, SIG_S4))

        size_before = agent.frame_size()

        # Save to dict (JSON)
        d = agent.to_dict()
        loaded = KAgent.from_dict(d)
        assert loaded.frame_size() == size_before

    def test_roundtrip_file_after_encoding(self):
        agent = KAgent(adapter=EventBus())

        nodes = agent.tokenizer.encode("File roundtrip test")
        kline = KLine(signature=signifier.make_signature(nodes), nodes=nodes)
        agent.rationalise(KValue(kline, SIG_S4))

        size_before = agent.frame_size()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test-agent.json"
            agent.save(path, format="json")
            loaded = KAgent.load(path, format="json")
            assert loaded.frame_size() == size_before
