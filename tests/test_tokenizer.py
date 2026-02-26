"""Tests for Tokenizer class."""

import json
import pytest
import tempfile
from pathlib import Path

from kalvin.tokenizer import (
    Tokenizer,
    TokenizerNotTrainedError,
    RustbpeNotInstalledError,
    TiktokenNotInstalledError,
)


# Check for optional dependencies
try:
    import rustbpe

    HAS_RUSTBPE = True
except ImportError:
    HAS_RUSTBPE = False

try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


@pytest.fixture
def trained_tokenizer():
    """Create a trained tokenizer for testing."""
    if not HAS_RUSTBPE:
        pytest.skip("rustbpe not installed")
    tokenizer = Tokenizer()
    texts = [
        "Hello, world!",
        "Hello there!",
        "The world is beautiful.",
        "Testing the tokenizer.",
    ]
    # vocab_size must be at least 256 for rustbpe
    tokenizer.train(texts, vocab_size=256)
    return tokenizer


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestTokenizerInit:
    """Tests for Tokenizer initialization."""

    def test_init_empty(self):
        """Tokenizer can be initialized without a pre-trained tokenizer."""
        tokenizer = Tokenizer()
        assert tokenizer._tokenizer is None

    def test_init_with_word_aligned_default(self):
        """word_aligned defaults to False."""
        tokenizer = Tokenizer()
        assert tokenizer.word_aligned is False

    def test_init_with_word_aligned_true(self):
        """word_aligned can be set to True."""
        tokenizer = Tokenizer(word_aligned=True)
        assert tokenizer.word_aligned is True

    def test_word_aligned_property_setter(self):
        """word_aligned can be changed after initialization."""
        tokenizer = Tokenizer()
        assert tokenizer.word_aligned is False
        tokenizer.word_aligned = True
        assert tokenizer.word_aligned is True


class TestTokenizerTrain:
    """Tests for tokenizer training."""

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_train_basic(self):
        """Tokenizer can be trained on basic texts."""
        tokenizer = Tokenizer()
        texts = ["hello world", "hello there"]
        tokenizer.train(texts, vocab_size=256)
        assert tokenizer.vocab_size > 0

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_train_with_custom_vocab_size(self):
        """Tokenizer respects vocab_size parameter."""
        tokenizer = Tokenizer()
        texts = ["hello world " * 100]
        tokenizer.train(texts, vocab_size=300)
        assert tokenizer.vocab_size <= 300

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_train_with_custom_pattern(self):
        """Tokenizer accepts custom pattern."""
        tokenizer = Tokenizer()
        texts = ["hello world"]
        # Should not raise
        tokenizer.train(texts, vocab_size=256, pattern=r"\w+|\s+")

    @pytest.mark.skipif(HAS_RUSTBPE, reason="rustbpe is installed")
    def test_train_without_rustbpe_raises(self):
        """Training without rustbpe raises RustbpeNotInstalledError."""
        tokenizer = Tokenizer()
        with pytest.raises(RustbpeNotInstalledError):
            tokenizer.train(["hello"], vocab_size=256)


class TestTokenizerEncode:
    """Tests for encoding."""

    def test_encode_without_training_raises(self):
        """Encoding without training raises TokenizerNotTrainedError."""
        tokenizer = Tokenizer()
        with pytest.raises(TokenizerNotTrainedError):
            tokenizer.encode("hello")

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_encode_returns_list_of_ints(self, trained_tokenizer):
        """encode returns a list of integers."""
        ids = trained_tokenizer.encode("Hello, world!")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_encode_empty_string(self, trained_tokenizer):
        """Encoding empty string returns empty list."""
        ids = trained_tokenizer.encode("")
        assert ids == []

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_encode_word_aligned_returns_groups(self, trained_tokenizer):
        """encode with word_aligned=True returns list of lists."""
        ids = trained_tokenizer.encode("Hello, world!", word_aligned=True)
        assert isinstance(ids, list)
        assert all(isinstance(g, list) for g in ids)

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_encode_word_aligned_groups_non_empty_tokens(self, trained_tokenizer):
        """word_aligned groups contain only non-whitespace tokens."""
        ids = trained_tokenizer.encode("Hello world", word_aligned=True)
        # Should have 2 groups (Hello and world)
        assert len(ids) == 2

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_encode_word_aligned_instance_setting(self, trained_tokenizer):
        """word_aligned can be set on instance and affects encode."""
        trained_tokenizer.word_aligned = True
        ids = trained_tokenizer.encode("Hello world")
        assert isinstance(ids, list)
        assert all(isinstance(g, list) for g in ids)

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_encode_word_aligned_override(self, trained_tokenizer):
        """word_aligned parameter overrides instance setting."""
        trained_tokenizer.word_aligned = True
        # Override to False
        ids = trained_tokenizer.encode("Hello world", word_aligned=False)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)


class TestTokenizerDecode:
    """Tests for decoding."""

    def test_decode_without_training_raises(self):
        """Decoding without training raises TokenizerNotTrainedError."""
        tokenizer = Tokenizer()
        with pytest.raises(TokenizerNotTrainedError):
            tokenizer.decode([1, 2, 3])

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_decode_returns_string(self, trained_tokenizer):
        """decode returns a string."""
        ids = trained_tokenizer.encode("Hello, world!")
        result = trained_tokenizer.decode(ids)
        assert isinstance(result, str)

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_decode_empty_list(self, trained_tokenizer):
        """Decoding empty list returns empty string."""
        result = trained_tokenizer.decode([])
        assert result == ""

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_decode_grouped_format(self, trained_tokenizer):
        """decode handles grouped format from word_aligned encode."""
        ids = trained_tokenizer.encode("Hello, world!", word_aligned=True)
        result = trained_tokenizer.decode(ids)
        assert isinstance(result, str)

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_encode_decode_roundtrip(self, trained_tokenizer):
        """encode/decode roundtrip preserves text."""
        original = "Hello, world!"
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == original

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_encode_decode_roundtrip_word_aligned_loses_spaces(self, trained_tokenizer):
        """encode/decode with word_aligned loses whitespace (expected behavior)."""
        original = "Hello, world!"
        ids = trained_tokenizer.encode(original, word_aligned=True)
        decoded = trained_tokenizer.decode(ids)
        # Spaces are stripped in word_aligned mode
        assert decoded == "Hello,world!"


class TestTokenizerSaveLoad:
    """Tests for save/load operations."""

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_save_to_directory(self, trained_tokenizer, temp_dir):
        """save_to_directory creates files."""
        trained_tokenizer.save_to_directory(temp_dir, name="test")

        assert (temp_dir / "test.json").exists()
        assert (temp_dir / "test.bin").exists()

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_save_creates_metadata(self, trained_tokenizer, temp_dir):
        """save creates valid JSON metadata."""
        trained_tokenizer.save_to_directory(temp_dir, name="test")

        meta = json.loads((temp_dir / "test.json").read_text())
        assert "pattern" in meta
        assert "vocab_size" in meta

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_save_without_training_raises(self, temp_dir):
        """Saving without training raises error."""
        tokenizer = Tokenizer()
        with pytest.raises(TokenizerNotTrainedError):
            tokenizer.save_to_directory(temp_dir)

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    @pytest.mark.skipif(not HAS_TIKTOKEN, reason="tiktoken not installed")
    def test_load_from_directory(self, trained_tokenizer, temp_dir):
        """from_directory loads saved tokenizer."""
        trained_tokenizer.save_to_directory(temp_dir, name="test")

        loaded = Tokenizer.from_directory(temp_dir, name="test")
        assert loaded.vocab_size == trained_tokenizer.vocab_size

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    @pytest.mark.skipif(not HAS_TIKTOKEN, reason="tiktoken not installed")
    def test_load_preserves_encoding(self, trained_tokenizer, temp_dir):
        """Loaded tokenizer produces same encodings."""
        original_text = "Hello, world!"
        original_ids = trained_tokenizer.encode(original_text)

        trained_tokenizer.save_to_directory(temp_dir, name="test")
        loaded = Tokenizer.from_directory(temp_dir, name="test")
        loaded_ids = loaded.encode(original_text)

        assert loaded_ids == original_ids

    @pytest.mark.skipif(not HAS_TIKTOKEN, reason="tiktoken not installed")
    def test_load_missing_files_raises(self, temp_dir):
        """Loading from missing files raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            Tokenizer.from_directory(temp_dir, name="nonexistent")

    @pytest.mark.skipif(HAS_TIKTOKEN, reason="tiktoken is installed")
    def test_load_without_tiktoken_raises(self, temp_dir):
        """Loading without tiktoken raises TiktokenNotInstalledError."""
        # Create dummy files
        (temp_dir / "test.json").write_text('{"pattern": "\\w+", "vocab_size": 100}')
        (temp_dir / "test.bin").write_text("[]")

        with pytest.raises(TiktokenNotInstalledError):
            Tokenizer.from_directory(temp_dir, name="test")


class TestTokenizerVocabSize:
    """Tests for vocab_size property."""

    def test_vocab_size_without_training_raises(self):
        """vocab_size without training raises error."""
        tokenizer = Tokenizer()
        with pytest.raises(TokenizerNotTrainedError):
            _ = tokenizer.vocab_size

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_vocab_size_returns_int(self, trained_tokenizer):
        """vocab_size returns an integer."""
        size = trained_tokenizer.vocab_size
        assert isinstance(size, int)

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_vocab_size_positive(self, trained_tokenizer):
        """vocab_size is positive after training."""
        assert trained_tokenizer.vocab_size > 0


class TestTokenizerBatchEncode:
    """Tests for batch encoding."""

    def test_batch_encode_without_training_raises(self):
        """Batch encoding without training raises error."""
        tokenizer = Tokenizer()
        with pytest.raises(TokenizerNotTrainedError):
            tokenizer.batch_encode(["hello", "world"])

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_batch_encode_returns_list_of_lists(self, trained_tokenizer):
        """batch_encode returns list of lists."""
        texts = ["Hello", "World"]
        result = trained_tokenizer.batch_encode(texts)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(ids, list) for ids in result)

    @pytest.mark.skipif(not HAS_RUSTBPE, reason="rustbpe not installed")
    def test_batch_encode_empty_list(self, trained_tokenizer):
        """Batch encoding empty list returns empty list."""
        result = trained_tokenizer.batch_encode([])
        assert result == []
