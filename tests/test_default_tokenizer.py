"""Tests for the no-fallback behaviour of ``_default_tokenizer()``.

The kalvin tokenizer is the sole production tokenizer.  When
``Tokenizer.from_files()`` fails — because the data directory is absent, the
BPE backend (``tiktoken``/``rustbpe``) is missing, or a data file is
unreadable — the factory must raise a descriptive :class:`RuntimeError`
naming ``scripts/rebuild-tokenizer-data.sh``, chaining the original cause.

These tests simulate every failure mode with ``unittest.mock.patch`` so they
run regardless of whether the tokenizer data assets are present on the host.
The single positive-path test is gated by :data:`requires_tokenizer_data`.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kalvin.agent import _default_tokenizer
from kalvin.tokenizer import TiktokenNotInstalledError, Tokenizer
from tests.conftest import requires_tokenizer_data

_TARGET = "kalvin.agent.Tokenizer.from_files"


def test_raises_runtime_error_on_file_not_found():
    """A missing data directory surfaces as a RuntimeError, not a bare traceback."""
    with patch(_TARGET, side_effect=FileNotFoundError("no such dir")):
        with pytest.raises(RuntimeError, match="rebuild-tokenizer-data"):
            _default_tokenizer()


def test_raises_runtime_error_on_import_error():
    """A failed optional-backend import surfaces as a RuntimeError."""
    with patch(_TARGET, side_effect=ImportError("no module")):
        with pytest.raises(RuntimeError):
            _default_tokenizer()


def test_raises_runtime_error_on_missing_backend():
    """A missing tiktoken backend surfaces as a RuntimeError.

    ``tiktoken`` import failures are swallowed at module load in
    ``kalvin.tokenizer`` and re-raised as the custom
    :class:`TiktokenNotInstalledError` inside ``Tokenizer.from_directory()``.
    This is a plain ``Exception`` subclass (not ``ImportError``/``OSError``),
    so it must be listed explicitly in the ``except`` tuple.
    """
    with patch(_TARGET, side_effect=TiktokenNotInstalledError("tiktoken gone")):
        with pytest.raises(RuntimeError, match="rebuild-tokenizer-data"):
            _default_tokenizer()


def test_error_message_names_rebuild_script():
    """The error message tells the user exactly how to regenerate the data."""
    with patch(_TARGET, side_effect=FileNotFoundError("missing")):
        with pytest.raises(RuntimeError) as excinfo:
            _default_tokenizer()
    assert "scripts/rebuild-tokenizer-data.sh" in str(excinfo.value)


def test_chains_original_exception():
    """The original exception is chained as ``__cause__`` for debugging."""
    original = FileNotFoundError("missing")
    with patch(_TARGET, side_effect=original):
        with pytest.raises(RuntimeError) as excinfo:
            _default_tokenizer()
    assert excinfo.value.__cause__ is original
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_does_not_swallow_unexpected_error():
    """Unexpected errors (e.g. corrupt data) propagate uncaught.

    The narrowed ``except`` must not catch every ``Exception``; otherwise
    genuine bugs would be masked behind a misleading "data unavailable"
    RuntimeError.
    """
    with patch(_TARGET, side_effect=ValueError("corrupt data")):
        with pytest.raises(ValueError):
            _default_tokenizer()


@requires_tokenizer_data
def test_returns_tokenizer_when_available():
    """When the data is present, the factory returns a Tokenizer."""
    tok = _default_tokenizer()
    assert isinstance(tok, Tokenizer)
