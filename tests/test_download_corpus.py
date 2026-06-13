"""Unit tests for ``dev.nlp.download_corpus``.

All tests mock ``datasets.load_dataset`` so no real network calls are made.
The retry, revision-pinning, non-streaming flag, and output-format logic is
exercised in isolation.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from dev.nlp.download_corpus import download_simplestories

#: Patch target for the module-level ``load_dataset`` name.
LOAD_DATASET = "dev.nlp.download_corpus.load_dataset"


def _make_rows(n: int) -> list[dict[str, str]]:
    """Return *n* fake dataset rows in the SimpleStories ``{story: ...}`` shape.

    A plain ``list`` doubles for the non-streaming ``Dataset`` returned by
    ``load_dataset(..., streaming=False)``: it supports both ``len()`` and
    iteration, which is all the truncation/iteration logic in
    ``download_simplestories`` requires.
    """
    return [{"story": f"Story number {i}."} for i in range(n)]


def test_retry_succeeds_after_failures(tmp_path):
    """Two transient failures then success should still write the corpus."""
    output = tmp_path / "corpus.json"
    rows = _make_rows(5)

    with patch(LOAD_DATASET) as mock_load:
        mock_load.side_effect = [
            ConnectionError("transient 1"),
            ConnectionError("transient 2"),
            rows,
        ]
        download_simplestories(
            output_path=str(output),
            num_samples=5,
            revision="abc123",
            base_delay=0,  # no real sleeping in tests
        )

    assert mock_load.call_count == 3
    data = json.loads(output.read_text())
    assert len(data) == 5
    assert data[0] == {"summary": "Story number 0."}


def test_retry_exhausted_raises(tmp_path):
    """When every attempt fails the last exception propagates."""
    output = tmp_path / "corpus.json"

    with patch(LOAD_DATASET) as mock_load:
        mock_load.side_effect = ConnectionError("permanent failure")
        with pytest.raises(ConnectionError, match="permanent failure"):
            download_simplestories(
                output_path=str(output),
                num_samples=3,
                revision="abc123",
                max_retries=3,
                base_delay=0,
            )

    # Exactly max_retries attempts were made.
    assert mock_load.call_count == 3
    # The output file was never written.
    assert not output.exists()


def test_revision_passed_to_load_dataset(tmp_path):
    """The ``revision`` kwarg is forwarded to ``load_dataset``."""
    output = tmp_path / "corpus.json"

    with patch(LOAD_DATASET) as mock_load:
        mock_load.return_value = _make_rows(2)
        download_simplestories(
            output_path=str(output),
            num_samples=2,
            revision="deadbeef",
            base_delay=0,
        )

    mock_load.assert_called_once()
    assert mock_load.call_args.kwargs["revision"] == "deadbeef"


def test_streaming_is_false(tmp_path):
    """``load_dataset`` is called with ``streaming=False``.

    This documents the intentional KB-218 switch from streaming to non-streaming
    mode so the full dataset is cached locally as arrow files (zero network on
    a warm cache, no HTTP 429 rate-limit exposure).
    """
    output = tmp_path / "corpus.json"

    with patch(LOAD_DATASET) as mock_load:
        mock_load.return_value = _make_rows(2)
        download_simplestories(
            output_path=str(output),
            num_samples=2,
            revision="abc123",
            base_delay=0,
        )

    mock_load.assert_called_once()
    assert mock_load.call_args.kwargs["streaming"] is False


def test_output_format(tmp_path):
    """Output is a JSON list of ``{"summary": ...}`` dicts."""
    output = tmp_path / "corpus.json"

    with patch(LOAD_DATASET) as mock_load:
        mock_load.return_value = _make_rows(3)
        download_simplestories(
            output_path=str(output),
            num_samples=20000,
            revision="abc123",
            base_delay=0,
        )

    data = json.loads(output.read_text())
    assert isinstance(data, list)
    assert len(data) == 3
    for entry in data:
        assert set(entry.keys()) == {"summary"}
        assert isinstance(entry["summary"], str)


def test_truncates_to_num_samples(tmp_path):
    """The corpus is truncated to exactly ``num_samples`` rows.

    The full SimpleStories dataset has ~2.1M rows; only ``num_samples`` should
    be written. The plain-list mock (50 rows) verifies the ``islice``
    truncation path.
    """
    output = tmp_path / "corpus.json"

    with patch(LOAD_DATASET) as mock_load:
        mock_load.return_value = _make_rows(50)
        download_simplestories(
            output_path=str(output),
            num_samples=10,
            revision="abc123",
            base_delay=0,
        )

    data = json.loads(output.read_text())
    assert len(data) == 10
    # First 10 rows of the 50-row mock.
    assert data[0] == {"summary": "Story number 0."}
    assert data[-1] == {"summary": "Story number 9."}


def test_default_revision_is_none(tmp_path):
    """With ``revision=None`` the kwarg is omitted (backward-compatible)."""
    output = tmp_path / "corpus.json"

    with patch(LOAD_DATASET) as mock_load:
        mock_load.return_value = _make_rows(1)
        download_simplestories(
            output_path=str(output),
            num_samples=1,
            revision=None,
            base_delay=0,
        )

    mock_load.assert_called_once()
    assert "revision" not in mock_load.call_args.kwargs
