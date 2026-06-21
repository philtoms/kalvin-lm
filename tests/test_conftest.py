"""Meta-tests for the shared test infrastructure in ``tests/conftest.py``.

These guard the tokenizer-data skip mechanism itself: the probe must return a
plain ``bool`` and ``requires_tokenizer_data`` must be a well-formed pytest
skipif mark. They run with **no** external data dependency, so they pass on a
fresh clone.
"""

from __future__ import annotations

import pytest

from tests import conftest


def test_tokenizer_data_available_returns_bool() -> None:
    """``_tokenizer_data_available()`` must return a real ``bool``."""
    result = conftest._tokenizer_data_available()
    assert isinstance(result, bool)


def test_requires_tokenizer_data_is_skipif_mark() -> None:
    """``requires_tokenizer_data`` must be a ``skipif`` MarkDecorator."""
    assert isinstance(conftest.requires_tokenizer_data, pytest.MarkDecorator)
    mark = conftest.requires_tokenizer_data.mark
    assert mark.name == "skipif"


def test_requires_tokenizer_data_reason_mentions_rebuild_script() -> None:
    """The skip reason should tell the contributor how to generate the assets."""
    reason = conftest.requires_tokenizer_data.mark.kwargs.get("reason", "")
    assert "rebuild-tokenizer-data.sh" in reason


@pytest.mark.parametrize("expected", [bool])
def test_tokenizer_data_available_is_deterministic(expected) -> None:
    """Two consecutive probes must agree (no flaky side effects)."""
    first = conftest._tokenizer_data_available()
    second = conftest._tokenizer_data_available()
    assert isinstance(first, expected)
    assert first == second
