"""Shared pytest fixtures and markers for the kalvin test suite.

This module is auto-discovered by pytest.  Fixtures defined here are
available to every test module under ``tests/`` without an explicit import.

The :data:`requires_tokenizer_data` marker (and the :func:`tokenizer`
fixture) exist so that tokenizer-dependent tests can be **cleanly skipped**
on fresh clones where the ``data/`` directory is absent.  The entire
``data/`` tree is gitignored (binary assets, ~35 MB); the documented way to
regenerate it is ``bash scripts/rebuild-tokenizer-data.sh``.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Tokenizer data-asset availability probe
# ---------------------------------------------------------------------------
#
# We probe *both* the import (optional heavy backends) and the actual data
# files, so a missing-dependency *or* a missing-data situation both resolve
# to a clean skip rather than a collection error or a fixture failure.


def _tokenizer_data_available() -> bool:
    """Return ``True`` when the kalvin tokenizer data assets are loadable.

    Probes :func:`kalvin.tokenizer.Tokenizer.from_files` and returns
    ``False`` on any of ``ImportError`` (optional backend missing),
    ``FileNotFoundError`` (data directory absent) or ``OSError``.
    """
    try:
        from kalvin.tokenizer import Tokenizer

        Tokenizer.from_files()
        return True
    except (ImportError, FileNotFoundError, OSError):
        return False


#: Skip marker for tests that require tokenizer data assets.  Apply to a
#: whole module via ``pytestmark = requires_tokenizer_data`` or to individual
#: items / classes with ``@requires_tokenizer_data``.
requires_tokenizer_data = pytest.mark.skipif(
    not _tokenizer_data_available(),
    reason=(
        "Tokenizer data files unavailable "
        "(run scripts/rebuild-tokenizer-data.sh to generate them)"
    ),
)


# ---------------------------------------------------------------------------
# Shared tokenizer fixture
# ---------------------------------------------------------------------------
#
# Returns the production kalvin tokenizer (base Tokenizer) loaded from the
# standard data files.  Built once per module.  Tests using this fixture
# should also be gated by :data:`requires_tokenizer_data` so the fixture is
# never instantiated when the data assets are absent.
#
# Tests that specifically exercise the NLP specialisation should construct
# ``NLPTokenizer`` directly (see tests/test_nlp_tokenizer.py).


@pytest.fixture(scope="module")
def tokenizer():
    """Load the production :class:`kalvin.tokenizer.Tokenizer` from data files."""
    from kalvin.tokenizer import Tokenizer

    return Tokenizer.from_files()
