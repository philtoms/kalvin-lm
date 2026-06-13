"""Shared pytest fixtures and markers for the kalvin test suite.

This module is auto-discovered by pytest.  Fixtures defined here are
available to every test module under ``tests/`` without an explicit import.

The :data:`requires_nlp_data` marker (and the :func:`nlp_tokenizer` /
:func:`nlp` fixtures) exist so that NLP-tokenizer tests can be **cleanly
skipped** on fresh clones where the ``data/`` directory is absent.  The
entire ``data/`` tree is gitignored (binary assets, ~35 MB); the documented
way to regenerate it is ``bash scripts/rebuild-tokenizer-data.sh``.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# NLP data-asset availability probe
# ---------------------------------------------------------------------------
#
# Mirrors the pattern previously inlined in tests/test_ks_compiler.py.  We probe
# *both* the import (optional heavy backends) and the actual data files, so a
# missing-dependency *or* a missing-data situation both resolve to a clean skip
# rather than a collection error or a fixture failure.


def _nlp_data_available() -> bool:
    """Return ``True`` when the NLPTokenizer data assets are loadable.

    Probes :func:`kalvin.nlp_tokenizer.NLPTokenizer.from_files` and returns
    ``False`` on any of ``ImportError`` (optional backend missing),
    ``FileNotFoundError`` (data directory absent) or ``OSError``.
    """
    try:
        from kalvin.nlp_tokenizer import NLPTokenizer

        NLPTokenizer.from_files()
        return True
    except (ImportError, FileNotFoundError, OSError):
        return False


#: Skip marker for tests that require NLP tokenizer data assets.  Apply to a
#: whole module via ``pytestmark = requires_nlp_data`` or to individual items /
#: classes with ``@requires_nlp_data``.
requires_nlp_data = pytest.mark.skipif(
    not _nlp_data_available(),
    reason=(
        "NLPTokenizer data files unavailable "
        "(run scripts/rebuild-tokenizer-data.sh to generate them)"
    ),
)


# ---------------------------------------------------------------------------
# Shared NLP tokenizer fixtures
# ---------------------------------------------------------------------------
#
# These replace the per-module ``nlp`` / ``nlp_tokenizer`` fixtures that were
# copy-pasted across test_agent.py, test_nlp_curriculum_compat.py and
# test_nlp_tokenizer.py.  ``nlp`` delegates to ``nlp_tokenizer`` so a single
# instance is built per module regardless of which name a test requests.


@pytest.fixture(scope="module")
def nlp_tokenizer():
    """Load an :class:`NLPTokenizer` from the standard data files.

    Built once per module.  Tests using this fixture should also be gated by
    :data:`requires_nlp_data` so the fixture is never instantiated when the
    data assets are absent.
    """
    from kalvin.nlp_tokenizer import NLPTokenizer

    return NLPTokenizer.from_files()


@pytest.fixture(scope="module")
def nlp(nlp_tokenizer):
    """Alias of :func:`nlp_tokenizer` exposed under the shorter name ``nlp``.

    Returns the *same* module-scoped instance, so modules that use ``nlp``
    and modules that use ``nlp_tokenizer`` never double-load the tokenizer.
    """
    return nlp_tokenizer
