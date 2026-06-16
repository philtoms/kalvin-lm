"""Content-assertion test pinning the KB-275 docs-layer cleanup.

Locks in two DOCS (WHY) layer changes:

* ``docs/tokenizer-significance.md`` is **deleted** — it was a Mod-vs-BPE
  comparison plus the fictional literal mechanism (``is_literal`` / literal
  mask / bit-0 literal flag, never implemented). Its surviving *tokenizers
  encode dimensionality, not knowledge* thesis was backfilled into
  ``specs/tokenizer.md`` by KB-272, so nothing is lost.
* ``docs/kalvin-origin.md`` — the project's authoritative origin document —
  no longer targets the Mod tokenizer. KScript now targets the **NLP
  (BPE-based)** tokenizer, and the Component Map lists the Tokenizer as
  ``(NLP / BPE)``.

These assertions prevent the deleted file from silently returning and the
Mod-targeting from creeping back into the authoritative origin doc.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: Repository root, resolved from this test file.
REPO = Path(__file__).resolve().parent.parent

#: Path to the deleted doc (must NOT exist).
SIGNIFICANCE_DOC = REPO / "docs" / "tokenizer-significance.md"

#: Path to the authoritative origin doc (must exist).
ORIGIN_DOC = REPO / "docs" / "kalvin-origin.md"

#: Pattern matching Mod-targeting references that must not return. The word
#: boundary on ``(Mod`` avoids false positives like "model" / "modular".
MOD_TARGETING = re.compile(r"Mod32|Mod64|ModTokenizer|\(Mod\b|Mod / BPE|single-character")


# --- Deleted doc stays gone ------------------------------------------------


def test_significance_doc_is_deleted() -> None:
    assert not SIGNIFICANCE_DOC.exists(), (
        f"{SIGNIFICANCE_DOC} should have been deleted by KB-275"
    )


# --- Authoritative origin doc cleanup --------------------------------------


@pytest.fixture(scope="module")
def origin_text() -> str:
    """The full text of ``docs/kalvin-origin.md``."""
    assert ORIGIN_DOC.exists(), f"origin doc not found at {ORIGIN_DOC}"
    return ORIGIN_DOC.read_text()


def test_origin_doc_has_no_mod_targeting(origin_text: str) -> None:
    match = MOD_TARGETING.search(origin_text)
    assert match is None, (
        "kalvin-origin.md must not contain Mod-targeting references; "
        f"found {match.group(0)!r}"
    )


def test_origin_doc_component_map_uses_nlp_bpe(origin_text: str) -> None:
    assert "(NLP / BPE)" in origin_text


def test_origin_doc_references_nlp_and_bpe(origin_text: str) -> None:
    assert "NLP" in origin_text
    assert "BPE" in origin_text
