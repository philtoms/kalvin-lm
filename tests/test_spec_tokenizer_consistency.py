"""Spec-consistency regression guard for ``specs/tokenizer.md``.

Locks in the KB-272 change:

* The **Mod** tokenizer (Mod32 / Mod64) is removed entirely.
* **NLP** is the sole production tokenizer, built on a **BPE** subword base.
* The *tokenizers encode dimensionality, not knowledge* thesis is backfilled.

These assertions prevent the Mod tokenizer, the Mod-vs-BPE framing, and the
fictional literal mechanism (``is_literal`` / literal mask / bit-0 literal
flag — never implemented, confirmed by KB-271) from silently returning.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: Path to the tokenizer spec, resolved from the repository root.
SPEC = Path(__file__).resolve().parent.parent / "specs" / "tokenizer.md"


@pytest.fixture(scope="module")
def text() -> str:
    """The full text of ``specs/tokenizer.md``."""
    assert SPEC.exists(), f"spec not found at {SPEC}"
    return SPEC.read_text()


def test_spec_is_non_empty(text: str) -> None:
    assert text.strip() != ""


# --- Mod tokenizer fully removed -------------------------------------------


def test_no_mod_tokenizer_section(text: str) -> None:
    assert "## Mod Tokenizer" not in text


def test_no_mod_variants(text: str) -> None:
    assert "Mod32" not in text
    assert "Mod64" not in text


def test_no_bare_mod_token(text: str) -> None:
    # Word boundaries mean "model" / "modular" cannot match.
    assert not re.search(r"\bMod\b", text)


# --- Term rename: MCS -> MTS ----------------------------------------------


def test_no_mcs_term(text: str) -> None:
    assert "MCS" not in text
    assert "Multi-Character Signature" not in text


# --- Fictional literal mechanism absent ------------------------------------


def test_no_literal_mechanism_phrases(text: str) -> None:
    assert "is_literal" not in text
    assert "literal mask" not in text
    assert "literal-content" not in text


def test_no_literal_node_value_pattern(text: str) -> None:
    # The literal node *value* form (pipe OR of 0xFFFFFFFF) is banned.
    # NOTE: ``node & 0xFFFFFFFF`` in the NLP ``### Decoding`` section is a
    # legitimate BPE-token-ID extraction mask that must stay — that uses
    # ``&``, not ``|``, so it does not match this pattern.
    assert not re.search(r"\|\s*0xFFFFFFFF", text)
    assert "(char_codepoint << 32)" not in text


# --- Overview reframed: BPE + NLP (NLP = production default) ---------------


def test_overview_no_three_types(text: str) -> None:
    assert "Three tokenizer types" not in text


def test_overview_names_bpe_and_nlp(text: str) -> None:
    assert "BPE" in text
    assert "NLP" in text


def test_nlp_is_production_default(text: str) -> None:
    assert "production default" in text


# --- Significance thesis backfilled ----------------------------------------


def test_thesis_dimensionality(text: str) -> None:
    assert "dimensionality" in text


def test_thesis_bitwise_algebra(text: str) -> None:
    assert "bitwise OR" in text
    assert "bitwise AND" in text


def test_nlp_node_format_conformance_line(text: str) -> None:
    assert "(nlp_type32 << 32) | bpe_token_id" in text
