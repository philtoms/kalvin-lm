"""Spec-rename regression guard for KB-291: MCS -> MTS in specs/kscript.md.

Locks in the terminological rename so the authoritative KScript spec cannot
silently regress to the obsolete 'MCS' (Multi-Character Signature) term.
Run: python -m pytest tests/test_kscript_spec_mts_rename.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1] / "specs" / "kscript.md"


def _spec() -> str:
    return SPEC.read_text()


def test_no_mcs_term_anywhere():
    """No standalone 'MCS' token or 'Multi-Character Signature' phrase remains."""
    text = _spec()
    assert not re.search(r"\bMCS\b", text), "residual 'MCS' token in kscript.md"
    assert "Multi-Character Signature" not in text


def test_mts_term_present():
    text = _spec()
    assert re.search(r"\bMTS\b", text)
    assert "Multi-Token Signature" in text


def test_section_8_heading_renamed():
    text = _spec()
    assert "## 8. MTS (Multi-Token Signature) Expansion" in text


def test_section_8_3_heading_renamed():
    assert "### 8.3 MTS Deduplication" in _spec()


def test_section_14_6_heading_renamed():
    assert "### 14.6 MTS Expansion" in _spec()


def test_test_matrix_category_column_renamed():
    """The Test Matrix group headers and per-row Category cells use MTS, not MCS."""
    text = _spec()
    assert "| **MTS** |" in text
    assert "| **MTS Deduplication** |" in text
    assert "MCS Dedup" not in text  # old Category cell value


def test_multi_character_adjective_preserved():
    """The descriptive 'multi-character' / 'multi-char' adjective must survive."""
    text = _spec()
    assert "multi-character" in text.lower()
