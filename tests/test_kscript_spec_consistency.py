"""Spec-consistency checks for specs/kscript.md (KB-273).

Guards the authoritative KScript spec against regressions of the NLP-only
changes: removal of the Mod32 fallback, the 'unbound' named state, and the
MCS -> MTS rename.
Run: python -m pytest tests/test_kscript_spec_consistency.py -v
"""
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1] / "specs" / "kscript.md"


def _spec() -> str:
    return SPEC.read_text()


def test_no_mod_references():
    text = _spec()
    assert "Mod32" not in text
    assert "Mod64" not in text


def test_no_unbound_term():
    assert "unbound" not in _spec().lower()


def test_no_mixed_nlp_regime():
    text = _spec()
    assert "mixed NLP" not in text
    assert "mixed NLP/Mod" not in text


def test_mod32_fallback_section_removed():
    text = _spec()
    assert "Mod32 Fallback" not in text
    assert "encode_unbound" not in text
    assert "bit 0 clear" not in text
    assert "67108864" not in text


def test_no_mcs_term():
    text = _spec()
    assert "MCS" not in text
    assert "Multi-Character Signature" not in text


def test_mts_renamed():
    text = _spec()
    assert "MTS" in text
    assert "Multi-Token Signature" in text
    assert "## 8. MTS (Multi-Token Signature) Expansion" in text
    assert "### 8.3 MTS Deduplication" in text


def test_mod32_section_numbering_has_no_gap():
    # After deleting old §11.3, §11 headings should run 11.1..11.5 with no skip.
    import re

    headings = re.findall(r"^### 11\.(\d+)", _spec(), flags=re.MULTILINE)
    assert headings == ["1", "2", "3", "4", "5"]


def test_test_matrix_reworded_and_ids_stable():
    text = _spec()
    rows = {
        ln.split("|")[1].strip(): ln
        for ln in text.splitlines()
        if ln.strip().startswith("| KS-")
    }
    for kid in ("KS-30", "KS-32", "KS-37"):
        assert kid in rows, f"{kid} row missing — IDs must be stable"
        assert "Mod32" not in rows[kid]
        assert "Unbound" not in rows[kid]
    assert "Mixed NLP/Mod" not in rows["KS-37"]
