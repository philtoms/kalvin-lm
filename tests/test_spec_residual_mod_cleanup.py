"""Spec-consistency checks for residual Mod/MCS cleanup (KB-283).

Guards specs/harness.md, specs/nlp-curriculum-compat.md, and
specs/nlp-first-curriculum-annotations.md against regression of the
NLP-only cleanup: the MCS->MTS abbreviation rename, the
'multi-character signature' -> 'multi-token signature' terminology
alignment, and removal of the stale 'Mod32 compilation' bullet.

Run: python -m pytest tests/test_spec_residual_mod_cleanup.py -v
"""

from pathlib import Path

SPECS = Path(__file__).resolve().parents[1] / "specs"


def _read(name: str) -> str:
    return (SPECS / name).read_text(encoding="utf-8")


class TestHarnessSpec:
    def test_no_mcs_abbreviation(self) -> None:
        assert "MCS" not in _read("harness.md")

    def test_mts_entries_present(self) -> None:
        assert "MTS entries" in _read("harness.md")

    def test_no_mod32(self) -> None:
        assert "Mod32" not in _read("harness.md")


class TestCurriculumCompatSpec:
    def test_no_multi_character_signature_term(self) -> None:
        text = _read("nlp-curriculum-compat.md")
        assert "multi-character signature" not in text.lower()

    def test_multi_token_term_present(self) -> None:
        assert "multi-token signature" in _read(
            "nlp-curriculum-compat.md"
        ).lower()

    def test_sc2_heading_renamed(self) -> None:
        assert "### SC-2: Multi-token signatures decompose correctly" in _read(
            "nlp-curriculum-compat.md"
        )

    def test_no_mcs_abbreviation(self) -> None:
        assert "MCS" not in _read("nlp-curriculum-compat.md")


class TestCurriculumAnnotationsSpec:
    def test_no_multi_character_signature_term(self) -> None:
        text = _read("nlp-first-curriculum-annotations.md")
        assert "multi-character signature" not in text.lower()

    def test_no_mod32(self) -> None:
        assert "Mod32" not in _read("nlp-first-curriculum-annotations.md")

    def test_multi_token_term_present(self) -> None:
        assert "multi-token signature" in _read(
            "nlp-first-curriculum-annotations.md"
        ).lower()

    def test_na2_heading_renamed(self) -> None:
        assert "### NA-2: Block comments on multi-token signatures" in _read(
            "nlp-first-curriculum-annotations.md"
        )

    def test_mod32_compilation_bullet_removed(self) -> None:
        text = _read("nlp-first-curriculum-annotations.md")
        assert "Mod32 mode" not in text
        assert "annotations are inert" not in text
