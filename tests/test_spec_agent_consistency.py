"""Spec-consistency regression guard for ``specs/agent.md``.

Locks in the NLP-only tokenizer default established by KB-274 (commit
``0109be4``) under the wider NLP-only decision (KB-271 / KB-272):

* The Agent's default tokenizer is ``NLPTokenizer`` — never ``Mod32``.
* NLP data is mandatory; construction raises if unavailable.
* The stale Mod32 default cannot silently return to the spec.

These assertions follow the established pattern from KB-272
(``test_spec_tokenizer_consistency.py``) and KB-281
(``test_spec_signature_consistency.py``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: Path to the agent spec, resolved from the repository root.
SPEC = Path(__file__).resolve().parent.parent / "specs" / "agent.md"


@pytest.fixture(scope="class")
def text() -> str:
    """The full text of ``specs/agent.md`` (read once per class)."""
    assert SPEC.exists(), f"spec not found at {SPEC}"
    return SPEC.read_text()


class TestAgentSpecConsistency:
    """Lock in the NLP-only tokenizer default in ``specs/agent.md`` (KB-282)."""

    def test_spec_is_non_empty(self, text: str) -> None:
        assert text.strip() != ""

    # --- Mod tokenizer vocabulary fully removed ----------------------------

    def test_no_mod_variants(self, text: str) -> None:
        assert "Mod32" not in text
        assert "Mod64" not in text

    def test_no_bare_mod_token(self, text: str) -> None:
        # Word boundaries mean "Model" / "model" / "modular" cannot match.
        assert not re.search(r"\bMod\b", text)

    def test_no_mod_tokenizer_reference(self, text: str) -> None:
        assert "mod_tokenizer" not in text.lower()

    def test_no_stale_mod32_default_phrase(self, text: str) -> None:
        # The historical stale default phrasing must not return. Strictly
        # weaker than banning "Mod32" outright, but documents the intent.
        assert "defaults to Mod32" not in text
        assert "Defaults to Mod32" not in text

    # --- Term rename: MCS -> MTS (KB-271) ----------------------------------

    def test_no_mcs_term(self, text: str) -> None:
        assert "MCS" not in text
        assert "Multi-Character Signature" not in text

    # --- Construction section defaults to NLPTokenizer ---------------------

    def test_construction_defaults_to_nlp(self, text: str) -> None:
        construction = self._construction_section(text)
        assert "NLPTokenizer" in construction

    def test_construction_nlp_data_is_mandatory(self, text: str) -> None:
        # Confirms the "NLP data is mandatory" framing introduced by KB-274.
        construction = self._construction_section(text)
        assert "mandatory" in construction

    @staticmethod
    def _construction_section(text: str) -> str:
        """Extract the Construction section text.

        Spans from the ``## Construction`` heading up to the next
        level-2 (``##``) heading.
        """
        marker = "## Construction"
        start = text.find(marker)
        assert start != -1, "Construction heading not found"
        end = text.find("\n## ", start + len(marker))
        if end == -1:
            end = len(text)
        return text[start:end]
