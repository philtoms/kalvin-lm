"""Spec-consistency regression guard for ``specs/signature.md``.

Locks in the KB-281 change:

* The Mod-flavored word **packed** is removed from the Test Matrix
  (SIG-14 row) and from the entire spec.
* The ``is_nlp_node`` note reads consistently with NLP as the sole
  tokenizer — it distinguishes NLP-BPE nodes from non-tokenized values,
  not from "other tokenizer types".

These assertions prevent residual Mod vocabulary (``packed``, ``Mod32``,
``Mod64``, bare ``Mod``) and the renamed MCS term (KB-271) from silently
returning to the signature spec.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: Path to the signature spec, resolved from the repository root.
SPEC = Path(__file__).resolve().parent.parent / "specs" / "signature.md"


@pytest.fixture(scope="class")
def text() -> str:
    """The full text of ``specs/signature.md`` (read once per class)."""
    assert SPEC.exists(), f"spec not found at {SPEC}"
    return SPEC.read_text()


class TestSignatureSpecConsistency:
    """Lock in the Mod-free state of ``specs/signature.md`` (KB-281)."""

    def test_spec_is_non_empty(self, text: str) -> None:
        assert text.strip() != ""

    # --- Mod tokenizer vocabulary fully removed ----------------------------

    def test_no_mod_variants(self, text: str) -> None:
        assert "Mod32" not in text
        assert "Mod64" not in text

    def test_no_bare_mod_token(self, text: str) -> None:
        # Word boundaries mean "model" / "modification" cannot match.
        assert not re.search(r"\bMod\b", text)

    def test_no_packed(self, text: str) -> None:
        assert "packed" not in text.lower()

    # --- Term rename: MCS -> MTS (KB-271) ----------------------------------

    def test_no_mcs_term(self, text: str) -> None:
        assert "MCS" not in text
        assert "Multi-Character Signature" not in text

    # --- SIG-14 OR-reduce property preserved -------------------------------

    def test_sig14_or_reduce_criterion(self, text: str) -> None:
        # The criterion is preserved; only the parenthetical was reworded.
        assert "make_signature([0b10, 0b100]) == 0b110" in text

    # --- is_nlp_node invariants --------------------------------------------

    def test_is_nlp_node_not_used_by_make_signature(self, text: str) -> None:
        # The "Not used by make_signature" clause must survive the rewording.
        subsection = self._tokenizer_subsection(text)
        assert "Not used by" in subsection
        assert "make_signature" in subsection

    def test_is_nlp_node_definition_preserved(self, text: str) -> None:
        subsection = self._tokenizer_subsection(text)
        assert "is_nlp_node(node) → bool" in subsection
        assert "non-zero high 32 bits" in subsection

    def test_tokenizer_subsection_has_no_backward_compat(self, text: str) -> None:
        subsection = self._tokenizer_subsection(text)
        assert "backward compat" not in subsection.lower()

    def test_tokenizer_subsection_has_no_packed(self, text: str) -> None:
        subsection = self._tokenizer_subsection(text)
        assert "packed" not in subsection.lower()

    def test_tokenizer_subsection_has_no_other_tokenizer_types(
        self, text: str
    ) -> None:
        # The note must not frame is_nlp_node as distinguishing from
        # "other tokenizer types" — NLP is the sole tokenizer.
        subsection = self._tokenizer_subsection(text)
        assert "other tokenizer" not in subsection.lower()
        assert "multiple tokenizer" not in subsection.lower()

    @staticmethod
    def _tokenizer_subsection(text: str) -> str:
        """Extract the Dependencies §Tokenizer subsection text.

        Spans from the ``### Tokenizer`` heading up to the next
        level-2 (``##``) heading.
        """
        start = text.find("### Tokenizer")
        assert start != -1, "Tokenizer subsection heading not found"
        end = text.find("\n## ", start)
        if end == -1:
            end = len(text)
        return text[start:end]
