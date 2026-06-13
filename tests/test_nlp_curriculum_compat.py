"""Tests for NLP tokenizer compatibility with existing curricula.

Verifies that all curricula compile and rationalize correctly with
the NLPTokenizer without requiring parenthetical comment annotations.

Spec ref: specs/nlp-curriculum-compat.md
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ks import compile_source
from kalvin.agent import KAgent
from kalvin.kline import sig_level
from kalvin.nlp_tokenizer import NLPTokenizer

# ── Fixtures ──────────────────────────────────────────────────────────

CURRICULA_DIR = Path(__file__).resolve().parents[1] / "curricula"


@pytest.fixture(scope="module")
def nlp_tokenizer() -> NLPTokenizer:
    """Load NLPTokenizer from standard data files."""
    return NLPTokenizer.from_files()


def _kscript_from_curriculum(path: Path) -> str:
    """Extract KScript source from a curriculum markdown file."""
    text = path.read_text(encoding="utf-8")
    in_block = False
    blocks: list[str] = []
    for line in text.splitlines():
        if line.strip() == "```":
            in_block = not in_block
            continue
        if in_block:
            blocks.append(line)
    return "\n".join(blocks)


class _CountingAdapter:
    """Adapter that counts frame and ground events."""

    def __init__(self) -> None:
        self.frame = 0
        self.ground = 0

    def on_event(self, event: object) -> None:
        kind = getattr(event, "kind", None)
        if kind == "frame":
            self.frame += 1
        elif kind == "ground":
            self.ground += 1


# ── SC-1: Bare single-char signatures ────────────────────────────────


class TestBareSingleChar:
    """SC-1: Bare single-char sigs produce consistent NLP-BPE tokens."""

    def test_m_compiles_to_nlp_bpe(self, nlp_tokenizer: NLPTokenizer) -> None:
        entries = compile_source("M", tokenizer=nlp_tokenizer, dev=True)
        assert len(entries) == 1
        sig = entries[0].signature
        # Upper 32 bits should have NLP type info (not zero)
        assert (sig >> 32) != 0
        # Lower 32 bits should be the BPE token ID for 'M'
        assert (sig & 0xFFFFFFFF) != 0

    def test_consistent_encoding(self, nlp_tokenizer: NLPTokenizer) -> None:
        """Same character always produces the same node value."""
        e1 = compile_source("M", tokenizer=nlp_tokenizer)
        e2 = compile_source("M", tokenizer=nlp_tokenizer)
        assert e1[0].signature == e2[0].signature

    def test_different_chars_different_tokens(
        self, nlp_tokenizer: NLPTokenizer
    ) -> None:
        e_m = compile_source("M", tokenizer=nlp_tokenizer)
        e_h = compile_source("H", tokenizer=nlp_tokenizer)
        assert e_m[0].signature != e_h[0].signature


# ── SC-2: Multi-character signature decomposition ────────────────────


class TestMultiCharDecomposition:
    """SC-2: Multi-char sigs decompose correctly with NLP tokenizer."""

    def test_mhall_decomposition(self, nlp_tokenizer: NLPTokenizer) -> None:
        entries = compile_source("MHALL", tokenizer=nlp_tokenizer, dev=True)
        # Under the regenerated BPE model, MHALL decomposes into its
        # character-component unsigneds (S4) plus decomposition entries (S2).
        assert len(entries) == 11

        # Component unsigneds are S4; decompositions are S2.
        s4 = [e for e in entries if sig_level(e) == "S4"]
        s2 = [e for e in entries if sig_level(e) == "S2"]
        assert len(s4) == 9
        assert len(s2) == 2

    def test_countersign_with_multi_char(
        self, nlp_tokenizer: NLPTokenizer
    ) -> None:
        entries = compile_source(
            "MHALL == SVO", tokenizer=nlp_tokenizer, dev=True
        )
        # Should have countersign entries for MHALL:SVO and SVO:MHALL
        countersigns = [e for e in entries if sig_level(e) == "S1"]
        assert len(countersigns) == 2


# ── SC-3: Curriculum compilation ─────────────────────────────────────


class TestCurriculumCompilation:
    """SC-3: All curricula compile and rationalize correctly."""

    @pytest.mark.parametrize(
        "curriculum_file",
        [
            "first-steps.md",
            "first-steps-s2.md",
            "mhall-svo-single.md",
            "mhall-svo-equivalence.md",
            "cascade-pressure.md",
            "conflict-drill.md",
            "s3-auto-countersign.md",
        ],
    )
    def test_curriculum_compiles(
        self, curriculum_file: str, nlp_tokenizer: NLPTokenizer
    ) -> None:
        path = CURRICULA_DIR / curriculum_file
        if not path.exists():
            pytest.skip(f"Curriculum not found: {path}")

        kscript = _kscript_from_curriculum(path)
        entries = compile_source(kscript, tokenizer=nlp_tokenizer, dev=True)
        assert len(entries) > 0

    @pytest.mark.parametrize(
        "curriculum_file",
        [
            "first-steps.md",
            "first-steps-s2.md",
            "mhall-svo-single.md",
            "mhall-svo-equivalence.md",
            "cascade-pressure.md",
            "conflict-drill.md",
            "s3-auto-countersign.md",
        ],
    )
    def test_curriculum_rationalizes(
        self, curriculum_file: str, nlp_tokenizer: NLPTokenizer
    ) -> None:
        path = CURRICULA_DIR / curriculum_file
        if not path.exists():
            pytest.skip(f"Curriculum not found: {path}")

        kscript = _kscript_from_curriculum(path)
        entries = compile_source(kscript, tokenizer=nlp_tokenizer, dev=True)
        assert len(entries) > 0

        adapter = _CountingAdapter()
        agent = KAgent(tokenizer=nlp_tokenizer, adapter=adapter)

        # Rationalize all entries without errors
        for entry in entries:
            agent.rationalise(entry)

        # Should have at least some events
        assert adapter.frame + adapter.ground > 0


# ── SC-4: Comments are optional ──────────────────────────────────────


class TestCommentsOptional:
    """SC-4: Bare sigs work; comments add semantic resolution."""

    def test_bare_and_annotated_both_work(
        self, nlp_tokenizer: NLPTokenizer
    ) -> None:
        bare = compile_source("M", tokenizer=nlp_tokenizer, dev=True)
        annotated = compile_source("M(ary)", tokenizer=nlp_tokenizer, dev=True)

        # Both produce entries
        assert len(bare) == 1
        # "ary" BPE-tokenizes into multiple subwords → additional entries
        assert len(annotated) == 4

        # Both signatures are NLP-BPE encoded (high bits set)
        assert (bare[0].signature >> 32) != 0
        assert (annotated[0].signature >> 32) != 0

        # The annotation expands "M" into more entries than the bare sig
        assert len(annotated) > len(bare)

    def test_block_comment_binding(self, nlp_tokenizer: NLPTokenizer) -> None:
        """Block comment before multi-char sig enables positional binding."""
        from ks.compiler import compile_source

        source = "(Mary Had A Little Lamb)\nMHALL"
        entries = compile_source(source, tokenizer=nlp_tokenizer, dev=True)

        # Block comment should bind M→Mary, H→Had, A→A, L→Little, L→Lamb
        # producing NLP-encoded signature entries.  With dev=True the sig field
        # is human-readable.
        assert len(entries) > 0

        # At least the first entry should resolve via binding
        # (signature should be an NLP-BPE token with high bits set)
        assert (entries[0].signature >> 32) != 0
