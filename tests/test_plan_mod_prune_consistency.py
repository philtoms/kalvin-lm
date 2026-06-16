"""Plan-consistency checks for the Mod / is_literal prune (KB-276).

Guards the four edited plans/ files so the removed Mod tokenizer, the
never-implemented is_literal / literal mechanism, and the MCS->MTS rename
cannot silently regress.
Run: python -m pytest tests/test_plan_mod_prune_consistency.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KALVIN = ROOT / "plans" / "implement-kalvin.md"
KSCRIPT = ROOT / "plans" / "implement-kscript.md"
FOUNDATIONS = ROOT / "plans" / "impl" / "foundations.md"
BUILD = ROOT / "plans" / "impl" / "build-phases.md"
ALL = [KALVIN, KSCRIPT, FOUNDATIONS, BUILD]

# Tokens that must never reappear in the edited plans. 'Mod32'/'Mod64' are
# case-sensitive substrings; the bare 'Mod' word is checked separately.
BANNED_SUBSTR = [
    "mod_tokenizer",
    "is_literal",
    "LITERAL_MASK",
    "LITERAL_ONLY",
    "MOD32_BITS",
    "MOD64_BITS",
    "Mod32",
    "Mod64",
]


def test_plan_files_exist():
    """Every guarded plan file must exist (guard against path drift)."""
    for p in ALL:
        assert p.is_file(), f"missing plan file: {p}"


def test_no_mod_or_literal_tokens_in_any_edited_plan():
    for p in ALL:
        text = p.read_text()
        for tok in BANNED_SUBSTR:
            assert tok not in text, f"{p.name}: residual '{tok}'"


def test_no_bare_mod_word_in_edited_plans():
    # \bMod\b must not appear as a standalone word (word boundaries exclude "model").
    for p in ALL:
        assert not re.search(r"\bMod\b", p.read_text()), (
            f"{p.name}: residual 'Mod' word"
        )


def test_kscript_plan_mcs_renamed_to_mts():
    text = KSCRIPT.read_text()
    assert "MCS" not in text
    assert "MTS" in text
    assert "_emit_mts" in text  # renamed method skeleton


def test_kscript_plan_no_mod32_fallback_or_mixed_regime():
    text = KSCRIPT.read_text().lower()
    assert "mod32 fallback" not in text
    assert "mixed nlp" not in text
    # 'unbound' named state removed
    assert "unbound" not in text


def test_kscript_plan_section_refs_repointed():
    text = KSCRIPT.read_text()
    # old §11.4 Multi-Token Words is now §11.3 post-KB-273; KS-42 ref must follow
    assert "Canonical encoding (§11.3/§11.4)" in text or "§11.3/§11.4" in text
    # stale references to the now-renumbered old §11.6 (Design Tension -> §11.5) gone
    assert "§11.6" not in text


def test_foundations_plan_no_mod_variants():
    text = FOUNDATIONS.read_text()
    assert "Variants: Mod" not in text
    assert "PACKED NODE (Mod tokenizer)" not in text
    # SIG-14 label no longer cites Mod32
    assert "Mod32 backward compat" not in text
    assert "OR-reduction of two node values" in text


def test_build_phases_no_literal_mechanism():
    text = BUILD.read_text().lower()
    assert "all-literal" not in text
    assert "literal mask" not in text
    assert "(codepoint << 32)" not in text
    assert "0xffffffff" not in text


def test_build_phases_subsections_renumbered_gap_free():
    text = BUILD.read_text()
    headings = re.findall(r"^### 1\.(\d+) ", text, re.MULTILINE)
    nums = [int(n) for n in headings]
    assert nums == list(range(1, len(nums) + 1)), (
        f"§1.x subsections not gap-free: {nums}"
    )


def test_kalvin_plan_is_literal_section_gone():
    text = KALVIN.read_text()
    assert "## 3. `is_literal`" not in text
    # the constants block no longer defines the literal mask
    assert "LITERAL_MASK = 0xFFFF_FFFF" not in text
