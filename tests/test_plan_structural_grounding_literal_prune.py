"""Plan-consistency checks for the literal-mechanism prune (KB-292).

Guards plans/impl/structural-grounding.md so the never-implemented literal
mechanism (is_literal_node, "All-literal" kline concept) and the old
pre-renumber MOD test IDs cannot silently regress into the HOW layer.
Run: python -m pytest tests/test_plan_structural_grounding_literal_prune.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plans" / "impl" / "structural-grounding.md"

# The literal mechanism was never implemented; these tokens must not appear.
BANNED_SUBSTR = [
    "is_literal_node",
    "All-literal",
    "old numbering",
]

# Old MOD IDs that the plan used before alignment (must not remain).
# NOTE: MOD-48-51 are deliberately excluded — they are OLD IDs in §2.5
# (generate_expansions -> MOD-56-59) but also NEW IDs in §1.5
# (promote_participating). They must be PRESENT, not absent. Their §2.5
# replacement is guarded by ALIGNED_MOD_IDS checking MOD-56-59 are present.
OLD_MOD_IDS = [
    "MOD-26", "MOD-27", "MOD-28",  # is_s1 -> MOD-34/35/36
    "MOD-40", "MOD-41", "MOD-42", "MOD-43",  # promote_participating -> MOD-48-51
    "MOD-44", "MOD-45", "MOD-46", "MOD-47",  # classify_misfit -> MOD-52-55
]

# Authoritative spec MOD IDs that MUST be present after alignment.
ALIGNED_MOD_IDS = [
    "MOD-34", "MOD-35", "MOD-36",  # is_s1
    "MOD-48", "MOD-49", "MOD-50", "MOD-51",  # promote_participating
    "MOD-52", "MOD-53", "MOD-54", "MOD-55",  # classify_misfit
    "MOD-56", "MOD-57", "MOD-58", "MOD-59",  # generate_expansions
]

# KB-294: KS-10 ("Empty source") is unrelated to the CANONIZE/S2-expansion
# tests in §2.5. The correct authoritative ID is KS-14 (CANONIZE aggregates).
STALE_KS_IDS = ["KS-10"]
ALIGNED_KS_IDS = ["KS-14"]


def _has_id(text: str, identifier: str) -> bool:
    """True if identifier appears as a standalone token (word boundaries),
    so MOD-4 is not confused with MOD-44."""
    return re.search(rf"\b{re.escape(identifier)}\b", text) is not None


def test_plan_file_exists():
    assert PLAN.is_file(), f"missing plan file: {PLAN}"


def test_no_literal_mechanism():
    text = PLAN.read_text()
    for tok in BANNED_SUBSTR:
        assert tok not in text, f"residual '{tok}' in structural-grounding.md"


def test_no_standalone_literal_word():
    text = PLAN.read_text().lower()
    assert "literal" not in text


def test_old_mod_ids_absent():
    """Old pre-renumber MOD IDs must not remain in the plan."""
    text = PLAN.read_text()
    for mid in OLD_MOD_IDS:
        assert not _has_id(text, mid), f"residual old ID '{mid}' in structural-grounding.md"


def test_aligned_mod_ids_present():
    """Authoritative spec MOD IDs must be present after alignment."""
    text = PLAN.read_text()
    for mid in ALIGNED_MOD_IDS:
        assert _has_id(text, mid), f"aligned ID '{mid}' missing from structural-grounding.md"


def test_no_stale_ks10_references():
    """KB-294: KS-10 ('Empty source') is a stale reference — the §2.5
    KScript rows use CANONIZE (KS-14), not the empty-source parser test."""
    text = PLAN.read_text()
    for ks in STALE_KS_IDS:
        assert not _has_id(text, ks), f"residual stale ID '{ks}' in structural-grounding.md"


def test_aligned_ks14_present():
    """KB-294: KS-14 (CANONIZE aggregates) is the authoritative kscript
    spec ID for the CANONIZE-based S2-expansion tests in §2.5."""
    text = PLAN.read_text()
    for ks in ALIGNED_KS_IDS:
        assert _has_id(text, ks), f"aligned KS ID '{ks}' missing from structural-grounding.md"


def test_surviving_agt_ids_present():
    """AGT IDs that already match the spec must still be present."""
    text = PLAN.read_text()
    for aid in ("AGT-29", "AGT-34", "AGT-36", "AGT-37"):
        assert _has_id(text, aid), f"surviving ID '{aid}' missing from structural-grounding.md"


def test_both_test_mapping_tables_have_separator_rows():
    """Guard against table corruption during edits — both §1.5 and §2.5
    must retain their markdown table separator rows."""
    text = PLAN.read_text()
    sep_count = text.count("| ------- |")
    assert sep_count >= 2, f"expected >=2 table separators, found {sep_count}"
