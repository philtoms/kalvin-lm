"""Plan-consistency checks for the literal-mechanism prune (KB-289).

Guards plans/impl/model.md and plans/impl/agent.md so the
never-implemented literal mechanism (is_literal, literal/non-literal
dedup) cannot silently regress into the HOW layer.
Run: python -m pytest tests/test_plan_literal_prune_consistency.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PLAN = ROOT / "plans" / "impl" / "model.md"
AGENT_PLAN = ROOT / "plans" / "impl" / "agent.md"
ALL = [MODEL_PLAN, AGENT_PLAN]

# The literal mechanism was never implemented; these tokens must not appear.
BANNED_SUBSTR = [
    "is_literal",
    "Literal dedup",
    "Non-literal",
    "non-lit",
    "All-literal",
    "Literal KLine",
    "Literal duplicate",
    "Literal nodes cannot match",
]

# Spec IDs removed by KB-274 (they described fictional literal dedup). The
# plans must not retain them. Checked with word boundaries so e.g. MOD-2 is
# not confused with MOD-20/MOD-21.
REMOVED_MOD_IDS = ["MOD-2", "MOD-3", "MOD-16", "MOD-27", "MOD-28", "MOD-29", "MOD-30", "MOD-31"]
REMOVED_AGT_IDS = ["AGT-13"]

# Surviving IDs that must remain present (guard against accidental deletion).
SURVIVING_MOD_IDS = ["MOD-1", "MOD-4", "MOD-23", "MOD-32", "MOD-33", "MOD-34", "MOD-47", "MOD-50"]
SURVIVING_AGT_IDS = ["AGT-12", "AGT-14", "AGT-15"]


def _has_id(text: str, identifier: str) -> bool:
    """True if identifier appears as a standalone token (word boundaries),
    so MOD-2 is not matched by MOD-20."""
    return re.search(rf"\b{re.escape(identifier)}\b", text) is not None


def test_plan_files_exist():
    for p in ALL:
        assert p.is_file(), f"missing plan file: {p}"


def test_no_literal_mechanism_in_either_plan():
    for p in ALL:
        text = p.read_text()
        for tok in BANNED_SUBSTR:
            assert tok not in text, f"{p.name}: residual '{tok}'"


def test_model_plan_no_literal_word():
    # No standalone 'literal' word (case-insensitive) in the model plan.
    text = MODEL_PLAN.read_text().lower()
    assert "literal" not in text


def test_agent_plan_no_literal_word():
    text = AGENT_PLAN.read_text().lower()
    assert "literal" not in text


def test_model_plan_removed_mod_ids_absent():
    """The spec removed these MOD IDs (they described fictional literal
    dedup); the plan must not retain them."""
    text = MODEL_PLAN.read_text()
    for mid in REMOVED_MOD_IDS:
        assert not _has_id(text, mid), f"model.md: residual removed ID '{mid}'"


def test_agent_plan_removed_agt_id_absent():
    """The spec removed AGT-13; the plan must not retain it."""
    text = AGENT_PLAN.read_text()
    for aid in REMOVED_AGT_IDS:
        assert not _has_id(text, aid), f"agent.md: residual removed ID '{aid}'"


def test_model_plan_surviving_ids_present():
    """Surviving MOD IDs must still be present (not accidentally deleted)."""
    text = MODEL_PLAN.read_text()
    for mid in SURVIVING_MOD_IDS:
        assert _has_id(text, mid), f"model.md: surviving ID '{mid}' missing"


def test_agent_plan_surviving_ids_present():
    """Surviving AGT IDs around the deletion point must still be present."""
    text = AGENT_PLAN.read_text()
    for aid in SURVIVING_AGT_IDS:
        assert _has_id(text, aid), f"agent.md: surviving ID '{aid}' missing"
