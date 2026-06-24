"""Documentation regression guard for the ``harness.yaml`` config-name policy.

Run: uv run pytest tests/test_config_name_consistency.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Matches a bare ``harness.yaml`` that is NOT preceded by ``training.`` (i.e. it
# does NOT match the canonical ``training.harness.yaml``). A fixed-width
# negative lookbehind keeps this robust against incidental prefixes while still
# flagging ``harness.yaml`` at the start of a line or after a backtick/space.
_BARE_HARNESS_YAML = re.compile(r"(?<!training\.)harness\.yaml")

# Matches ANY ``harness.yaml`` occurrence, including the canonical
# ``training.harness.yaml``.
# (``docs/cascade-development.md`` Structural Rule #5), so specs
# must carry no ``harness.yaml`` literal at all.
_HARNESS_YAML = re.compile(r"harness\.yaml")

# (see docs/cascade-development.md "Rule #5 — what counts as a
# file name"). Uniform/context-free: every token here is banned from specs/
# outright (no context filter needed, which is why a flat _offenders() scan
# suffices). ``<``, ``>``, and ``/`` are literal in the pattern.
_DATA_FILE_TOKENS = re.compile(
    r"config\.json|cmd\.json|status\.json|events\.jsonl|meta\.json|"
    r"state\.json|model\.bin|runs/<n>|curricula/<slug>|curricula/"
)


def _offenders(path: Path, pattern: re.Pattern[str] = _BARE_HARNESS_YAML) -> list[str]:
    """Return ``["file:lineno:line", ...]`` for each *pattern* match in *path*.

    Defaults to the bare-only ``_BARE_HARNESS_YAML`` pattern so the plan guards
    can call ``_offenders(plan)`` unchanged; the specs guard passes
    ``_HARNESS_YAML`` explicitly.
    """
    if not path.exists():
        return [f"{path}: <missing file>"]
    found: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern.search(line):
            found.append(f"{path}:{lineno}:{line}")
    return found


def test_specs_reference_harness_config_by_concept_not_filename() -> None:
    """Every ``specs/*.md`` must reference the harness config by concept, not filename.

    Per ``docs/cascade-development.md`` Structural Rule
    #5 ("No file names in specs. Code locations belong in plans only."), the WHAT
    layer describes the harness config by concept -- "the project harness config"
    / "the per-session harness config" -- never by a literal filename. Filenames
    and the concrete CLI invocation live in ``plans/``. This test rejects any
    ``harness.yaml`` form (bare or canonical).
    """
    offenders: list[str] = []
    for spec in sorted(REPO_ROOT.glob("specs/*.md")):
        offenders.extend(_offenders(spec, _HARNESS_YAML))
    assert not offenders, (
        "'harness.yaml' literal found in specs/ (specs must reference the harness "
        "config by concept, not filename -- see docs/cascade-development.md "
        "Structural Rule #5):\n" + "\n".join(offenders)
    )


def test_specs_reference_data_files_by_concept_not_filename() -> None:
    """Every ``specs/*.md`` must reference auto-tune session data files by concept.

    Structural Rule #5 ("No file names in specs. Code locations
    belong in plans only.") to the auto-tune session data-format filenames. Per
    the "Rule #5 — what counts as a file name" clarification in
    ``docs/cascade-development.md``, the WHAT layer describes each persisted
    artefact by concept (e.g. "the session configuration", "the event stream",
    "the run directory"); the concrete names and the on-disk layout live in the
    plan layer (``plans/impl/auto-tune-session-layout.md`` holds the
    concept→file mapping). This test rejects any of the purged data-format
    filenames or path templates in ``specs/``.
    """
    offenders: list[str] = []
    for spec in sorted(REPO_ROOT.glob("specs/*.md")):
        offenders.extend(_offenders(spec, _DATA_FILE_TOKENS))
    assert not offenders, (
        "purged data-format filename/path template found in specs/ (specs must "
        "reference session data files by concept, not filename -- see "
        "docs/cascade-development.md 'Rule #5 -- what counts as a file name' "
        "and plans/impl/auto-tune-session-layout.md):\n" + "\n".join(offenders)
    )


def test_harness_server_plan_usage_command_uses_canonical_name() -> None:
    """The harness-server plan's CLI usage command must use the canonical name."""
    plan = REPO_ROOT / "plans" / "implement-harness-server.md"
    offenders = _offenders(plan)
    assert not offenders, (
        "stale bare 'harness.yaml' found in plans/implement-harness-server.md "
        "(must be 'training.harness.yaml'):\n" + "\n".join(offenders)
    )


def test_all_plans_use_canonical_config_filename() -> None:
    """Every ``plans/**/*.md`` (recursive) must use the canonical name.
    """
    offenders: list[str] = []
    for plan in sorted(REPO_ROOT.glob("plans/**/*.md")):
        offenders.extend(_offenders(plan))
    assert not offenders, (
        "stale bare 'harness.yaml' found in plans/ (must be 'training.harness.yaml'):\n"
        + "\n".join(offenders)
    )
