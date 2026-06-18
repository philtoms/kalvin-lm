"""Documentation regression guard for the ``harness.yaml`` config-name policy.

KB-311 renamed the project config file ``harness.yaml`` -> ``training.harness.yaml``
at the repo root and refreshed the code, README, and docstrings. KB-314 swept the
remaining stale ``harness.yaml`` literals out of the behavioural-contract layer
(``specs/*.md``) and the single plainly user-facing usage command in
``plans/implement-harness-server.md``, pinning the canonical name in place.

KB-322 then applied ``docs/cascade-development.md`` Structural Rule #5 ("No file
names in specs. Code locations belong in plans only."): it removed the
``harness.yaml`` literal from ``specs/`` *entirely*, rewording the affected
rules to reference the harness config by concept ("the project harness config"
/ "the per-session harness config"). The concrete CLI invocation (``python -m
training.harness --config training.harness.yaml``) now lives only in
``plans/implement-harness-server.md``.

Two policies, two patterns:

- The specs guard (``test_specs_reference_harness_config_by_concept_not_filename``)
  uses ``_HARNESS_YAML``, which matches ANY ``harness.yaml`` form (bare or
  canonical) -- specs must carry no harness-config filename at all.
- The plan guards use ``_BARE_HARNESS_YAML`` (negative lookbehind on
  ``training.``) -- plans legitimately hold filenames/code locations per rule #5,
  but must still use the canonical ``training.harness.yaml``, never the bare
  pre-rename name.

Note: KB-314 originally scoped the plan guard to the single
``implement-harness-server.md`` usage command. KB-321 extended it to the whole
HOW layer: ``test_all_plans_use_canonical_config_filename`` now scans every
``plans/**/*.md`` recursively (including ``plans/impl/``), so any plan that
names the project config must use ``training.harness.yaml``. The bare
``harness.yaml`` literals that previously lived in ``plans/impl/curriculum.md``,
``plans/impl/reactive-delegation.md`` and ``plans/role-based-routing.md`` were
swept to the canonical name (KB-316) and are now pinned here.

Run: uv run pytest tests/test_config_name_consistency.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

# Match the KB-311 pattern used by tests/test_harness_cli.py.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Matches a bare ``harness.yaml`` that is NOT preceded by ``training.`` (i.e. it
# does NOT match the canonical ``training.harness.yaml``). A fixed-width
# negative lookbehind keeps this robust against incidental prefixes while still
# flagging ``harness.yaml`` at the start of a line or after a backtick/space.
_BARE_HARNESS_YAML = re.compile(r"(?<!training\.)harness\.yaml")

# Matches ANY ``harness.yaml`` occurrence, including the canonical
# ``training.harness.yaml``. KB-322 tightened the specs policy from "use the
# canonical name" (KB-314) to "reference the harness config by concept, not by
# any filename" (``docs/cascade-development.md`` Structural Rule #5), so specs
# must carry no ``harness.yaml`` literal at all.
_HARNESS_YAML = re.compile(r"harness\.yaml")


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

    KB-314 canonicalised the literal to ``training.harness.yaml``; KB-322 removed
    it from specs entirely. Per ``docs/cascade-development.md`` Structural Rule
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

    KB-321 extended this guard from the single ``implement-harness-server.md``
    usage command to the entire HOW layer. Any plan — including future additions
    under ``plans/impl/`` — that names the project config must use
    ``training.harness.yaml``, never the pre-rename bare ``harness.yaml``. The
    recursive ``plans/**/*.md`` glob is intentional so new plan files are covered
    automatically without having to extend a file list.
    """
    offenders: list[str] = []
    for plan in sorted(REPO_ROOT.glob("plans/**/*.md")):
        offenders.extend(_offenders(plan))
    assert not offenders, (
        "stale bare 'harness.yaml' found in plans/ (must be 'training.harness.yaml'):\n"
        + "\n".join(offenders)
    )
