"""Documentation regression guard for KB-314: ``harness.yaml`` -> ``training.harness.yaml`` rename.

KB-311 renamed the project config file ``harness.yaml`` -> ``training.harness.yaml``
at the repo root and refreshed the code, README, and docstrings. KB-314 swept the
remaining stale ``harness.yaml`` literals out of the behavioural-contract layer
(``specs/*.md``) and the single plainly user-facing usage command in
``plans/implement-harness-server.md``.

This test locks the rename in so the pre-rename filename cannot silently return
to ``specs/`` or to that plan's CLI usage command. The detection pattern matches
a *bare* ``harness.yaml`` (the obsolete name) but NOT the canonical
``training.harness.yaml``.

Note: this guard is intentionally scoped to the layers KB-314 touched. Other
``plans/*.md`` files retain historical ``harness.yaml`` references (headings,
code locations, task trees, implementation instructions) that are out of scope;
those are tracked as follow-up work, not asserted here.

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


def _offenders(path: Path) -> list[str]:
    """Return ``["file:lineno:line", ...]`` for each bare ``harness.yaml`` in *path*."""
    if not path.exists():
        return [f"{path}: <missing file>"]
    found: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if _BARE_HARNESS_YAML.search(line):
            found.append(f"{path}:{lineno}:{line}")
    return found


def test_specs_do_not_reference_stale_config_filename() -> None:
    """Every ``specs/*.md`` must use the canonical ``training.harness.yaml``."""
    offenders: list[str] = []
    for spec in sorted(REPO_ROOT.glob("specs/*.md")):
        offenders.extend(_offenders(spec))
    assert not offenders, (
        "stale bare 'harness.yaml' found in specs/ (must be 'training.harness.yaml'):\n"
        + "\n".join(offenders)
    )


def test_harness_server_plan_usage_command_uses_canonical_name() -> None:
    """The harness-server plan's CLI usage command must use the canonical name."""
    plan = REPO_ROOT / "plans" / "implement-harness-server.md"
    offenders = _offenders(plan)
    assert not offenders, (
        "stale bare 'harness.yaml' found in plans/implement-harness-server.md "
        "(must be 'training.harness.yaml'):\n" + "\n".join(offenders)
    )
