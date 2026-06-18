"""Scoped regression guard for KB-316: bare ``harness.yaml`` audit.

KB-311 renamed the project config ``harness.yaml`` -> ``training.harness.yaml``.
KB-316 swept the remaining stale bare ``harness.yaml`` literals out of the HOW
layer (``plans/``) and the test suite's docstrings/comments — the *code* in
those tests already used the canonical name; only the prose was stale. This
test pins the sweep so the pre-rename filename cannot silently return to those
files.

Scope and exclusions
--------------------
This guard scans exactly the seven files KB-316 edited (see ``_SCOPE_FILES``):
three ``plans/`` files and four ``tests/`` files. It is deliberately disjoint
from the other rename follow-on tasks:

* ``tests/test_config_name_consistency.py`` (KB-314) — guards ``specs/*.md`` and
  ``plans/implement-harness-server.md``.
* ``src/training/harness/__main__.py`` (KB-312) and
  ``src/training/participants/auto_tune/session.py`` (KB-313).

Detection
---------
``_STALE_HARNESS_YAML`` matches a *bare* ``harness.yaml`` (the obsolete name)
but NOT the canonical ``training.harness.yaml`` (negative lookbehind on
``training.``) and NOT the distinct ``full_harness.yaml`` test fixture
(negative lookbehind on ``full_``). Both lookbehinds are fixed-width, so
Python's ``re`` accepts them.

Legitimate exemption
--------------------
``tests/test_auto_tune_session.py`` contains KB-313's
``TestSessionDirInitDocstring`` regression test, which *deliberately* references
the pre-rename filename to assert its absence from the ``init`` docstring (its
negative assertion at the end of that class would break if the literal were
swapped). That class's bare ``harness.yaml`` occurrences are legitimate; the
scan skips its body — from the ``class`` line to the next top-level
``class``/``def`` (or end-of-file) — so they are not flagged.

Run: uv run pytest tests/test_harness_config_name_audit.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

# Match the KB-311 pattern used by tests/test_harness_cli.py and KB-314's guard.
REPO_ROOT = Path(__file__).resolve().parent.parent

# A bare obsolete filename. Excludes the canonical ``training.harness.yaml``
# (``training.`` lookbehind) and the ``full_harness.yaml`` fixture (``full_``
# lookbehind). Both lookbehinds are fixed-width, so Python's ``re`` accepts them.
_STALE_HARNESS_YAML = re.compile(r"(?<!full_)(?<!training\.)harness\.yaml")

# KB-313's rename-regression test deliberately names the OLD filename to assert
# its absence; its bare ``harness.yaml`` occurrences are legitimate and exempt.
_EXEMPT_CLASS_START = re.compile(r"^class\s+TestSessionDirInitDocstring\b")
_TOP_LEVEL_DEF = re.compile(r"^(class|def)\s")

# The seven files KB-316 edited in Steps 2-3. Disjoint from KB-312/KB-313/KB-314.
_SCOPE_FILES = [
    "plans/impl/curriculum.md",
    "plans/impl/reactive-delegation.md",
    "plans/role-based-routing.md",
    "tests/test_auto_tune_integration.py",
    "tests/test_auto_tune_lifecycle.py",
    "tests/test_auto_tune_session.py",
    "tests/test_harness_cli.py",
]


def _offenders(rel: str) -> list[str]:
    """Return ``["rel:lineno:line", ...]`` for each stale bare ``harness.yaml``.

    Skips the body of ``TestSessionDirInitDocstring`` (KB-313's legitimate
    rename-regression test), whose bare ``harness.yaml`` references are
    intentional and must not be token-swapped.
    """
    path = REPO_ROOT / rel
    if not path.exists():
        return [f"{rel}: <missing file>"]
    found: list[str] = []
    in_exempt_class = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if _EXEMPT_CLASS_START.match(line):
            in_exempt_class = True
        elif _TOP_LEVEL_DEF.match(line):
            # A new top-level class/function closes the exempt class block.
            in_exempt_class = False
        if in_exempt_class:
            continue
        if _STALE_HARNESS_YAML.search(line):
            found.append(f"{rel}:{lineno}:{line}")
    return found


def test_no_stale_bare_harness_yaml_in_audit_scope() -> None:
    """No bare ``harness.yaml`` (the pre-rename name) remains in KB-316's scope.

    The guard passes once Steps 2-3 have swapped every stale literal to the
    canonical ``training.harness.yaml`` (and reworded the single historical
    reference at ``test_harness_cli.py``). The ``TestSessionDirInitDocstring``
    class in ``test_auto_tune_session.py`` is an intentional, exempted
    rename-regression test (see module docstring).
    """
    offenders: list[str] = []
    for rel in _SCOPE_FILES:
        offenders.extend(_offenders(rel))
    assert not offenders, (
        "stale bare 'harness.yaml' found in KB-316's audit scope "
        "(must be 'training.harness.yaml'):\n" + "\n".join(offenders)
    )
