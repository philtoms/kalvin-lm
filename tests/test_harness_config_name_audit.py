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
Some rename-regression tests *deliberately* reference the pre-rename filename
inside phrase-scoped negative assertions (asserting the stale literal's
absence). For those the bare ``harness.yaml`` is required, not stale, and the
scan skips their body so they are not flagged:

* ``TestSessionDirInitDocstring`` (KB-313) — a class-level guard; the scan
  skips it from the ``class`` line to the next top-level ``class``/``def`` (or
  end-of-file).
* ``test_*_docstring_uses_canonical_filename`` methods (KB-317, e.g.
  ``test_host_port_override_docstring_uses_canonical_filename``, plus the
  KB-325 follow-up) — method-level guards using the same phrase-scoped
  technique; the scan skips each from its ``def`` line to the next sibling
  ``def`` (4-space indent) or the next top-level ``class``/``def``.
  Generalising to the naming convention (rather than hard-coding one method
  name) covers KB-325's planned guard by construction. KB-326 reconciled this
  exemption with KB-317's method-level drift-guard.

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

# KB-317 (and KB-325) added METHOD-LEVEL drift-guards that also deliberately
# name the OLD filename inside phrase-scoped negative assertions. They follow
# the ``test_*_docstring_uses_canonical_filename`` naming convention, so the
# exemption generalises to that convention rather than hard-coding one name.
_EXEMPT_METHOD_START = re.compile(r"^    def\s+test_\w+_docstring_uses_canonical_filename\b")
# A sibling method (same 4-space indent) closes a method-level exemption so it
# cannot blanket-mask the rest of the class. A top-level class/def also closes
# it (handled by ``_TOP_LEVEL_DEF`` in ``_scan_lines``).
_INDENTED_METHOD_DEF = re.compile(r"^    def\s")

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


def _scan_lines(lines: list[str]) -> list[tuple[int, str]]:
    """Return ``(lineno, line)`` for each stale bare ``harness.yaml`` line.

    Skips two kinds of legitimate drift-guard region, both of which
    *deliberately* name the pre-rename filename inside phrase-scoped negative
    assertions (so the literal must not be token-swapped):

    * ``TestSessionDirInitDocstring`` (KB-313) — a top-level class; the region
      runs from its ``class`` line to the next top-level ``class``/``def``.
    * ``test_*_docstring_uses_canonical_filename`` methods (KB-317 / KB-325) —
      4-space-indented method drift-guards; the region runs from the ``def``
      line to the next sibling ``def`` (same 4-space indent) or to any
      top-level ``class``/``def``.
    """
    found: list[tuple[int, str]] = []
    in_exempt_class = False
    in_exempt_method = False
    for lineno, line in enumerate(lines, start=1):
        # Class-region state (KB-313): opens at the exempt class line, closes
        # at the next top-level class/function.
        if _EXEMPT_CLASS_START.match(line):
            in_exempt_class = True
        elif _TOP_LEVEL_DEF.match(line):
            in_exempt_class = False
        # Method-region state (KB-317/KB-325): opens at the exempt method line,
        # closes at the next sibling method (4-space indent) or any top-level
        # class/function. A top-level class/def closes BOTH regions.
        if _EXEMPT_METHOD_START.match(line):
            in_exempt_method = True
        elif _TOP_LEVEL_DEF.match(line) or _INDENTED_METHOD_DEF.match(line):
            in_exempt_method = False
        if in_exempt_class or in_exempt_method:
            continue
        if _STALE_HARNESS_YAML.search(line):
            found.append((lineno, line))
    return found


def _offenders(rel: str) -> list[str]:
    """Return ``["rel:lineno:line", ...]`` for each stale bare ``harness.yaml``.

    Delegates the line scan to ``_scan_lines`` and formats each hit as
    ``rel:lineno:line``. Skips the exempt drift-guard regions documented on
    ``_scan_lines`` (KB-313's class guard and KB-317/KB-325's method guards),
    whose bare ``harness.yaml`` references are intentional.
    """
    path = REPO_ROOT / rel
    if not path.exists():
        return [f"{rel}: <missing file>"]
    lines = path.read_text(encoding="utf-8").splitlines()
    return [f"{rel}:{lineno}:{line}" for lineno, line in _scan_lines(lines)]


def test_no_stale_bare_harness_yaml_in_audit_scope() -> None:
    """No bare ``harness.yaml`` (the pre-rename name) remains in KB-316's scope.

    The guard passes once Steps 2-3 have swapped every stale literal to the
    canonical ``training.harness.yaml`` (and reworded the single historical
    reference at ``test_harness_cli.py``). Two kinds of deliberate
    rename-regression guard are intentionally exempted (see module docstring):
    the ``TestSessionDirInitDocstring`` class (KB-313) and the
    ``test_*_docstring_uses_canonical_filename`` method drift-guards
    (KB-317/KB-325), both in ``test_auto_tune_session.py``.
    """
    offenders: list[str] = []
    for rel in _SCOPE_FILES:
        offenders.extend(_offenders(rel))
    assert not offenders, (
        "stale bare 'harness.yaml' found in KB-316's audit scope "
        "(must be 'training.harness.yaml'):\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# Unit tests for the exemption boundary logic (KB-326)
# ---------------------------------------------------------------------------
# These exercise the pure ``_scan_lines`` helper directly with synthetic line
# lists, locking the surgical behaviour of both exemption kinds: drift-guards
# are skipped, but genuine stale literals elsewhere are still caught.


def test_exempt_method_drift_guard_is_skipped() -> None:
    """A KB-317 ``test_*_docstring_uses_canonical_filename`` body is skipped."""
    lines = [
        "class TestSessionDirInit:",
        "    def test_host_port_override_docstring_uses_canonical_filename(self) -> None:",
        '        assert "overrides harness.yaml defaults" not in (doc or "")',
        "        return",
    ]
    assert _scan_lines(lines) == []


def test_exempt_class_drift_guard_is_skipped() -> None:
    """The KB-313 ``TestSessionDirInitDocstring`` class body is skipped."""
    lines = [
        "class TestSessionDirInitDocstring:",
        "    def test_init_docstring_names_canonical_harness_file(self) -> None:",
        '        assert "Reads ``harness.yaml``" not in doc',
        "        return",
        "",
        "class NextTopLevel:",
        "    pass",
    ]
    assert _scan_lines(lines) == []


def test_non_exempt_bare_harness_yaml_is_flagged() -> None:
    """A bare ``harness.yaml`` in a PLAIN (non-exempt) method is flagged."""
    lines = [
        "class TestSomething:",
        "    def test_plain(self) -> None:",
        '        """reads harness.yaml when no overrides given."""',
        "        return",
    ]
    result = _scan_lines(lines)
    assert len(result) == 1
    lineno, line = result[0]
    assert lineno == 3
    assert "harness.yaml" in line


def test_exempt_method_closes_at_next_sibling_def() -> None:
    """Exempt method region closes at the next sibling def (4-space indent)."""
    lines = [
        "class TestSessionDirInit:",
        "    def test_host_port_override_docstring_uses_canonical_filename(self) -> None:",
        '        assert "overrides harness.yaml defaults" not in (doc or "")',
        "    def test_next_sibling(self) -> None:",
        "        # legacy harness.yaml comment",
        "        return",
    ]
    result = _scan_lines(lines)
    assert len(result) == 1
    lineno, line = result[0]
    assert lineno == 5
    assert "harness.yaml" in line


def test_canonical_training_harness_yaml_never_flagged() -> None:
    """The canonical ``training.harness.yaml`` name is never flagged."""
    lines = [
        'x = root / "training.harness.yaml"',
        "        # training.harness.yaml is canonical",
    ]
    assert _scan_lines(lines) == []


def test_exempt_method_closes_at_next_top_level_class() -> None:
    """Exempt method region also closes at the next top-level ``class``/``def``.

    After a ``test_*_docstring_uses_canonical_filename`` method, a top-level
    ``class Other:`` (indent 0) must NOT stay masked: a bare ``harness.yaml``
    in its body is still flagged. The ``_TOP_LEVEL_DEF`` boundary closes the
    method exemption (as well as the class exemption).
    """
    lines = [
        "class TestSessionDirInit:",
        "    def test_host_port_override_docstring_uses_canonical_filename(self) -> None:",
        '        assert "overrides harness.yaml defaults" not in (doc or "")',
        "",
        "class Other:",
        "    # legacy harness.yaml comment",
        "    pass",
    ]
    result = _scan_lines(lines)
    assert len(result) == 1
    lineno, line = result[0]
    assert lineno == 6
    assert "harness.yaml" in line


def test_module_level_bare_harness_yaml_is_flagged() -> None:
    """A bare ``harness.yaml`` at module top level is flagged.

    Confirms the baseline ``_STALE_HARNESS_YAML`` detection is unchanged
    outside any exemption block: a module-level literal (no enclosing class or
    method) is still caught.
    """
    lines = [
        "# legacy reference to harness.yaml here",
        "x = 1",
    ]
    result = _scan_lines(lines)
    assert len(result) == 1
    lineno, line = result[0]
    assert lineno == 1
    assert "harness.yaml" in line


def test_full_harness_fixture_never_flagged() -> None:
    """The ``full_harness.yaml`` test fixture is never flagged.

    ``_STALE_HARNESS_YAML`` carries a ``full_`` negative lookbehind so the
    distinct ``full_harness.yaml`` fixture (not the renamed config) is not
    mistaken for the obsolete bare name. This pins that lookbehind alongside
    the ``training.`` lookbehind covered above.
    """
    lines = [
        'fixture = root / "full_harness.yaml"',
        "        # full_harness.yaml fixture",
    ]
    assert _scan_lines(lines) == []
