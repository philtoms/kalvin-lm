"""Documentation regression guard for the kvalue re-derivation cascade.

Run: uv run pytest tests/test_kvalue_spec_consistency.py -v

The §Retrieval cascade table in ``specs/kvalue.md`` must match the implemented
``derive_significance`` cascade in ``src/kalvin/expand.py``. The S1 branch is
``is_countersigned`` — *not* ``is_s1`` — because ``is_s1`` is defined as
``is_canon OR is_countersigned`` and placing it before the ``is_canon → S2``
row would swallow every canonical kline into S1, rendering that branch
unreachable (breaking KV-9 and contradicting the producer mapping
CANONIZES → S2). These tests lock the spec text to the code so the wording
cannot silently revert.

The guard also locks the ``derive_significance`` cascade descriptions in the
plans-layer trace ``plans/impl/kvalue-exchange-unit.md`` (§2 Design Decision D4
and §3 Task B) to the same implementation. The plan trace must not pair
``is_s1`` with the S1 band; it must use ``is_countersigned`` — mirroring the
spec table and the implemented cascade.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SPEC = REPO_ROOT / "specs" / "kvalue.md"
EXPAND = REPO_ROOT / "src" / "kalvin" / "expand.py"
PLAN = REPO_ROOT / "plans" / "impl" / "kvalue-exchange-unit.md"

# A band cell: S1, S2, S3, or S4.
_BAND = re.compile(r"S[1-4]")

# A plan cascade description pairs a structural predicate with a significance
# band via the U+2192 arrow (→) — not ASCII "->". A closing backtick/quote may
# sit between the predicate name and the arrow: e.g. `is_countersigned`→S1 in
# Task B vs is_countersigned→S1 in D4. The optional [`']? absorbs it so the
# guard catches both quote styles. (The unrelated Trainer._is_s1 /
# _S1_FRAME_THRESHOLD threshold concept at lines 243/308 never uses the arrow.)
_PLAN_S1_VIA_IS_S1 = re.compile(r"is_s1[`']?\s*→\s*S1")
_PLAN_S1_VIA_IS_COUNTERSIGNS = re.compile(r"is_countersigned[`']?\s*→\s*S1")


def _cascade_rows() -> list[tuple[str, str]]:
    """Return ``[(structural_test, band), ...]`` for the §Retrieval cascade table.

    Locates the markdown table whose header line contains
    ``Structural test (in order)`` and collects the pipe-delimited data rows
    that follow (up to the first non-pipe line). The header and the dashed
    separator row are skipped.
    """
    lines = SPEC.read_text(encoding="utf-8").splitlines()
    rows: list[tuple[str, str]] = []
    in_table = False
    for line in lines:
        if "Structural test (in order)" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break  # first non-pipe line ends the table
        cells = [c.strip() for c in line.split("|")]
        # cells[0] / cells[-1] are the empty fragments outside the pipes.
        if len(cells) < 3:
            continue
        left, right = cells[1], cells[2]
        if _BAND.fullmatch(right):
            rows.append((left, right))
    assert rows, (
        f"no re-derivation cascade table found in {SPEC} "
        "(expected a header containing 'Structural test (in order)')"
    )
    return rows


def _derive_significance_code() -> str:
    """Return the source of ``derive_significance`` with its docstring removed.

    Stripping the docstring keeps the cross-check honest: it inspects the
    executable cascade (the ``if`` statements), not the prose that merely
    documents it, so a code-only change cannot be masked by docstring wording.
    """
    text = EXPAND.read_text(encoding="utf-8").splitlines()
    start = next(
        (i for i, ln in enumerate(text) if ln.startswith("def derive_significance(")),
        None,
    )
    assert start is not None, "derive_significance not found in src/kalvin/expand.py"
    body: list[str] = [text[start]]
    for ln in text[start + 1 :]:
        if ln.strip() == "" or ln[0].isspace():
            body.append(ln)
        else:
            break  # next flush-left statement ends the function
    src = "\n".join(body)
    marker = '"""'
    first = src.find(marker)
    if first != -1:
        second = src.find(marker, first + 3)
        if second != -1:
            src = src[:first] + src[second + 3 :]
    return src


def test_re_derivation_cascade_band_order() -> None:
    """The cascade bands read S4, S1, S2, S3 in table order (first match wins)."""
    bands = [band for _, band in _cascade_rows()]
    assert bands == ["S4", "S1", "S2", "S3"], (
        "§Retrieval cascade band order must be S4, S1, S2, S3 "
        "(identity, countersigned, canon, otherwise — first match wins). "
        f"Got: {bands}"
    )


def test_re_derivation_s1_branch_uses_is_countersigned() -> None:
    """The S1 cascade row must reference ``is_countersigned``, not ``is_s1``."""
    s1_rows = [test for test, band in _cascade_rows() if band == "S1"]
    assert s1_rows, "§Retrieval cascade has no S1 row"
    s1_test = s1_rows[0]
    assert "is_countersigned" in s1_test and "is_s1" not in s1_test, (
        "The S1 cascade row must use is_countersigned(kline, model), not is_s1. "
        "is_s1 is is_canon OR is_countersigned; on the S1 row (before is_canon → S2) "
        "it would swallow every canonical kline into S1, breaking KV-9 "
        "(canonical → S2) and the producer mapping CANONIZES → S2. "
        f"Got: {s1_test!r}"
    )


def test_re_derivation_never_pairs_is_s1_with_s1_band() -> None:
    """Regression guard: no cascade row pairs ``is_s1`` with the S1 band.

    This is the exact wording reversion this task (KB-357) fixes. A future edit
    that rewrites the S1 row back to ``is_s1(kline)`` would reintroduce an
    unreachable ``is_canon → S2`` branch, breaking KV-9 (canonical → S2) and
    contradicting the producer mapping CANONIZES → S2.
    """
    offenders = [test for test, band in _cascade_rows() if band == "S1" and "is_s1" in test]
    assert not offenders, (
        "re-derivation cascade reverted: the S1 row references is_s1 (must be "
        "is_countersigned). is_s1 = is_canon OR is_countersigned; placed before "
        "is_canon → S2 it makes the canonical branch unreachable, breaking KV-9 "
        "(canonical → S2) and the producer mapping CANONIZES → S2. "
        f"Offenders: {offenders}"
    )


def test_re_derivation_cascade_keeps_is_canon_reachable() -> None:
    """The ``is_countersigned`` row must precede the ``is_canon`` row."""
    tests = [test for test, _ in _cascade_rows()]
    csigned = next((i for i, t in enumerate(tests) if "is_countersigned(" in t), None)
    canon = next((i for i, t in enumerate(tests) if "is_canon(" in t), None)
    assert csigned is not None, "no is_countersigned row in the cascade table"
    assert canon is not None, "no is_canon row in the cascade table"
    assert csigned < canon, (
        "is_countersigned row must come before is_canon so the is_canon → S2 "
        "branch is reachable (a canonical kline with no reciprocal falls through "
        f"is_countersigned to S2). Got countersigned at {csigned}, canon at {canon}."
    )


def test_spec_cascade_matches_derive_significance() -> None:
    """Lock the spec to the code: ``is_countersigned`` precedes ``is_canon`` in
    ``derive_significance``."""
    code = _derive_significance_code()
    csigned = code.find("is_countersigned(")
    canon = code.find("is_canon(")
    assert csigned != -1, "is_countersigned( not found in derive_significance body"
    assert canon != -1, "is_canon( not found in derive_significance body"
    assert csigned < canon, (
        "derive_significance must test is_countersigned before is_canon so a "
        "canonical kline reaches S2 (KV-9). The spec cascade mirrors this; the "
        "code must not drift out of sync."
    )


def test_plan_derive_significance_s1_branch_is_not_is_s1() -> None:
    """The plan's cascade descriptions must not pair ``is_s1`` with the S1 band.

    Mirrors the spec guard (KB-357) for the plans-layer trace
    ``plans/impl/kvalue-exchange-unit.md`` (§2 D4 and §3 Task B). The S1 branch
    must be ``is_countersigned``: ``is_s1`` is ``is_canon OR is_countersigned``,
    so placing it before ``is_canon → S2`` swallows every canonical kline into
    S1, rendering the ``is_canon → S2`` branch unreachable (breaking KV-9 and
    contradicting the producer mapping CANONIZES → S2). The U+2192 arrow (→)
    scopes the check to the cascade descriptions, excluding the unrelated
    ``Trainer._is_s1`` / ``_S1_FRAME_THRESHOLD`` threshold concept (which never
    uses the arrow).
    """
    text = PLAN.read_text(encoding="utf-8")
    offenders = _PLAN_S1_VIA_IS_S1.findall(text)
    assert not offenders, (
        "plan derive_significance cascade reverted: it pairs is_s1 with the S1 "
        "band (in §2 D4 and/or §3 Task B). The S1 branch must be is_countersigned. "
        "is_s1 = is_canon OR is_countersigned; placed before is_canon → S2 it "
        "makes the canonical branch unreachable, breaking KV-9 (canonical → S2) "
        "and the producer mapping CANONIZES → S2. "
        f"Offenders: {offenders}"
    )


def test_plan_derive_significance_s1_branch_is_is_countersigned() -> None:
    """The plan's cascade descriptions must pair ``is_countersigned`` with S1.

    Confirms the corrected wording (KB-358) is present in the plans-layer trace,
    matching the §Retrieval cascade table and the implemented
    ``derive_significance``.
    """
    text = PLAN.read_text(encoding="utf-8")
    assert _PLAN_S1_VIA_IS_COUNTERSIGNS.search(text), (
        "plan derive_significance cascade must describe the S1 branch as "
        "is_countersigned→S1 (matching the spec table and derive_significance)."
    )


def test_plan_predicate_reuse_list_does_not_name_is_s1() -> None:
    """The D4 predicate-reuse list must not list ``is_s1`` as a reused predicate.

    ``derive_significance`` invokes ``is_countersigned`` for the S1 branch
    (never ``is_s1``), so the plan's description of which structural predicates
    it reuses must not name ``is_s1``.
    """
    lines = PLAN.read_text(encoding="utf-8").splitlines()
    idx = next(
        (i for i, ln in enumerate(lines) if "reuses the existing structural predicates" in ln),
        None,
    )
    assert idx is not None, (
        "could not locate the D4 predicate-reuse list "
        "('reuses the existing structural predicates') in the plan"
    )
    # The predicate list may wrap onto the following line(s); check the anchor
    # line and the next two.
    block = "\n".join(lines[idx : idx + 3])
    assert "is_s1" not in block, (
        "the D4 predicate-reuse list still names is_s1, but derive_significance "
        "invokes is_countersigned for the S1 branch (never is_s1). "
        f"Block:\n{block}"
    )
