"""Annotation guard for stale references in tracked worktree-context files.

The file ``.pi/git-worktrees/context/simplification-kscript-relationships.md`` is
a historical planning checklist from the (completed)
``simplification-kscript-relationships`` worktree (BWD operator removal). Item 9
referenced ``docs/tokenizer-significance.md``; that file was later deleted by
KB-275 (Mod/tokenizer cleanup) for reasons unrelated to BWD. KB-284 annotated
that single line with a parenthetical ``Resolution`` note so automated grep
sweeps do not flag it as a bare stale reference.

This test pins the annotation in place. If the follow-up gitignore task moves
``.pi/git-worktrees/`` out of tracking, this test should be removed — it only
guards tracked content.
"""

from __future__ import annotations

from pathlib import Path

#: Repository root, resolved from this test file.
REPO = Path(__file__).resolve().parent.parent

#: The tracked worktree-context file containing the annotated checklist item.
CONTEXT_FILE = (
    REPO / ".pi" / "git-worktrees" / "context" / "simplification-kscript-relationships.md"
)


def test_context_file_exists() -> None:
    assert CONTEXT_FILE.exists(), f"context file not found at {CONTEXT_FILE}"


def test_tokenizer_significance_line_is_annotated() -> None:
    """The tokenizer-significance checklist item must carry a KB-275 resolution."""
    lines = CONTEXT_FILE.read_text().splitlines()
    matches = [line for line in lines if "tokenizer-significance.md" in line]
    assert matches, "no line referencing tokenizer-significance.md found"

    # There should be exactly one such line in the historical checklist.
    assert len(matches) == 1, (
        f"expected exactly one tokenizer-significance.md reference, "
        f"found {len(matches)}"
    )
    line = matches[0]
    assert "KB-275" in line, (
        f"tokenizer-significance line must reference KB-275; got: {line!r}"
    )
    assert "Resolution" in line, (
        f"tokenizer-significance line must include a Resolution note; got: {line!r}"
    )
