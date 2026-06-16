"""Regression guard for agent-tooling artifacts under ``.pi/git-worktrees/``.

KB-287 decided that ``.pi/git-worktrees/`` should be gitignored, exactly like
the already-ignored ``.kb`` and ``.worktrees`` directories. The directory holds
ephemeral agent-tooling artifacts — worktree session logs, planning scratch, and
a ``registry.json`` whose ``mainSessionFile`` is a machine-specific absolute path
that is meaningless on any other machine. Keeping it tracked meant stale
references in scratch files leaked into grep sweeps (the situation KB-284 had to
patch with a guard test, since removed).

These tests pin the gitignore decision in place: the ``.gitignore`` must list
the directory, and no files under it may be tracked by git.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

#: Repository root, resolved from this test file.
REPO = Path(__file__).resolve().parent.parent

#: Path to the repo .gitignore.
GITIGNORE = REPO / ".gitignore"

#: The agent-tooling directory that must be ignored.
GIT_WORKTREES_DIR = ".pi/git-worktrees/"


def test_gitignore_lists_git_worktrees_dir() -> None:
    """``.gitignore`` must contain a pattern for the agent worktree directory."""
    expected = GIT_WORKTREES_DIR.rstrip("/")
    patterns = GITIGNORE.read_text().splitlines()
    matches = [line for line in patterns if line.strip().rstrip("/") == expected]
    assert matches, (
        f".gitignore must list {expected} so agent-tooling artifacts "
        f"(session logs, planning scratch, registry) are not tracked"
    )


def test_no_tracked_files_under_git_worktrees() -> None:
    """No files under ``.pi/git-worktrees/`` may be tracked by git."""
    result = subprocess.run(
        ["git", "ls-files", GIT_WORKTREES_DIR],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    assert not tracked, (
        f"expected no tracked files under {GIT_WORKTREES_DIR}, but found: "
        f"{tracked}"
    )
