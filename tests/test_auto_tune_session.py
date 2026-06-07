"""Tests for auto-tune session management (SessionConfig, SessionDir).

Covers spec test matrix AT-1, AT-2, AT-3 plus host/port override and
load round-trip.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from participants.auto_tune.session import SessionConfig, SessionDir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repo with an initial commit and return its path.

    Tests that call ``SessionDir.init()`` must operate here to avoid
    polluting the real project repo with branch switches.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    # Initial commit so HEAD is defined
    (repo / "README.md").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


# ---------------------------------------------------------------------------
# SessionConfig
# ---------------------------------------------------------------------------


class TestSessionConfig:
    """Tests for the SessionConfig dataclass."""

    def test_construction_defaults(self) -> None:
        cfg = SessionConfig(
            session="exp-1",
            curriculum="curricula/topic.md",
            harness_url="ws://localhost:8765",
        )
        assert cfg.session == "exp-1"
        assert cfg.curriculum == "curricula/topic.md"
        assert cfg.harness_url == "ws://localhost:8765"
        assert cfg.model_path is None
        assert cfg.run_counter == 0
        assert cfg.created_from_branch == ""
        assert cfg.created_from_commit == ""

    def test_construction_all_fields(self) -> None:
        cfg = SessionConfig(
            session="exp-2",
            curriculum="curricula/other.md",
            harness_url="ws://10.0.0.1:9999",
            model_path="data/agent.bin",
            run_counter=3,
            created_from_branch="main",
            created_from_commit="abc123",
        )
        assert cfg.model_path == "data/agent.bin"
        assert cfg.run_counter == 3
        assert cfg.created_from_branch == "main"
        assert cfg.created_from_commit == "abc123"

    def test_serialisation_round_trip(self) -> None:
        original = SessionConfig(
            session="round-trip",
            curriculum="curriculum.md",
            harness_url="ws://localhost:8765",
            model_path="data/agent.bin",
            run_counter=5,
            created_from_branch="develop",
            created_from_commit="deadbeef",
        )
        d = original.to_dict()
        restored = SessionConfig.from_dict(d)
        assert restored == original

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = SessionConfig.from_dict({
            "session": "s",
            "curriculum": "c.md",
            "harness_url": "ws://x:1",
            "future_field": "ignored",
        })
        assert cfg.session == "s"
        assert not hasattr(cfg, "future_field")

    def test_to_dict_is_json_serialisable(self) -> None:
        cfg = SessionConfig(
            session="json",
            curriculum="c.md",
            harness_url="ws://localhost:8765",
        )
        text = json.dumps(cfg.to_dict())
        assert '"session": "json"' in text


# ---------------------------------------------------------------------------
# SessionDir — paths
# ---------------------------------------------------------------------------


class TestSessionDirPaths:
    """Tests for SessionDir path properties (no filesystem side effects)."""

    def test_config_path(self) -> None:
        sd = SessionDir(root=Path("/project"), _session="exp-1")
        assert sd.config_path == Path("/project/auto-tune/exp-1/config.json")

    def test_cmd_path(self) -> None:
        sd = SessionDir(root=Path("/project"), _session="exp-1")
        assert sd.cmd_path == Path("/project/auto-tune/exp-1/cmd.json")

    def test_status_path(self) -> None:
        sd = SessionDir(root=Path("/project"), _session="exp-1")
        assert sd.status_path == Path("/project/auto-tune/exp-1/status.json")

    def test_events_path(self) -> None:
        sd = SessionDir(root=Path("/project"), _session="exp-1")
        assert sd.events_path == Path("/project/auto-tune/exp-1/events.jsonl")

    def test_runs_dir(self) -> None:
        sd = SessionDir(root=Path("/project"), _session="exp-1")
        assert sd.runs_dir == Path("/project/auto-tune/exp-1/runs")

    def test_config_raises_without_bind(self) -> None:
        sd = SessionDir(root=Path("/project"))
        with pytest.raises(ValueError, match="No config loaded"):
            _ = sd.config


# ---------------------------------------------------------------------------
# SessionDir — init (AT-1, AT-2, AT-3)
# ---------------------------------------------------------------------------


class TestSessionDirInit:
    """Tests for SessionDir.init() covering AT-1, AT-2, AT-3."""

    def test_at1_creates_directory_structure(self, tmp_git_repo: Path) -> None:
        """AT-1: init creates session directory with config.json, events.jsonl, runs/."""
        sd = SessionDir.init(
            "exp-1",
            curriculum="curricula/topic.md",
            root=tmp_git_repo,
        )
        worktree_path = tmp_git_repo / ".worktrees" / "auto-tune" / "exp-1"
        session_dir = worktree_path / "auto-tune" / "exp-1"

        assert sd.config_path.is_file(), "config.json must exist"
        assert sd.events_path.is_file(), "events.jsonl must exist"
        assert sd.runs_dir.is_dir(), "runs/ subdirectory must exist"
        assert sd._root == worktree_path, "root should be the worktree"

    def test_at2_creates_git_branch(self, tmp_git_repo: Path) -> None:
        """AT-2: init creates auto-tune/<session> git branch as a worktree."""
        SessionDir.init(
            "exp-2",
            curriculum="curricula/topic.md",
            root=tmp_git_repo,
        )
        # The branch should exist
        branches = subprocess.run(
            ["git", "branch", "--list", "auto-tune/exp-2"],
            cwd=tmp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert "auto-tune/exp-2" in branches

        # Main repo should NOT be on the auto-tune branch
        current = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=tmp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert current != "auto-tune/exp-2", "main repo should stay on original branch"

        # The worktree should exist and be on the auto-tune branch
        worktree_current = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=tmp_git_repo / ".worktrees" / "auto-tune" / "exp-2",
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert worktree_current == "auto-tune/exp-2"

    def test_at3_config_json_contents(self, tmp_git_repo: Path) -> None:
        """AT-3: config.json has correct created_from_branch, commit, harness_url, model_path."""
        # Capture the starting state
        original_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=tmp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        original_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=tmp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        sd = SessionDir.init(
            "exp-3",
            curriculum="curricula/topic.md",
            root=tmp_git_repo,
        )

        cfg = sd.config
        assert cfg.created_from_branch == original_branch
        assert cfg.created_from_commit == original_commit
        assert cfg.harness_url == "ws://localhost:8765"
        assert cfg.model_path == "data/agent.bin"

    def test_host_port_override(self, tmp_git_repo: Path) -> None:
        """Explicit host/port overrides harness.yaml defaults."""
        sd = SessionDir.init(
            "exp-4",
            curriculum="curricula/topic.md",
            host="10.0.0.1",
            port=9999,
            root=tmp_git_repo,
        )
        assert sd.config.harness_url == "ws://10.0.0.1:9999"

    def test_load_round_trip(self, tmp_git_repo: Path) -> None:
        """init then load produces identical SessionConfig."""
        sd_init = SessionDir.init(
            "exp-5",
            curriculum="curricula/topic.md",
            root=tmp_git_repo,
        )

        # Load from main repo (via .worktrees convention)
        sd_load = SessionDir.load("exp-5", root=tmp_git_repo)
        assert sd_load.config == sd_init.config

        # Load from worktree directly
        worktree_path = tmp_git_repo / ".worktrees" / "auto-tune" / "exp-5"
        sd_load_wt = SessionDir.load("exp-5", root=worktree_path)
        assert sd_load_wt.config == sd_init.config

    def test_events_jsonl_starts_empty(self, tmp_git_repo: Path) -> None:
        """events.jsonl is created empty."""
        sd = SessionDir.init(
            "exp-6",
            curriculum="curricula/topic.md",
            root=tmp_git_repo,
        )
        assert sd.events_path.read_text(encoding="utf-8") == ""

    def test_init_with_custom_base_dir(self, tmp_git_repo: Path) -> None:
        """init respects custom base_dir parameter."""
        sd = SessionDir.init(
            "exp-7",
            curriculum="curricula/topic.md",
            root=tmp_git_repo,
            base_dir="tuning",
        )
        worktree_path = tmp_git_repo / ".worktrees" / "tuning" / "exp-7"
        assert (worktree_path / "tuning" / "exp-7" / "config.json").is_file()

        # Worktree should be on the custom branch
        worktree_current = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert worktree_current == "tuning/exp-7"

    def test_init_reads_harness_yaml(self, tmp_git_repo: Path) -> None:
        """init reads host/port from harness.yaml when no overrides given."""
        harness_yaml = tmp_git_repo / "harness.yaml"
        harness_yaml.write_text(
            "server:\n  host: 'myhost'\n  port: 5555\n",
            encoding="utf-8",
        )
        sd = SessionDir.init(
            "exp-8",
            curriculum="curricula/topic.md",
            root=tmp_git_repo,
        )
        assert sd.config.harness_url == "ws://myhost:5555"

    def test_teardown_removes_worktree(self, tmp_git_repo: Path) -> None:
        """teardown removes the worktree and branch."""
        SessionDir.init(
            "exp-td",
            curriculum="curricula/topic.md",
            root=tmp_git_repo,
        )
        worktree_path = tmp_git_repo / ".worktrees" / "auto-tune" / "exp-td"
        assert worktree_path.exists()

        SessionDir.teardown("exp-td", root=tmp_git_repo)
        assert not worktree_path.exists()

        # Branch should be gone
        branches = subprocess.run(
            ["git", "branch", "--list", "auto-tune/exp-td"],
            cwd=tmp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert "auto-tune/exp-td" not in branches

    def test_init_creates_worktree_path_in_config(self, tmp_git_repo: Path) -> None:
        """config.json records the worktree path."""
        sd = SessionDir.init(
            "exp-wt",
            curriculum="curricula/topic.md",
            root=tmp_git_repo,
        )
        worktree_path = tmp_git_repo / ".worktrees" / "auto-tune" / "exp-wt"
        assert sd.config.worktree_path == str(worktree_path)
