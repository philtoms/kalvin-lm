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
        session_dir = tmp_git_repo / "auto-tune" / "exp-1"

        assert sd.config_path.is_file(), "config.json must exist"
        assert sd.events_path.is_file(), "events.jsonl must exist"
        assert sd.runs_dir.is_dir(), "runs/ subdirectory must exist"

    def test_at2_creates_and_checks_out_branch(self, tmp_git_repo: Path) -> None:
        """AT-2: init creates and checks out auto-tune/<session> git branch."""
        SessionDir.init(
            "exp-2",
            curriculum="curricula/topic.md",
            root=tmp_git_repo,
        )
        current = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=tmp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert current == "auto-tune/exp-2"

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

        # Switch back to main branch so we can load without being on the session branch
        subprocess.run(
            ["git", "checkout", "-"],
            cwd=tmp_git_repo,
            check=True,
            capture_output=True,
        )

        sd_load = SessionDir.load("exp-5", root=tmp_git_repo)
        assert sd_load.config == sd_init.config

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
        assert (tmp_git_repo / "tuning" / "exp-7" / "config.json").is_file()
        current = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=tmp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert current == "tuning/exp-7"

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
