"""Tests for auto-tune reset command.

Spec ref: specs/auto-tune.md §Reset (rules 39–42)
Test IDs: AT-18, AT-19
"""

import json
from pathlib import Path

import pytest

from participants.auto_tune.session import SessionConfig, SessionDir
from participants.auto_tune.snapshots import reset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_dir(
    tmp_path: Path,
    *,
    curriculum: str | None = None,
    model_path: str | None = None,
) -> tuple[SessionDir, dict[str, Path]]:
    """Create a minimal session directory tree and return a bound SessionDir.

    Returns (session_dir, paths) where *paths* maps role names to the absolute
    Path of each file the caller may want to create/inspect.

    Paths stored in config are absolute so that ``Path(p).with_suffix()`` works
    regardless of the process CWD.
    """
    session_name = "test-session"
    session_dir = tmp_path / "auto-tune" / session_name
    session_dir.mkdir(parents=True)
    (session_dir / "runs").mkdir()

    # Defaults
    if curriculum is None:
        curriculum = str(tmp_path / "curricula" / "test.md")
    if model_path is None:
        model_path = str(tmp_path / "data" / "agent.bin")

    config = SessionConfig(
        session=session_name,
        curriculum=curriculum,
        harness_url="ws://localhost:8765",
        model_path=model_path,
        run_counter=3,
        created_from_branch="main",
        created_from_commit="abc123",
    )
    config_path = session_dir / "config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    sd = SessionDir(
        root=tmp_path,
        base_dir="auto-tune",
        _session=session_name,
        _config=config,
    )

    paths = {
        "curriculum_md": Path(curriculum),
        "state_file": Path(curriculum).with_suffix(".json"),
        "model_file": Path(model_path) if model_path else None,
    }
    return sd, paths


# ---------------------------------------------------------------------------
# AT-18: reset deletes state file and truncates events
# ---------------------------------------------------------------------------


class TestResetBasic:
    """AT-18 — reset deletes curriculum state file and truncates events.jsonl."""

    def test_state_file_deleted(self, tmp_path: Path) -> None:
        sd, paths = _make_session_dir(tmp_path)

        # Create curriculum markdown and its state file
        paths["curriculum_md"].parent.mkdir(parents=True, exist_ok=True)
        paths["curriculum_md"].write_text("# Curriculum", encoding="utf-8")
        paths["state_file"].write_text('{"progress": 0.5}', encoding="utf-8")

        assert paths["state_file"].exists()
        reset(sd)
        assert not paths["state_file"].exists()

    def test_events_truncated(self, tmp_path: Path) -> None:
        sd, _ = _make_session_dir(tmp_path)
        events = sd.events_path
        events.write_text(
            '{"seq":1,"type":"connected"}\n{"seq":2,"type":"progress"}\n',
            encoding="utf-8",
        )
        assert events.stat().st_size > 0

        reset(sd)

        assert events.exists()
        assert events.stat().st_size == 0

    def test_config_untouched(self, tmp_path: Path) -> None:
        sd, _ = _make_session_dir(tmp_path)
        original = sd.config_path.read_text(encoding="utf-8")

        sd.events_path.write_text("some content\n", encoding="utf-8")

        reset(sd)

        assert sd.config_path.read_text(encoding="utf-8") == original

    def test_runs_directory_untouched(self, tmp_path: Path) -> None:
        sd, _ = _make_session_dir(tmp_path)

        # Create a run snapshot
        run_dir = sd.runs_dir / "001"
        run_dir.mkdir()
        (run_dir / "meta.json").write_text('{"run": 1}', encoding="utf-8")
        (run_dir / "state.json").write_text('{"progress": 0.3}', encoding="utf-8")

        sd.events_path.write_text("event\n", encoding="utf-8")

        reset(sd)

        assert (run_dir / "meta.json").exists()
        assert (run_dir / "state.json").exists()
        assert (run_dir / "meta.json").read_text(encoding="utf-8") == '{"run": 1}'
        assert sd.runs_dir.exists()


# ---------------------------------------------------------------------------
# AT-19: reset --fresh-model also deletes model file
# ---------------------------------------------------------------------------


class TestResetFreshModel:
    """AT-19 — reset with fresh_model=True also deletes Kalvin model file."""

    def test_model_deleted_with_fresh_model(self, tmp_path: Path) -> None:
        sd, paths = _make_session_dir(tmp_path)

        model_file = paths["model_file"]
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_bytes(b"\x00\x01\x02model-data")

        sd.events_path.write_text("event\n", encoding="utf-8")

        reset(sd, fresh_model=True)

        assert not model_file.exists()

    def test_state_and_events_also_cleared(self, tmp_path: Path) -> None:
        """Fresh model doesn't skip the regular reset steps."""
        sd, paths = _make_session_dir(tmp_path)

        # Create curriculum and state file
        paths["curriculum_md"].parent.mkdir(parents=True, exist_ok=True)
        paths["curriculum_md"].write_text("# C", encoding="utf-8")
        paths["state_file"].write_text("{}", encoding="utf-8")

        # Create model file
        model_file = paths["model_file"]
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_bytes(b"model")

        sd.events_path.write_text("line\n", encoding="utf-8")

        reset(sd, fresh_model=True)

        assert not paths["state_file"].exists()
        assert not model_file.exists()
        assert sd.events_path.exists()
        assert sd.events_path.stat().st_size == 0

    def test_model_not_deleted_without_fresh_model(self, tmp_path: Path) -> None:
        sd, paths = _make_session_dir(tmp_path)

        model_file = paths["model_file"]
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_bytes(b"model")

        sd.events_path.write_text("event\n", encoding="utf-8")

        reset(sd, fresh_model=False)

        assert model_file.exists()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestResetIdempotency:
    """Calling reset() twice does not raise when files are already gone."""

    def test_double_reset_no_error(self, tmp_path: Path) -> None:
        sd, _ = _make_session_dir(tmp_path)
        sd.events_path.write_text("event\n", encoding="utf-8")

        reset(sd)
        reset(sd)  # second call — files already gone

        assert sd.events_path.exists()
        assert sd.events_path.stat().st_size == 0
