"""Session management for auto-tune.

Provides ``SessionConfig`` for serialisable session configuration and
``SessionDir`` for directory layout, git branch management, and config I/O.

Spec ref: specs/auto-tune.md §Session Configuration, §Session Initialisation
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# SessionConfig
# ---------------------------------------------------------------------------


@dataclass
class SessionConfig:
    """Serialisable configuration for an auto-tune session.

    Fields match the schema in specs/auto-tune.md §Session Configuration.
    """

    session: str
    curriculum: str
    harness_url: str
    model_path: str | None = None
    run_counter: int = 0
    created_from_branch: str = ""
    created_from_commit: str = ""

    # -- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this config."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionConfig:
        """Construct a SessionConfig from a dict (e.g. parsed JSON).

        Unknown keys are silently ignored so that forward-compatible configs
        load without error.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# SessionDir
# ---------------------------------------------------------------------------


class SessionDir:
    """Manages the directory layout and git branch for an auto-tune session.

    Usage::

        sd = SessionDir.init("exp-1", curriculum="curricula/topic.md")
        print(sd.config)
        print(sd.config_path)

        # Later:
        sd = SessionDir.load("exp-1")
    """

    def __init__(
        self,
        root: Path = Path("."),
        base_dir: str = "auto-tune",
        _session: str | None = None,
        _config: SessionConfig | None = None,
    ) -> None:
        self._root = Path(root)
        self._base_dir = base_dir
        self._session = _session
        self._config = _config

    # -- Path helpers --------------------------------------------------------

    def _dir(self, session: str | None = None) -> Path:
        """Return the session directory path."""
        name = session or self._session
        if name is None:
            raise ValueError("No session bound to this SessionDir")
        return self._root / self._base_dir / name

    @property
    def config_path(self) -> Path:
        """Path to ``config.json`` for the bound session."""
        return self._dir() / "config.json"

    @property
    def cmd_path(self) -> Path:
        """Path to ``cmd.json`` for the bound session."""
        return self._dir() / "cmd.json"

    @property
    def status_path(self) -> Path:
        """Path to ``status.json`` for the bound session."""
        return self._dir() / "status.json"

    @property
    def events_path(self) -> Path:
        """Path to ``events.jsonl`` for the bound session."""
        return self._dir() / "events.jsonl"

    @property
    def runs_dir(self) -> Path:
        """Path to ``runs/`` directory for the bound session."""
        return self._dir() / "runs"

    @property
    def config(self) -> SessionConfig:
        """The loaded ``SessionConfig`` for the bound session."""
        if self._config is None:
            raise ValueError("No config loaded — call init() or load() first")
        return self._config

    # -- Factory: init -------------------------------------------------------

    @classmethod
    def init(
        cls,
        session: str,
        curriculum: str,
        host: str | None = None,
        port: int | None = None,
        *,
        root: Path = Path("."),
        base_dir: str = "auto-tune",
    ) -> SessionDir:
        """Create a new auto-tune session.

        1. Reads ``harness.yaml`` for default host/port.
        2. Creates the session directory tree.
        3. Writes ``config.json``.
        4. Creates and checks out git branch ``auto-tune/<session>``.

        Args:
            session: Codename for the session.
            curriculum: Path to the curriculum markdown file.
            host: Override for harness server host.
            port: Override for harness server port.
            root: Project root directory (default ``"."``).
            base_dir: Prefix directory name (default ``"auto-tune"``).

        Returns:
            A ``SessionDir`` bound to the new session.
        """
        root = Path(root)

        # 1. Read harness.yaml for defaults
        resolved_host, resolved_port = _read_harness_defaults(root / "harness.yaml")
        if host is not None:
            resolved_host = host
        if port is not None:
            resolved_port = port

        # 2. Derive model_path
        model_path = "data/agent.bin"

        # 3. Capture git state
        created_from_commit = _git("rev-parse", "HEAD", cwd=root).strip()
        created_from_branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root).strip()

        # 4. Build config
        harness_url = f"ws://{resolved_host}:{resolved_port}"
        cfg = SessionConfig(
            session=session,
            curriculum=curriculum,
            harness_url=harness_url,
            model_path=model_path,
            run_counter=0,
            created_from_branch=created_from_branch,
            created_from_commit=created_from_commit,
        )

        # 5. Create directory structure
        session_dir = root / base_dir / session
        runs = session_dir / "runs"
        runs.mkdir(parents=True, exist_ok=True)

        # 6. Write config.json
        config_path = session_dir / "config.json"
        config_path.write_text(
            json.dumps(cfg.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

        # 7. Create empty events.jsonl
        events_path = session_dir / "events.jsonl"
        events_path.write_text("", encoding="utf-8")

        # 8. Create and checkout git branch
        branch_name = f"{base_dir}/{session}"
        _git("checkout", "-b", branch_name, cwd=root)

        return cls(
            root=root,
            base_dir=base_dir,
            _session=session,
            _config=cfg,
        )

    # -- Factory: load -------------------------------------------------------

    @classmethod
    def load(
        cls,
        session: str,
        *,
        root: Path = Path("."),
        base_dir: str = "auto-tune",
    ) -> SessionDir:
        """Load an existing auto-tune session from its ``config.json``.

        Args:
            session: Codename of the session to load.
            root: Project root directory (default ``"."``).
            base_dir: Prefix directory name (default ``"auto-tune"``).

        Returns:
            A ``SessionDir`` bound to the loaded session.
        """
        root = Path(root)
        config_path = root / base_dir / session / "config.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        cfg = SessionConfig.from_dict(data)

        return cls(
            root=root,
            base_dir=base_dir,
            _session=session,
            _config=cfg,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_harness_defaults(config_path: Path) -> tuple[str, int]:
    """Read host and port defaults from a harness YAML config.

    Falls back to ``"localhost"`` / ``8765`` when the file is missing or
    the keys are absent — matching ``src/harness/__main__.py`` behaviour.
    """
    if not config_path.exists():
        return "localhost", 8765
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return "localhost", 8765
    if not isinstance(data, dict):
        return "localhost", 8765
    server_cfg = data.get("server", {})
    host = server_cfg.get("host", "localhost")
    port = server_cfg.get("port", 8765)
    return host, int(port)


def _git(*args: str, cwd: Path) -> str:
    """Run a git subprocess and return stdout. Raises on failure."""
    env = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    # Remove HOME to avoid .gitconfig access issues in sandboxed environments
    env.pop("HOME", None)
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return result.stdout
