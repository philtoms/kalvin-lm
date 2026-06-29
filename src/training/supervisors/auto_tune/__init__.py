"""Auto-tune session management and process lifecycle.

Provides session initialisation, configuration, directory management,
orchestration functions, snapshot/restore, and process lifecycle commands
for automated training sessions driven by an LLM coding agent.

Auto-tune is the codebase-tuning loop (session/worktree/snapshot orchestration).
The CLI supervisor participant that auto-tune launches lives in
``training.supervisors.cli_supervisor`` (a peer of the TUI/Slack/LLM
supervisors) — it is not part of this package.

Exports:
    SessionConfig — serialisable configuration for an auto-tune session.
    SessionDir — manages session directories, config files, and git branches.
"""


def __getattr__(name: str):
    """Lazy imports matching the supervisors package convention."""
    if name == "SessionConfig":
        from training.supervisors.auto_tune.session import SessionConfig

        return SessionConfig
    if name == "SessionDir":
        from training.supervisors.auto_tune.session import SessionDir

        return SessionDir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SessionConfig", "SessionDir"]
