"""Auto-tune session management and process lifecycle.

Provides session initialisation, configuration, directory management,
orchestration functions, snapshot/restore, and process lifecycle commands
for automated training sessions driven by an LLM coding agent.

Exports:
    SessionConfig — serialisable configuration for an auto-tune session.
    SessionDir — manages session directories, config files, and git branches.
    CLISupervisor — headless WebSocket client for auto-tune sessions.
"""


def __getattr__(name: str):
    """Lazy imports matching the participants package convention."""
    if name == "SessionConfig":
        from participants.auto_tune.session import SessionConfig

        return SessionConfig
    if name == "SessionDir":
        from participants.auto_tune.session import SessionDir

        return SessionDir
    if name == "CLISupervisor":
        from participants.auto_tune.supervisor import CLISupervisor

        return CLISupervisor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CLISupervisor", "SessionConfig", "SessionDir"]
