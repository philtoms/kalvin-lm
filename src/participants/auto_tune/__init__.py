"""Auto-tune session management and CLI supervisor.

Provides session initialisation, configuration, and directory management
for automated training sessions driven by an LLM coding agent.

Exports:
    SessionConfig — serialisable configuration for an auto-tune session.
    SessionDir — manages session directories, config files, and git branches.
"""


def __getattr__(name: str):
    """Lazy imports matching the participants package convention."""
    if name == "SessionConfig":
        from participants.auto_tune.session import SessionConfig
        return SessionConfig
    if name == "SessionDir":
        from participants.auto_tune.session import SessionDir
        return SessionDir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SessionConfig", "SessionDir"]
