"""Auto-tune session management and process lifecycle.

Provides session initialisation, configuration, directory management,
and process lifecycle commands for automated training sessions driven
by an LLM coding agent.

Exports:
    SessionConfig — serialisable configuration for an auto-tune session.
    SessionDir — manages session directories, config files, and git branches.
    start_harness — start the harness server as a background process.
    stop_harness — stop the harness server (SIGTERM → SIGKILL).
    start_supervisor — start the CLI supervisor as a background process.
    stop_supervisor — stop the CLI supervisor (shutdown cmd → SIGKILL).
    reset — clear auto-tune session state for a fresh start.
"""


def __getattr__(name: str):
    """Lazy imports matching the participants package convention."""
    if name == "SessionConfig":
        from participants.auto_tune.session import SessionConfig
        return SessionConfig
    if name == "SessionDir":
        from participants.auto_tune.session import SessionDir
        return SessionDir
    if name in ("start_harness", "stop_harness", "start_supervisor", "stop_supervisor"):
        from participants.auto_tune import lifecycle as _lifecycle
        return getattr(_lifecycle, name)
    if name == "reset":
        from participants.auto_tune.snapshots import reset
        return reset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SessionConfig",
    "SessionDir",
    "start_harness",
    "stop_harness",
    "start_supervisor",
    "stop_supervisor",
    "reset",
]
