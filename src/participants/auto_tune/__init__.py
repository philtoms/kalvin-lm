"""Auto-tune session management and CLI supervisor.

Provides session initialisation, configuration, directory management,
and orchestration functions for automated training sessions driven by an
LLM coding agent.

Exports:
    SessionConfig — serialisable configuration for an auto-tune session.
    SessionDir — manages session directories, config files, and git branches.
    send_command — write a command to a session's cmd.json.
    read_events — read and filter events from a session's events.jsonl.
    step — send a command and block until a new event appears.
    read_status — read a session's status.json.
"""


def __getattr__(name: str):
    """Lazy imports matching the participants package convention."""
    if name == "SessionConfig":
        from participants.auto_tune.session import SessionConfig
        return SessionConfig
    if name == "SessionDir":
        from participants.auto_tune.session import SessionDir
        return SessionDir
    if name == "send_command":
        from participants.auto_tune.orchestrate import send_command
        return send_command
    if name == "read_events":
        from participants.auto_tune.orchestrate import read_events
        return read_events
    if name == "step":
        from participants.auto_tune.orchestrate import step
        return step
    if name == "read_status":
        from participants.auto_tune.orchestrate import read_status
        return read_status
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SessionConfig",
    "SessionDir",
    "send_command",
    "read_events",
    "step",
    "read_status",
]
