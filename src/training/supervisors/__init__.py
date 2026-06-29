"""Harness WebSocket client supervisors.

Four client supervisors that connect to the harness server, all registering
as the ``supervisor`` role and sharing one decision contract
(``@specs/supervisor-decision.md``) — they differ only in the process that
produces an answer:

- **SlackParticipant** — a human on Slack.
- **TUIParticipant** (``TUIApp`` / ``HarnessClient``) — a human on a Textual TUI.
- **CLISupervisor** — pi (an LLM coding agent) via the auto-tune file protocol
  (``events.jsonl`` out, ``cmd.json`` in). The protocol relay auto-tune
  launches; pi is the decider.
- **LLMSupervisor** — an LLM decider participant.

All register via ``{"register": "<role>"}`` followed by bidirectional JSON
frames.
"""


def __getattr__(name: str):
    """Lazy imports to avoid import errors when modules are partially built."""
    if name == "SlackParticipant":
        from training.supervisors.slack_agent import SlackParticipant

        return SlackParticipant
    if name in ("HarnessClient", "TUIApp"):
        from training.supervisors.tui_client import HarnessClient, TUIApp

        return HarnessClient if name == "HarnessClient" else TUIApp
    if name == "CLISupervisor":
        from training.supervisors.cli_supervisor import CLISupervisor

        return CLISupervisor
    if name == "LLMSupervisor":
        from training.supervisors.llm_supervisor import LLMSupervisor

        return LLMSupervisor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CLISupervisor",
    "HarnessClient",
    "LLMSupervisor",
    "SlackParticipant",
    "TUIApp",
]
