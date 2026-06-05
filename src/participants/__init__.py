"""Harness WebSocket client participants.

Two client participants that connect to the harness server:

- **SlackParticipant** — bridges Slack API and the harness message bus.
  Registers as the supervisor role and renders all supervisor actions
  (progress, event, escalation, ratify_request) to Slack. Forwards supervisor
  commands through the shared command parser to the appropriate harness role.

- **TUIParticipant** (``TUIApp`` / ``HarnessClient``) — a Textual TUI that
  displays KAgent events and provides ratification (countersign) controls.

Both participants register on connect via the WebSocket wire protocol:
``{"register": "<role>"}`` followed by bidirectional JSON message frames.

Running
-------

**Slack Participant:**

    Set environment variables ``SLACK_BOT_TOKEN`` and ``SLACK_APP_TOKEN``,
    then connect to the harness::

        from participants import SlackParticipant

        agent = SlackParticipant(
            harness_url="ws://localhost:8765",
            channel_id="C01234567",
        )
        await agent.start()

**TUI Participant:**

    Launch the Textual TUI app::

        from participants import TUIApp

        app = TUIApp(harness_url="ws://localhost:8765")
        app.run()
"""


def __getattr__(name: str):
    """Lazy imports to avoid import errors when modules are partially built."""
    if name == "SlackParticipant":
        from participants.slack_agent import SlackParticipant
        return SlackParticipant
    if name in ("HarnessClient", "TUIApp"):
        from participants.tui_client import HarnessClient, TUIApp
        return HarnessClient if name == "HarnessClient" else TUIApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["HarnessClient", "SlackParticipant", "TUIApp"]
