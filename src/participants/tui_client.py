"""TUI Participant — WebSocket client and Textual app for the multi-agent harness.

Provides:

- **HarnessClient**: async WebSocket client that handles registration
  (``{"register": "ui"}``) and bidirectional JSON message send/receive via
  asyncio queues.
- **TUIApp**: Textual application that renders KAgent events and provides
  ratification (countersign) controls.

Spec reference: specs/harness-server.md §TUI Participant
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets

logger = logging.getLogger(__name__)


class HarnessClient:
    """Async WebSocket client for connecting to the harness server.

    Handles registration (``{"register": "ui"}``) and provides queue-based
    bidirectional JSON message send/receive.

    Parameters
    ----------
    url:
        WebSocket URL of the harness server (e.g. ``"ws://localhost:8765"``).
    address:
        Registration address (default ``"ui"``).
    """

    def __init__(self, url: str, address: str = "ui") -> None:
        self._url = url
        self._address = address
        self._ws: websockets.asyncio.client.ClientConnection | None = None
        self._send_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._receive_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._read_task: asyncio.Task[None] | None = None
        self._write_task: asyncio.Task[None] | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        """Whether the client is currently connected."""
        return self._connected

    # -- public API ----------------------------------------------------------

    async def connect(self) -> None:
        """Open WebSocket, send registration frame, start read/write loops."""
        self._ws = await websockets.connect(self._url)

        # Send registration frame
        await self._ws.send(json.dumps({"register": self._address}))
        logger.info("HarnessClient registered as %r", self._address)

        self._connected = True

        # Start background tasks
        self._read_task = asyncio.create_task(self._read_loop())
        self._write_task = asyncio.create_task(self._write_loop())

    async def send(self, address: str, action: str, message: Any) -> None:
        """Enqueue an outgoing JSON message to be sent to the harness.

        Parameters
        ----------
        address:
            Recipient address.
        action:
            Verb interpreted by the recipient.
        message:
            Arbitrary payload.
        """
        frame = {"address": address, "action": action, "message": message}
        await self._send_queue.put(frame)

    async def receive(self) -> dict[str, Any] | None:
        """Return the next incoming message from the receive queue.

        Returns ``None`` if the client is disconnected and the queue is empty.
        """
        try:
            return self._receive_queue.get_nowait()
        except asyncio.QueueEmpty:
            if not self._connected:
                return None
            # Wait briefly for a message
            try:
                return await asyncio.wait_for(
                    self._receive_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                return None

    async def disconnect(self) -> None:
        """Close the WebSocket and cancel background tasks."""
        self._connected = False

        if self._read_task is not None:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None

        if self._write_task is not None:
            self._write_task.cancel()
            try:
                await self._write_task
            except asyncio.CancelledError:
                pass
            self._write_task = None

        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    # -- background loops ----------------------------------------------------

    async def _read_loop(self) -> None:
        """Pull WebSocket frames and push to the receive queue."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    frame = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Malformed frame from harness: %s", raw[:200])
                    continue
                await self._receive_queue.put(frame)
        except websockets.ConnectionClosed:
            logger.info("Harness WebSocket connection closed")
        except asyncio.CancelledError:
            pass
        finally:
            self._connected = False

    async def _write_loop(self) -> None:
        """Pull from the send queue and write to the WebSocket."""
        assert self._ws is not None
        try:
            while True:
                frame = await self._send_queue.get()
                raw = json.dumps(frame)
                await self._ws.send(raw)
        except websockets.ConnectionClosed:
            logger.info("Harness WebSocket closed during write")
        except asyncio.CancelledError:
            pass
        finally:
            self._connected = False


# ---------------------------------------------------------------------------
# TUIApp — Textual application for the TUI harness participant
# ---------------------------------------------------------------------------


from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input

from participants.tui_regions import EventLog, InputBar, RatifyBar


class TUIApp(App):
    """Textual TUI for the harness participant.

    Displays KAgent events routed from the harness via ``EventLog`` and
    provides ratification (countersign) controls via ``RatifyBar``.

    On mount: creates a ``HarnessClient``, connects to the harness, and
    starts a background task to poll ``client.receive()`` and push events
    to the ``EventLog``.
    """

    CSS = """
    Screen {
        layout: vertical;
    }
    """

    TITLE = "Kalvin Harness TUI"

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+r", "ratify", "Ratify"),
        ("ctrl+s", "send_input", "Send Input"),
    ]

    def __init__(
        self,
        harness_url: str = "ws://localhost:8765",
        address: str = "ui",
    ) -> None:
        super().__init__()
        self._harness_url = harness_url
        self._address = address
        self._client = HarnessClient(harness_url, address)
        self._poll_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield EventLog()
        yield RatifyBar()
        yield InputBar()
        yield Footer()

    async def on_mount(self) -> None:
        """Connect to harness and start event polling."""
        await self._client.connect()
        self._poll_task = asyncio.create_task(self._poll_harness_events())

    async def on_unmount(self) -> None:
        """Disconnect from harness."""
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        await self._client.disconnect()

    async def _poll_harness_events(self) -> None:
        """Background task: poll ``client.receive()`` and push to EventLog.

        Stores the latest event's message payload for ratification.
        """
        event_log = self.query_one(EventLog)
        ratify_bar = self.query_one(RatifyBar)

        try:
            while self._client.connected:
                frame = await self._client.receive()
                if frame is not None:
                    event_log.add_event(frame)
                    # Auto-enable ratify for events that carry a message payload
                    if frame.get("message") is not None:
                        ratify_bar.enable_ratify(frame["message"])
                else:
                    await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass

    def on_ratify_bar_ratify_clicked(self, event: RatifyBar.RatifyClicked) -> None:
        """Handle Ratify button click — send countersign via HarnessClient.

        Sends ``{address: "kalvin", action: "countersign", message: <event_data>}``
        where ``event_data`` is the raw ``message`` payload from the selected
        harness event frame (a JSON-serializable value, not a KLine object).
        """
        # Schedule the async send as a background task
        asyncio.create_task(
            self._client.send("kalvin", "countersign", event.event_data)
        )
        # Disable ratify after sending
        ratify_bar = self.query_one(RatifyBar)
        ratify_bar.disable_ratify()

    def action_ratify(self) -> None:
        """Keyboard shortcut: trigger ratify via ctrl+r."""
        ratify_bar = self.query_one(RatifyBar)
        if ratify_bar._selected_event_data is not None:
            ratify_bar.on_button_pressed(
                type("FakeButton", (), {"button": type("Btn", (), {"id": "ratify-btn"})()})()
            )

    def on_input_bar_submitted(self, event: InputBar.Submitted) -> None:
        """Handle InputBar submission — send free-form text to the Trainer.

        Sends ``{address: "trainer", action: "input", message: <text>}``
        via the HarnessClient. The input field is cleared by InputBar
        automatically after submission (HRNS-28).
        """
        asyncio.create_task(
            self._client.send("trainer", "input", event.text)
        )

    def action_send_input(self) -> None:
        """Keyboard shortcut: focus the input bar and submit via ctrl+s.

        Focuses the input field so the user can start typing immediately.
        If the input already has text, submits it.
        """
        input_bar = self.query_one(InputBar)
        input_field = input_bar.query_one("#input-bar-field", Input)
        input_field.focus()
        if input_field.value.strip():
            input_bar._submit()


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765"
    app = TUIApp(harness_url=url)
    app.run()
