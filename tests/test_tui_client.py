"""Tests for HarnessClient, TUIApp, and TUI regions.

Uses a stub WebSocket server to avoid requiring a running harness.
Widget tests use DOM mocking patterns following test_toolbar_progress.py.

Test mapping:
  - HarnessClient: registration, send, receive, disconnect
  - EventLog: displays events
  - RatifyBar: button state management
  - InputBar: Submitted message, clear method
  - TUIApp: countersign integration, event polling
  - HRNS-25: test_renders_received_events — EventLog renders all incoming harness frames
  - HRNS-26: test_sends_freeform_input_to_trainer — InputBar dispatches freeform input to trainer
  - HRNS-27: test_sends_countersign_on_ratify — RatifyBar sends countersign to kalvin (ctrl+r)
  - HRNS-28: test_input_bar_clears_after_send — InputBar clears text field after submission
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets

from participants.tui_client import HarnessClient, TUIApp
from participants.tui_regions import EventLog, EventItem, InputBar, RatifyBar


# ---------------------------------------------------------------------------
# Stub harness WebSocket server
# ---------------------------------------------------------------------------


class StubHarness:
    """Minimal WebSocket server for testing.

    Records received frames and allows sending frames to the client.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._port = port
        self._server: websockets.asyncio.server.Server | None = None
        self._client_ws: websockets.asyncio.server.ServerConnection | None = None
        self.received_frames: list[dict[str, Any]] = []
        self._connected = asyncio.Event()

    @property
    def url(self) -> str:
        port = self._server.sockets[0].getsockname()[1] if self._server else 0
        return f"ws://{self._host}:{port}"

    async def __aenter__(self) -> StubHarness:
        self._server = await websockets.serve(
            self._handle, self._host, self._port
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client_ws is not None:
            await self._client_ws.close()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws: websockets.asyncio.server.ServerConnection) -> None:
        """Handle a single client connection."""
        self._client_ws = ws
        self._connected.set()
        try:
            async for raw in ws:
                frame = json.loads(raw)
                self.received_frames.append(frame)
        except websockets.ConnectionClosed:
            pass

    async def send_to_client(self, frame: dict[str, Any]) -> None:
        """Send a JSON frame to the connected client."""
        await self._connected.wait()
        assert self._client_ws is not None
        await self._client_ws.send(json.dumps(frame))

    async def wait_for_frames(self, n: int, timeout: float = 2.0) -> None:
        """Wait until *n* frames have been received."""
        deadline = asyncio.get_event_loop().time() + timeout
        while len(self.received_frames) < n:
            await asyncio.sleep(0.05)
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(
                    f"Expected {n} frames, got {len(self.received_frames)}"
                )


# ═══════════════════════════════════════════════════════════════════════
# HarnessClient tests
# ═══════════════════════════════════════════════════════════════════════


async def test_tui_registers_on_connect():
    """HarnessClient sends ``{"register": "ui"}`` on connect."""
    async with StubHarness() as stub:
        client = HarnessClient(stub.url)
        await client.connect()

        await stub.wait_for_frames(1)
        assert len(stub.received_frames) >= 1
        reg = stub.received_frames[0]
        assert reg == {"register": "ui"}

        await client.disconnect()


async def test_tui_sends_message():
    """``client.send()`` produces a correct JSON frame on the server side."""
    async with StubHarness() as stub:
        client = HarnessClient(stub.url)
        await client.connect()
        await stub.wait_for_frames(1)  # registration frame

        await client.send("kalvin", "countersign", {"sig": 42})
        await stub.wait_for_frames(2)

        msg = stub.received_frames[1]
        assert msg == {
            "address": "kalvin",
            "action": "countersign",
            "message": {"sig": 42},
        }

        await client.disconnect()


async def test_tui_receives_message():
    """Server-sent frames appear in ``client.receive()``."""
    async with StubHarness() as stub:
        client = HarnessClient(stub.url)
        await client.connect()
        await stub.wait_for_frames(1)

        event_data = {"address": "ui", "action": "event", "message": {"kind": "frame"}}
        await stub.send_to_client(event_data)

        # Poll until we get a message
        received = None
        for _ in range(20):
            received = await client.receive()
            if received is not None:
                break
            await asyncio.sleep(0.05)

        assert received is not None
        assert received["action"] == "event"
        assert received["message"] == {"kind": "frame"}

        await client.disconnect()


async def test_tui_disconnect():
    """``client.disconnect()`` closes the WebSocket cleanly."""
    async with StubHarness() as stub:
        client = HarnessClient(stub.url)
        await client.connect()
        assert client.connected is True

        await client.disconnect()
        assert client.connected is False


# ═══════════════════════════════════════════════════════════════════════
# TUI Region widget tests (headless — no Textual app loop)
# ═══════════════════════════════════════════════════════════════════════


def test_event_log_stores_events():
    """EventLog.add_event stores frames and they are retrievable."""
    log = EventLog()
    frame = {"address": "ui", "action": "event", "message": {"kind": "ground"}}
    log.add_event(frame)
    assert len(log.events) == 1
    assert log.events[0] == frame


def test_event_log_displays_multiple_events():
    """Multiple events are stored in order."""
    log = EventLog()
    for i in range(3):
        log.add_event({"action": f"action-{i}", "message": f"msg-{i}"})
    assert len(log.events) == 3
    assert log.events[0]["action"] == "action-0"
    assert log.events[2]["action"] == "action-2"


def test_ratify_button_disabled_by_default():
    """RatifyBar starts with the button disabled."""
    bar = RatifyBar()
    # Access internal state before mount (no DOM)
    assert bar._selected_event_data is None


def test_ratify_button_enabled_on_selection():
    """enable_ratify stores event data."""
    bar = RatifyBar()
    bar.enable_ratify({"sig": 42, "nodes": [1, 2]})
    assert bar._selected_event_data == {"sig": 42, "nodes": [1, 2]}


def test_ratify_button_disabled_after_disable():
    """disable_ratify clears selection."""
    bar = RatifyBar()
    bar.enable_ratify({"sig": 42})
    bar.disable_ratify()
    assert bar._selected_event_data is None


# ═══════════════════════════════════════════════════════════════════════
# InputBar widget tests (headless — no Textual app loop)
# ═══════════════════════════════════════════════════════════════════════


def test_input_bar_imports():
    """InputBar is importable from participants.tui_regions."""
    assert InputBar is not None
    assert hasattr(InputBar, "Submitted")


def test_input_bar_submitted_message_carries_text():
    """InputBar.Submitted stores the submitted text."""
    msg = InputBar.Submitted("hello")
    assert msg.text == "hello"


def test_input_bar_submitted_message_empty_string():
    """InputBar.Submitted can carry an empty string."""
    msg = InputBar.Submitted("")
    assert msg.text == ""


def test_input_bar_has_clear_method():
    """InputBar instance has a clear() method."""
    bar = InputBar()
    assert hasattr(bar, "clear")
    assert callable(bar.clear)


# ═══════════════════════════════════════════════════════════════════════
# TUIApp integration tests
# ═══════════════════════════════════════════════════════════════════════


async def test_tuiapp_ratify_sends_countersign():
    """TUIApp's ratify handler sends countersign via HarnessClient.

    Verifies that on ratify click, the app sends
    ``{address: "kalvin", action: "countersign", message: <event_data>}``
    through the HarnessClient.
    """
    async with StubHarness() as stub:
        app = TUIApp(harness_url=stub.url)

        # Mount the app
        async with app.run_test() as pilot:
            # Wait for client to connect and register
            await stub.wait_for_frames(1)
            reg = stub.received_frames[0]
            assert reg == {"register": "ui"}

            # Simulate: enable ratify with event data
            event_data = {"signature": 42, "nodes": [10, 20]}
            ratify_bar = app.query_one(RatifyBar)
            ratify_bar.enable_ratify(event_data)

            # Trigger ratify by clicking the button
            await pilot.click("#ratify-btn")
            await asyncio.sleep(0.2)

            # Wait for the countersign message to arrive at the stub
            await stub.wait_for_frames(2, timeout=3.0)

            msg = stub.received_frames[1]
            assert msg["address"] == "kalvin"
            assert msg["action"] == "countersign"
            assert msg["message"] == {"signature": 42, "nodes": [10, 20]}


async def test_tuiapp_event_polling_populates_log():
    """TUIApp's event polling receives harness events and displays in EventLog.

    Verifies that a message sent from the server appears in the EventLog.
    """
    async with StubHarness() as stub:
        app = TUIApp(harness_url=stub.url)

        async with app.run_test() as pilot:
            await stub.wait_for_frames(1)

            # Server sends an event
            await stub.send_to_client({
                "address": "ui",
                "action": "event",
                "message": {"kind": "frame", "sig": "0xABCD"},
            })

            # Wait for event polling to pick it up
            await asyncio.sleep(0.5)

            event_log = app.query_one(EventLog)
            assert len(event_log.events) >= 1
            assert event_log.events[0]["action"] == "event"
            assert event_log.events[0]["message"]["kind"] == "frame"


# ═══════════════════════════════════════════════════════════════════════
# InputBar headless widget tests
# ═══════════════════════════════════════════════════════════════════════


async def test_input_bar_submitted_clears_input():
    """InputBar clears the input field after submission."""
    from textual.app import App, ComposeResult

    messages: list[InputBar.Submitted] = []

    class _TestApp(App):
        def compose(self) -> ComposeResult:
            yield InputBar()

        def on_input_bar_submitted(self, event: InputBar.Submitted) -> None:
            messages.append(event)

    async with _TestApp().run_test() as pilot:
        input_field = pilot.app.query_one("#input-bar-field")
        input_field.value = "hello trainer"
        await pilot.press("enter")
        await asyncio.sleep(0.1)

        # Input should be cleared
        assert input_field.value == ""
        # Message should have been posted
        assert len(messages) == 1
        assert messages[0].text == "hello trainer"


async def test_input_bar_ignores_empty_input():
    """InputBar does not post Submitted for empty/whitespace input."""
    from textual.app import App, ComposeResult

    messages: list[InputBar.Submitted] = []

    class _TestApp(App):
        def compose(self) -> ComposeResult:
            yield InputBar()

        def on_input_bar_submitted(self, event: InputBar.Submitted) -> None:
            messages.append(event)

    async with _TestApp().run_test() as pilot:
        input_field = pilot.app.query_one("#input-bar-field")

        # Try submitting empty
        input_field.value = ""
        await pilot.press("enter")
        await asyncio.sleep(0.1)
        assert len(messages) == 0

        # Try submitting whitespace only
        input_field.value = "   "
        await pilot.press("enter")
        await asyncio.sleep(0.1)
        assert len(messages) == 0


# ═══════════════════════════════════════════════════════════════════════
# InputBar + TUIApp integration tests
# ═══════════════════════════════════════════════════════════════════════


async def test_tuiapp_input_sends_to_trainer():
    """TUIApp's InputBar submission sends text to the Trainer.

    Verifies that submitting text via the input bar sends
    ``{address: "trainer", action: "input", message: <text>}``
    through the HarnessClient.
    """
    async with StubHarness() as stub:
        app = TUIApp(harness_url=stub.url)

        async with app.run_test() as pilot:
            await stub.wait_for_frames(1)

            # Type text into the input bar and submit
            input_field = app.query_one("#input-bar-field")
            input_field.focus()
            input_field.value = "hello from human"
            await pilot.press("enter")
            await asyncio.sleep(0.2)

            # Wait for the input message to arrive at the stub
            await stub.wait_for_frames(2, timeout=3.0)

            msg = stub.received_frames[1]
            assert msg["address"] == "trainer"
            assert msg["action"] == "input"
            assert msg["message"] == "hello from human"


async def test_tuiapp_ctrl_s_sends_input():
    """ctrl+s focuses the input bar and submits if text is present."""
    async with StubHarness() as stub:
        app = TUIApp(harness_url=stub.url)

        async with app.run_test() as pilot:
            await stub.wait_for_frames(1)

            # Put text into the input field first
            input_field = app.query_one("#input-bar-field")
            input_field.value = "ctrl+s message"

            # Press ctrl+s to submit
            await pilot.press("ctrl+s")
            await asyncio.sleep(0.2)

            # Wait for the input message to arrive at the stub
            await stub.wait_for_frames(2, timeout=3.0)

            msg = stub.received_frames[1]
            assert msg["address"] == "trainer"
            assert msg["action"] == "input"
            assert msg["message"] == "ctrl+s message"

            # Input should be cleared
            assert input_field.value == ""


# ═══════════════════════════════════════════════════════════════════════
# HRNS-25..28 TUI Participant capability tests
# ═══════════════════════════════════════════════════════════════════════


async def test_renders_received_events():
    """HRNS-25: EventLog renders all received harness events in order.

    The server sends multiple frames of different action types (event,
    notify, proposal). All frames must appear in EventLog.events with
    correct action and message fields, in order, with none dropped.
    """
    async with StubHarness() as stub:
        app = TUIApp(harness_url=stub.url)

        async with app.run_test() as pilot:
            await stub.wait_for_frames(1)  # registration frame

            # Server sends three frames of different action types
            frames = [
                {"address": "ui", "action": "event", "message": {"kind": "frame"}},
                {"address": "ui", "action": "notify", "message": "session started"},
                {"address": "ui", "action": "proposal", "message": {"sig": 42, "nodes": [1, 2]}},
            ]
            for frame in frames:
                await stub.send_to_client(frame)

            # Wait for event polling to pick up all frames
            await asyncio.sleep(0.8)

            event_log = app.query_one(EventLog)
            assert len(event_log.events) >= 3

            # Verify each frame is stored with correct action and message
            stored = event_log.events
            assert stored[0]["action"] == "event"
            assert stored[0]["message"] == {"kind": "frame"}

            assert stored[1]["action"] == "notify"
            assert stored[1]["message"] == "session started"

            assert stored[2]["action"] == "proposal"
            assert stored[2]["message"] == {"sig": 42, "nodes": [1, 2]}


async def test_sends_freeform_input_to_trainer():
    """HRNS-26: InputBar dispatches freeform human input to the Trainer.

    Simulates typing text and pressing Enter in the InputBar.
    Verifies the stub server receives
    ``{address: "trainer", action: "input", message: <typed_text>}``.
    """
    async with StubHarness() as stub:
        app = TUIApp(harness_url=stub.url)

        async with app.run_test() as pilot:
            await stub.wait_for_frames(1)  # registration frame

            input_field = app.query_one("#input-bar-field")
            input_field.focus()
            input_field.value = "what is the capital of France?"
            await pilot.press("enter")
            await asyncio.sleep(0.2)

            await stub.wait_for_frames(2, timeout=3.0)

            msg = stub.received_frames[1]
            assert msg["address"] == "trainer"
            assert msg["action"] == "input"
            assert msg["message"] == "what is the capital of France?"


async def test_sends_countersign_on_ratify():
    """HRNS-27: RatifyBar sends countersign to kalvin on ctrl+r.

    Enables ratify with event data, then presses ctrl+r to trigger
    the ratify keyboard shortcut. Verifies the stub server receives
    ``{address: "kalvin", action: "countersign", message: <event_data>}``.
    """
    async with StubHarness() as stub:
        app = TUIApp(harness_url=stub.url)

        async with app.run_test() as pilot:
            await stub.wait_for_frames(1)  # registration frame

            # Enable ratify with event data
            event_data = {"sig": 99, "nodes": [5, 10]}
            ratify_bar = app.query_one(RatifyBar)
            ratify_bar.enable_ratify(event_data)

            # Trigger ratify via ctrl+r keyboard shortcut
            await pilot.press("ctrl+r")
            await asyncio.sleep(0.2)

            await stub.wait_for_frames(2, timeout=3.0)

            msg = stub.received_frames[1]
            assert msg["address"] == "kalvin"
            assert msg["action"] == "countersign"
            assert msg["message"] == {"sig": 99, "nodes": [5, 10]}


async def test_input_bar_clears_after_send():
    """HRNS-28: InputBar clears its text field after submission.

    Types text into the InputBar, triggers submission, and verifies
    the InputBar's text value is empty after the send completes.
    """
    async with StubHarness() as stub:
        app = TUIApp(harness_url=stub.url)

        async with app.run_test() as pilot:
            await stub.wait_for_frames(1)  # registration frame

            input_field = app.query_one("#input-bar-field")
            input_field.focus()
            input_field.value = "clear me after send"
            await pilot.press("enter")
            await asyncio.sleep(0.2)

            # Verify the input field is cleared
            assert input_field.value == ""

            # Verify the message was actually sent (not just cleared)
            await stub.wait_for_frames(2, timeout=3.0)
            msg = stub.received_frames[1]
            assert msg["action"] == "input"
            assert msg["message"] == "clear me after send"
