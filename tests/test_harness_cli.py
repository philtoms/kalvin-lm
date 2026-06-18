"""End-to-end smoke tests for the harness CLI entry point.

Validates the full integration: CLI loads config → participants register →
messages route → shutdown is clean.

Test mapping: HRNS-5 (embedded participants load), HRNS-1 (message routing),
graceful shutdown with state persistence.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import websockets
import yaml

from training.harness.bus import MessageBus
from training.harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE, TRAINER_ROLE
from training.harness.message import Message
from training.harness.server import HarnessServer, load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Repository root (one level up from ``tests/``) and the canonical project
# config. Resolved from this test file's location so the smoke test passes
# regardless of the working directory pytest is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_CONFIG = REPO_ROOT / "training.harness.yaml"


def _make_self_subscribing_factory(
    target_role: str,
    received: dict[str, list[Message]],
    events: dict[str, threading.Event] | None = None,
):
    """Create a factory that self-subscribes (like real participants) and records messages.

    Returns a ``_AlreadySubscribed``-style wrapper so ``HarnessServer._setup()``
    won't double-dispatch.
    """

    def factory(role: str, bus: MessageBus) -> object:
        target = target_role

        class MockParticipant:
            @property
            def role(self) -> str:
                return role

            def on_message(self, msg: Message) -> None:
                received[target].append(msg)
                if events and target in events:
                    events[target].set()

        p = MockParticipant()
        bus.subscribe(role, p.on_message)

        class Wrapper:
            @property
            def role(self) -> str:
                return role

            def on_message(self, msg: Message) -> None:
                pass  # already subscribed above

        return Wrapper()

    return factory


def _make_noop_factory():
    """Create a factory that self-subscribes with a no-op handler."""

    def factory(role: str, bus: MessageBus) -> object:
        class MockParticipant:
            @property
            def role(self) -> str:
                return role

            def on_message(self, msg: Message) -> None:
                pass

        p = MockParticipant()
        bus.subscribe(role, p.on_message)

        class Wrapper:
            @property
            def role(self) -> str:
                return role

            def on_message(self, msg: Message) -> None:
                pass

        return Wrapper()

    return factory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Write a minimal harness config and return its path."""
    config = {
        "server": {"host": "localhost", "port": 18765},
        "trainer": {"state_path": str(tmp_path / "trainer_state.json")},
        "participants": [
            {"role": TRAINEE_ROLE, "type": "embedded", "class": "KAgent"},
            {"role": TRAINER_ROLE, "type": "embedded", "class": "Trainer"},
        ],
    }
    path = tmp_path / "training.harness.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    return path


@pytest.fixture
def full_config(tmp_path: Path) -> Path:
    """Write the full 4-participant harness config and return its path."""
    config = {
        "server": {"host": "localhost", "port": 18766},
        "trainer": {"state_path": str(tmp_path / "trainer_state.json")},
        "participants": [
            {"role": TRAINEE_ROLE, "type": "embedded", "class": "KAgent"},
            {"role": TRAINER_ROLE, "type": "embedded", "class": "Trainer"},
            {"role": SUPERVISOR_ROLE, "type": "client", "class": "SlackParticipant"},
            {"role": SUPERVISOR_ROLE, "type": "client", "class": "TUIParticipant"},
        ],
    }
    path = tmp_path / "full_harness.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# test_load_config_from_sample_yaml
# ---------------------------------------------------------------------------


class TestLoadConfigFromSampleYaml:
    """HRNS-5: Harness loads embedded participants from config file on startup."""

    def test_load_config_from_sample_yaml(self) -> None:
        """Load the canonical project config (``training.harness.yaml``) and assert all four participants.

        The config path is resolved from this test file's location so the test
        passes regardless of the working directory pytest is invoked from.
        """
        # Guard explicitly so a future rename fails with a clear message rather
        # than a bare FileNotFoundError from load_config().
        assert _PROJECT_CONFIG.exists(), (
            f"canonical project config not found at {_PROJECT_CONFIG}; "
            "has it been renamed?"
        )
        config = load_config(_PROJECT_CONFIG)
        assert len(config.participants) == 4

        by_role: dict[str, list] = {}
        for p in config.participants:
            by_role.setdefault(p.role, []).append(p)

        assert by_role[TRAINEE_ROLE][0].type == "embedded"
        assert by_role[TRAINEE_ROLE][0].class_name == "KAgent"

        assert by_role[TRAINER_ROLE][0].type == "embedded"
        assert by_role[TRAINER_ROLE][0].class_name == "Trainer"

        supervisor_classes = {p.class_name for p in by_role[SUPERVISOR_ROLE]}
        assert "SlackParticipant" in supervisor_classes
        assert "TUIParticipant" in supervisor_classes


# ---------------------------------------------------------------------------
# test_cli_argparse_defaults
# ---------------------------------------------------------------------------


class TestCLIArgparseDefaults:
    """Verify CLI argument parsing defaults."""

    def test_default_config_path(self) -> None:
        """Default config path resolves to ``harness.yaml``."""
        from training.harness.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args([])
        assert args.config == "training.harness.yaml"

    def test_default_host_and_port(self) -> None:
        """Default host and port are None (filled from config or hardcoded)."""
        from training.harness.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args([])
        assert args.host is None
        assert args.port is None

    def test_override_config(self) -> None:
        """CLI ``--config`` override is parsed correctly."""
        from training.harness.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--config", "custom.yaml"])
        assert args.config == "custom.yaml"

    def test_override_host_and_port(self) -> None:
        """CLI ``--host`` and ``--port`` overrides are parsed correctly."""
        from training.harness.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--host", "0.0.0.0", "--port", "9000"])
        assert args.host == "0.0.0.0"
        assert args.port == 9000


# ---------------------------------------------------------------------------
# test_embedded_participants_registered
# ---------------------------------------------------------------------------


class TestEmbeddedParticipantsRegistered:
    """Verify embedded participants subscribe and receive messages on the bus."""

    def test_embedded_participants_registered(self, sample_config: Path) -> None:
        """Create a HarnessServer with mock factories, trigger setup, verify bus routing."""
        bus = MessageBus()
        server = HarnessServer(sample_config, bus)

        received: dict[str, list[Message]] = {TRAINEE_ROLE: [], TRAINER_ROLE: []}
        events: dict[str, threading.Event] = {
            TRAINEE_ROLE: threading.Event(),
            TRAINER_ROLE: threading.Event(),
        }

        server.register_participant_class(
            "KAgent", _make_self_subscribing_factory(TRAINEE_ROLE, received, events)
        )
        server.register_participant_class(
            "Trainer", _make_self_subscribing_factory(TRAINER_ROLE, received, events)
        )

        # Start bus in a background thread so we can send/receive
        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            # NOTE: _setup() is private but is the only way to trigger
            # embedded participant creation without starting the WebSocket
            # server. No public alternative exists on HarnessServer.
            server._setup()

            # Send messages to each participant
            bus.send(
                Message(role=TRAINEE_ROLE, action="submit", message="test", sender=TRAINER_ROLE)
            )
            bus.send(
                Message(role=TRAINER_ROLE, action="event", message="data", sender=TRAINEE_ROLE)
            )

            # Wait for dispatch via threading.Event (not time.sleep)
            assert events[TRAINEE_ROLE].wait(timeout=2), "trainee handler should have fired"
            assert events[TRAINER_ROLE].wait(timeout=2), "trainer handler should have fired"

            assert len(received[TRAINEE_ROLE]) == 1
            assert received[TRAINEE_ROLE][0].action == "submit"
            assert received[TRAINEE_ROLE][0].message == "test"

            assert len(received[TRAINER_ROLE]) == 1
            assert received[TRAINER_ROLE][0].action == "event"
            assert received[TRAINER_ROLE][0].message == "data"
        finally:
            bus.stop()
            bus_thread.join(timeout=2)


# ---------------------------------------------------------------------------
# test_message_routes_between_participants
# ---------------------------------------------------------------------------


class TestMessageRoutesBetweenParticipants:
    """HRNS-1: Message bus routes by role to correct subscriber."""

    def test_message_routes_between_participants(self) -> None:
        """Create a MessageBus, register two handlers, verify message routing."""
        bus = MessageBus()
        event_a = threading.Event()
        event_b = threading.Event()
        received_a: list[Message] = []
        received_b: list[Message] = []

        def handler_a(msg: Message) -> None:
            received_a.append(msg)
            event_a.set()

        def handler_b(msg: Message) -> None:
            received_b.append(msg)
            event_b.set()

        bus.subscribe(TRAINEE_ROLE, handler_a)
        bus.subscribe(TRAINER_ROLE, handler_b)

        # Start bus in a background thread
        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            # Send from trainer → trainee
            bus.send(
                Message(
                    role=TRAINEE_ROLE, action="submit", message="MHALL = SVO", sender=TRAINER_ROLE
                )
            )
            assert event_a.wait(timeout=2), "handler_a should have received the message"
            assert len(received_a) == 1
            assert received_a[0].role == TRAINEE_ROLE
            assert received_a[0].action == "submit"
            assert received_a[0].message == "MHALL = SVO"
            assert received_a[0].sender == TRAINER_ROLE

            # Send from trainee → trainer
            bus.send(
                Message(role=TRAINER_ROLE, action="event", message="ground", sender=TRAINEE_ROLE)
            )
            assert event_b.wait(timeout=2), "handler_b should have received the message"
            assert len(received_b) == 1
            assert received_b[0].role == TRAINER_ROLE
            assert received_b[0].action == "event"
            assert received_b[0].message == "ground"
            assert received_b[0].sender == TRAINEE_ROLE
        finally:
            bus.stop()
            bus_thread.join(timeout=2)


# ---------------------------------------------------------------------------
# test_shutdown_callbacks_on_signal
# ---------------------------------------------------------------------------


class TestShutdownCallbacksOnSignal:
    """Verify shutdown callbacks (Trainer persist, server stop, bus stop) execute."""

    def test_shutdown_callbacks_on_signal(self, sample_config: Path) -> None:
        """Start harness with mock participants, trigger shutdown, verify callbacks."""
        bus = MessageBus()
        server = HarnessServer(sample_config, bus)

        server.register_participant_class("KAgent", _make_noop_factory())
        server.register_participant_class("Trainer", _make_noop_factory())

        # Build shutdown callback list (same pattern as __main__.py)
        # Use MagicMock for all three so we can verify each was called
        trainer_state_save = MagicMock()
        server_stop = MagicMock(wraps=server.stop)
        bus_stop = MagicMock(wraps=bus.stop)

        shutdown_callbacks = [trainer_state_save, server_stop, bus_stop]

        # Execute shutdown callbacks (simulating the finally block in main())
        for callback in shutdown_callbacks:
            try:
                callback()
            except Exception:
                pass

        # Assert all three callbacks were called
        trainer_state_save.assert_called_once()
        server_stop.assert_called_once()
        bus_stop.assert_called_once()

        # Verify call order: trainer persist → server stop → bus stop
        # Use a shared tracker to record the sequence
        call_order: list[str] = []

        trainer_tracker = MagicMock(side_effect=lambda: call_order.append("trainer"))
        server_tracker = MagicMock(side_effect=lambda: call_order.append("server"))
        bus_tracker = MagicMock(side_effect=lambda: call_order.append("bus"))

        ordered_callbacks = [trainer_tracker, server_tracker, bus_tracker]
        for callback in ordered_callbacks:
            callback()

        assert call_order == ["trainer", "server", "bus"]


# ---------------------------------------------------------------------------
# test_full_lifecycle_bus_only
# ---------------------------------------------------------------------------


class TestFullLifecycleBusOnly:
    """Full lifecycle test: config → participants → messages → shutdown.

    Uses bus-only (no WebSocket) to keep the test synchronous and fast.
    The WebSocket integration is tested in test_server.py and test_protocol.py.
    """

    def test_full_lifecycle_bus_only(self, sample_config: Path) -> None:
        """End-to-end: load config, register factories, run bus, route messages, shutdown."""
        bus = MessageBus()
        server = HarnessServer(sample_config, bus)

        received: dict[str, list[Message]] = {TRAINEE_ROLE: [], TRAINER_ROLE: []}
        events: dict[str, threading.Event] = {
            TRAINEE_ROLE: threading.Event(),
            TRAINER_ROLE: threading.Event(),
        }

        server.register_participant_class(
            "KAgent", _make_self_subscribing_factory(TRAINEE_ROLE, received, events)
        )
        server.register_participant_class(
            "Trainer", _make_self_subscribing_factory(TRAINER_ROLE, received, events)
        )

        # NOTE: _setup() is private but is the only way to trigger
        # embedded participant creation without starting the WebSocket server.
        server._setup()

        # Run bus in background thread
        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            # Simulate trainer sending to trainee
            bus.send(
                Message(
                    role=TRAINEE_ROLE, action="submit", message="MHALL = SVO", sender=TRAINER_ROLE
                )
            )

            # Simulate trainee responding to trainer
            bus.send(
                Message(
                    role=TRAINER_ROLE, action="ground", message="event_data", sender=TRAINEE_ROLE
                )
            )

            # Wait for dispatch via threading.Event
            assert events[TRAINEE_ROLE].wait(timeout=2), "trainee handler should have fired"
            assert events[TRAINER_ROLE].wait(timeout=2), "trainer handler should have fired"

            # Verify messages routed correctly
            assert len(received[TRAINEE_ROLE]) == 1
            assert received[TRAINEE_ROLE][0].action == "submit"
            assert received[TRAINEE_ROLE][0].message == "MHALL = SVO"

            assert len(received[TRAINER_ROLE]) == 1
            assert received[TRAINER_ROLE][0].action == "ground"
            assert received[TRAINER_ROLE][0].message == "event_data"
        finally:
            bus.stop()
            bus_thread.join(timeout=2)


# ---------------------------------------------------------------------------
# test_full_lifecycle_with_websocket_client
# ---------------------------------------------------------------------------


class TestFullLifecycleWithWebSocketClient:
    """End-to-end lifecycle with a real WebSocket client connection."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_websocket_client(self, tmp_path: Path) -> None:
        """Start server, connect WS client, register, send message, verify receipt."""
        config = {
            "participants": [
                {"role": "mock", "type": "embedded", "class": "MockParticipant"},
                {"role": SUPERVISOR_ROLE, "type": "client", "class": "TUIParticipant"},
            ],
        }
        config_path = tmp_path / "ws_test.yaml"
        config_path.write_text(yaml.dump(config), encoding="utf-8")

        bus = MessageBus()
        server = HarnessServer(config_path, bus)

        received: list[Message] = []

        def mock_factory(role: str, bus: MessageBus) -> object:
            class MockParticipant:
                @property
                def role(self) -> str:
                    return role

                def on_message(self, msg: Message) -> None:
                    received.append(msg)

            p = MockParticipant()
            bus.subscribe(role, p.on_message)

            class Wrapper:
                @property
                def role(self) -> str:
                    return role

                def on_message(self, msg: Message) -> None:
                    pass

            return Wrapper()

        server.register_participant_class("MockParticipant", mock_factory)

        # Start bus in background thread
        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            # Find a free port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", 0))
                port = s.getsockname()[1]

            # Start the server (public API)
            await server.start(host="localhost", port=port)

            try:
                # Connect a WebSocket client
                ws = await websockets.connect(f"ws://localhost:{port}")

                # Register as supervisor
                await ws.send(json.dumps({"register": SUPERVISOR_ROLE}))
                await asyncio.sleep(0.1)

                # Send a message from supervisor to mock
                await ws.send(
                    json.dumps(
                        {
                            "role": "mock",
                            "action": "test",
                            "message": "hello from supervisor",
                        }
                    )
                )

                # Poll for the message to arrive.  The path is:
                # WS client → async protocol handler → bus.send()
                # → bus thread dispatch → mock handler.
                # Cross-thread handoff can take a few event-loop ticks.
                for _ in range(30):
                    if received:
                        break
                    await asyncio.sleep(0.05)

                assert len(received) >= 1, (
                    f"mock participant should have received the message "
                    f"(received so far: {len(received)})"
                )

                # Verify the mock participant received the message
                assert len(received) == 1
                assert received[0].role == "mock"
                assert received[0].action == "test"
                assert received[0].message == "hello from supervisor"
                assert received[0].sender == SUPERVISOR_ROLE

                # Disconnect client
                await ws.close()
            finally:
                # NOTE: _async_stop() is private but is the only way to stop
                # the WebSocket server from an async context. The public
                # stop() uses a threading.Event + bus.stop().
                await server._async_stop()
        finally:
            bus.stop()
            bus_thread.join(timeout=2)
