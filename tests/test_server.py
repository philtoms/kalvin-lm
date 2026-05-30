"""Tests for the harness server (HRNS-5, HRNS-6).

Covers config loading, embedded participant setup, WebSocket client
connections, config validation, and the participant registry.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from harness.bus import MessageBus
from harness.message import Message
from harness.server import ConfigError, HarnessConfig, HarnessServer, load_config


# -- helpers ---------------------------------------------------------------


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    """Write *data* as YAML to *path* and return the path."""
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return path


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    """Write *data* as JSON to *path* and return the path."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _sample_config() -> dict[str, Any]:
    """Return a minimal valid harness config dict."""
    return {
        "participants": [
            {
                "address": "kalvin",
                "type": "embedded",
                "class": "KAgent",
            },
            {
                "address": "trainer",
                "type": "embedded",
                "class": "Trainer",
            },
            {
                "address": "slack",
                "type": "client",
                "class": "SlackParticipant",
            },
        ]
    }


class MockParticipant:
    """A mock participant for testing the registry and bus wiring."""

    def __init__(self, address: str, bus: MessageBus) -> None:
        self.address = address
        self.bus = bus
        self.messages: list[Message] = []

    def on_message(self, msg: Message) -> None:
        self.messages.append(msg)


# -- config loading --------------------------------------------------------


class TestLoadConfigYamlAndJson:
    """Equivalent YAML and JSON configs produce identical ``HarnessConfig``."""

    def test_load_config_yaml_and_json(self, tmp_path: Path) -> None:
        data = _sample_config()
        yaml_path = _write_yaml(tmp_path / "harness.yaml", data)
        json_path = _write_json(tmp_path / "harness.json", data)

        yaml_config = load_config(yaml_path)
        json_config = load_config(json_path)

        assert yaml_config == json_config

        # Spot-check the parsed content.
        assert len(yaml_config.participants) == 3
        assert yaml_config.participants[0].address == "kalvin"
        assert yaml_config.participants[0].type == "embedded"
        assert yaml_config.participants[0].class_name == "KAgent"
        assert yaml_config.participants[2].type == "client"


# -- config validation -----------------------------------------------------


class TestConfigValidation:
    """``load_config`` raises ``ConfigError`` for invalid configs."""

    def test_missing_participants_key(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path / "bad.yaml", {"other": "stuff"})
        with pytest.raises(ConfigError, match="missing 'participants'"):
            load_config(path)

    def test_participant_missing_address(self, tmp_path: Path) -> None:
        data = {"participants": [{"type": "embedded", "class": "KAgent"}]}
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ConfigError, match="missing required 'address'"):
            load_config(path)

    def test_participant_invalid_type(self, tmp_path: Path) -> None:
        data = {
            "participants": [
                {"address": "kalvin", "type": "invalid", "class": "KAgent"}
            ]
        }
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ConfigError, match="invalid type 'invalid'"):
            load_config(path)

    def test_duplicate_addresses(self, tmp_path: Path) -> None:
        data = {
            "participants": [
                {"address": "kalvin", "type": "embedded", "class": "KAgent"},
                {"address": "kalvin", "type": "client", "class": "TUI"},
            ]
        }
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ConfigError, match="duplicate address 'kalvin'"):
            load_config(path)

    def test_participant_missing_class(self, tmp_path: Path) -> None:
        data = {
            "participants": [
                {"address": "kalvin", "type": "embedded"}
            ]
        }
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ConfigError, match="missing required 'class'"):
            load_config(path)

    def test_participants_not_list(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path / "bad.yaml", {"participants": "nope"})
        with pytest.raises(ConfigError, match="'participants' must be a list"):
            load_config(path)


# -- embedded participants -------------------------------------------------


class TestLoadEmbeddedParticipants:
    """HRNS-5: Harness loads embedded participants from config file."""

    def test_load_embedded_participants(self, tmp_path: Path) -> None:
        config_data = {
            "participants": [
                {
                    "address": "kalvin",
                    "type": "embedded",
                    "class": "MockKAgent",
                },
            ]
        }
        path = _write_yaml(tmp_path / "harness.yaml", config_data)

        bus = MessageBus()
        server = HarnessServer(path, bus)
        server.register_participant_class("MockKAgent", MockParticipant)

        # Run _setup to instantiate embedded participants.
        server._setup()

        # The participant should be subscribed on the bus.
        assert "kalvin" in server._embedded_participants
        participant = server._embedded_participants["kalvin"]
        assert isinstance(participant, MockParticipant)
        assert participant.address == "kalvin"

        # Verify bus dispatch triggers the participant's on_message.
        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            bus.send(
                Message(
                    address="kalvin",
                    action="submit",
                    message="MHALL = SVO",
                    sender="trainer",
                )
            )
            # Give the bus time to dispatch.
            import time

            time.sleep(0.1)
        finally:
            bus.stop()
            bus_thread.join(timeout=5)

        assert len(participant.messages) == 1
        assert participant.messages[0].action == "submit"
        assert participant.messages[0].sender == "trainer"


class TestParticipantRegistry:
    """``register_participant_class`` wires factory to config entries."""

    def test_participant_registry(self, tmp_path: Path) -> None:
        config_data = {
            "participants": [
                {
                    "address": "agent",
                    "type": "embedded",
                    "class": "MockParticipant",
                },
            ]
        }
        path = _write_yaml(tmp_path / "harness.yaml", config_data)

        bus = MessageBus()
        server = HarnessServer(path, bus)

        # Track factory invocations.
        factory_calls: list[tuple[str, Any]] = []

        def factory(address: str, bus: MessageBus) -> MockParticipant:
            factory_calls.append((address, bus))
            return MockParticipant(address, bus)

        server.register_participant_class("MockParticipant", factory)
        server._setup()

        assert len(factory_calls) == 1
        assert factory_calls[0][0] == "agent"
        assert factory_calls[0][1] is bus


class TestUnregisteredEmbeddedClass:
    """Missing factory for an embedded participant raises ConfigError."""

    def test_unregistered_class(self, tmp_path: Path) -> None:
        config_data = {
            "participants": [
                {
                    "address": "kalvin",
                    "type": "embedded",
                    "class": "UnknownClass",
                },
            ]
        }
        path = _write_yaml(tmp_path / "harness.yaml", config_data)

        bus = MessageBus()
        server = HarnessServer(path, bus)

        with pytest.raises(ConfigError, match="no factory registered"):
            server._setup()


# -- WebSocket client connections ------------------------------------------


class TestWebSocketClientConnect:
    """HRNS-6: Harness accepts WebSocket client connections."""

    @pytest.mark.asyncio
    async def test_websocket_client_connect(self, tmp_path: Path) -> None:
        import websockets

        config_data = {
            "participants": [
                {
                    "address": "ui",
                    "type": "client",
                    "class": "TUIParticipant",
                },
            ]
        }
        path = _write_yaml(tmp_path / "harness.yaml", config_data)

        bus = MessageBus()
        server = HarnessServer(path, bus)

        try:
            await server.start(host="localhost", port=18780)

            # Connect a client and register.
            ws = await websockets.connect("ws://localhost:18780")
            await ws.send(json.dumps({"register": "ui"}))
            await asyncio.sleep(0.05)

            # Client should still be open (no error).
            # Send a ping to verify.
            await ws.send(
                json.dumps(
                    {"address": "kalvin", "action": "submit", "message": "test"}
                )
            )
            await asyncio.sleep(0.05)

            # Clean disconnect.
            await ws.close()
        finally:
            await server._async_stop()
