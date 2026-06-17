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

from training.harness.bus import MessageBus
from training.harness.constants import SUPERVISOR_ROLE, TRAINER_ROLE
from training.harness.message import Message
from training.harness.server import ConfigError, HarnessServer, load_config

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
                "role": "kalvin",
                "type": "embedded",
                "class": "KAgent",
            },
            {
                "role": TRAINER_ROLE,
                "type": "embedded",
                "class": "Trainer",
            },
            {
                "role": "slack",
                "type": "client",
                "class": "SlackParticipant",
            },
        ]
    }


class MockParticipant:
    """A mock participant for testing the registry and bus wiring."""

    def __init__(self, role: str, bus: MessageBus) -> None:
        self.role = role
        self.bus = bus
        self.messages: list[Message] = []

    def on_message(self, msg: Message) -> None:
        self.messages.append(msg)


# -- config loading --------------------------------------------------------


class TestLoadConfigYamlAndJson:
    """Equivalent YAML and JSON configs produce identical ``HarnessConfig``."""

    def test_load_config_yaml_and_json(self, tmp_path: Path) -> None:
        data = _sample_config()
        yaml_path = _write_yaml(tmp_path / "training.harness.yaml", data)
        json_path = _write_json(tmp_path / "training.harness.json", data)

        yaml_config = load_config(yaml_path)
        json_config = load_config(json_path)

        assert yaml_config == json_config

        # Spot-check the parsed content.
        assert len(yaml_config.participants) == 3
        assert yaml_config.participants[0].role == "kalvin"
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

    def test_participant_missing_role(self, tmp_path: Path) -> None:
        data = {"participants": [{"type": "embedded", "class": "KAgent"}]}
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ConfigError, match="missing required 'role'"):
            load_config(path)

    def test_participant_invalid_type(self, tmp_path: Path) -> None:
        data = {"participants": [{"role": "kalvin", "type": "invalid", "class": "KAgent"}]}
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ConfigError, match="invalid type 'invalid'"):
            load_config(path)

    def test_duplicate_embedded_roles(self, tmp_path: Path) -> None:
        data = {
            "participants": [
                {"role": "kalvin", "type": "embedded", "class": "KAgent"},
                {"role": "kalvin", "type": "embedded", "class": "AnotherAgent"},
            ]
        }
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ConfigError, match="duplicate role 'kalvin' for embedded"):
            load_config(path)

    def test_participant_missing_class(self, tmp_path: Path) -> None:
        data = {"participants": [{"role": "kalvin", "type": "embedded"}]}
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ConfigError, match="missing required 'class'"):
            load_config(path)

    def test_participants_not_list(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path / "bad.yaml", {"participants": "nope"})
        with pytest.raises(ConfigError, match="'participants' must be a list"):
            load_config(path)

    def test_duplicate_roles_allowed_for_clients(self, tmp_path: Path) -> None:
        """Duplicate roles are allowed when all entries are type: client."""
        data = {
            "participants": [
                {"role": SUPERVISOR_ROLE, "type": "client", "class": "TUI1"},
                {"role": SUPERVISOR_ROLE, "type": "client", "class": "TUI2"},
            ]
        }
        path = _write_yaml(tmp_path / "ok.yaml", data)
        config = load_config(path)
        assert len(config.participants) == 2
        assert config.participants[0].role == SUPERVISOR_ROLE
        assert config.participants[1].role == SUPERVISOR_ROLE

    def test_mixed_duplicate_role_embedded_and_client_ok(self, tmp_path: Path) -> None:
        """A client may share a role with an embedded participant."""
        data = {
            "participants": [
                {"role": "kalvin", "type": "embedded", "class": "KAgent"},
                {"role": "kalvin", "type": "client", "class": "TUI"},
            ]
        }
        path = _write_yaml(tmp_path / "ok.yaml", data)
        config = load_config(path)
        assert len(config.participants) == 2


# -- embedded participants -------------------------------------------------


class TestLoadEmbeddedParticipants:
    """HRNS-5: Harness loads embedded participants from config file."""

    def test_load_embedded_participants(self, tmp_path: Path) -> None:
        config_data = {
            "participants": [
                {
                    "role": "kalvin",
                    "type": "embedded",
                    "class": "MockKAgent",
                },
            ]
        }
        path = _write_yaml(tmp_path / "training.harness.yaml", config_data)

        bus = MessageBus()
        server = HarnessServer(path, bus)
        server.register_participant_class("MockKAgent", MockParticipant)

        # Run _setup to instantiate embedded participants.
        server._setup()

        # The participant should be subscribed on the bus.
        assert "kalvin" in server._embedded_participants
        participant = server._embedded_participants["kalvin"]
        assert isinstance(participant, MockParticipant)
        assert participant.role == "kalvin"

        # Verify bus dispatch triggers the participant's on_message.
        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            bus.send(
                Message(
                    role="kalvin",
                    action="submit",
                    message="MHALL = SVO",
                    sender=TRAINER_ROLE,
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
        assert participant.messages[0].sender == TRAINER_ROLE


class TestParticipantRegistry:
    """``register_participant_class`` wires factory to config entries."""

    def test_participant_registry(self, tmp_path: Path) -> None:
        config_data = {
            "participants": [
                {
                    "role": "agent",
                    "type": "embedded",
                    "class": "MockParticipant",
                },
            ]
        }
        path = _write_yaml(tmp_path / "training.harness.yaml", config_data)

        bus = MessageBus()
        server = HarnessServer(path, bus)

        # Track factory invocations.
        factory_calls: list[tuple[str, Any]] = []

        def factory(role: str, bus: MessageBus) -> MockParticipant:
            factory_calls.append((role, bus))
            return MockParticipant(role, bus)

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
                    "role": "kalvin",
                    "type": "embedded",
                    "class": "UnknownClass",
                },
            ]
        }
        path = _write_yaml(tmp_path / "training.harness.yaml", config_data)

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
                    "role": "ui",
                    "type": "client",
                    "class": "TUIParticipant",
                },
            ]
        }
        path = _write_yaml(tmp_path / "training.harness.yaml", config_data)

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
            await ws.send(json.dumps({"role": "kalvin", "action": "submit", "message": "test"}))
            await asyncio.sleep(0.05)

            # Clean disconnect.
            await ws.close()
        finally:
            await server._async_stop()
