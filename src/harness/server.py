"""Harness server — loads participants, starts WebSocket, runs the bus loop.

Reads a YAML/JSON configuration file listing participants (embedded or client),
instantiates embedded participants from a registry, starts a WebSocket server
for client participants, and runs the bus event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import websockets

from harness.bus import MessageBus
from harness.protocol import WebSocketProtocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParticipantConfig:
    """A single participant entry from the harness config file."""

    address: str
    type: str  # "embedded" | "client"
    class_name: str


@dataclass(frozen=True)
class HarnessConfig:
    """Parsed harness configuration."""

    participants: list[ParticipantConfig]


def load_config(path: str | Path) -> HarnessConfig:
    """Load a YAML or JSON config file and return a validated ``HarnessConfig``.

    Raises ``ConfigError`` on validation failures.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in {path}: {exc}") from exc

    return _validate_config(data, path)


def _validate_config(data: Any, path: Path) -> HarnessConfig:
    """Validate the raw config dict and return a ``HarnessConfig``."""
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: config must be a mapping")

    participants_raw = data.get("participants")
    if participants_raw is None:
        raise ConfigError(f"{path}: missing 'participants' key")
    if not isinstance(participants_raw, list):
        raise ConfigError(f"{path}: 'participants' must be a list")

    entries: list[ParticipantConfig] = []
    seen_addresses: set[str] = set()

    for idx, entry in enumerate(participants_raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: participants[{idx}] must be a mapping")

        address = entry.get("address")
        if not address or not isinstance(address, str):
            raise ConfigError(
                f"{path}: participants[{idx}] missing required 'address'"
            )
        if address in seen_addresses:
            raise ConfigError(
                f"{path}: duplicate address '{address}' in participants"
            )
        seen_addresses.add(address)

        ptype = entry.get("type")
        if ptype not in ("embedded", "client"):
            raise ConfigError(
                f"{path}: participants[{idx}] has invalid type '{ptype}' "
                f"(must be 'embedded' or 'client')"
            )

        class_name = entry.get("class")
        if not class_name or not isinstance(class_name, str):
            raise ConfigError(
                f"{path}: participants[{idx}] missing required 'class'"
            )

        entries.append(
            ParticipantConfig(address=address, type=ptype, class_name=class_name)
        )

    return HarnessConfig(participants=entries)


class ConfigError(Exception):
    """Raised when the harness configuration is invalid."""


# ---------------------------------------------------------------------------
# Participant registry
# ---------------------------------------------------------------------------

# Type for factory callables: (address, bus) -> participant with .on_message(msg)
ParticipantFactory = Callable[[str, MessageBus], Any]


# ---------------------------------------------------------------------------
# Harness server
# ---------------------------------------------------------------------------


class HarnessServer:
    """The multi-agent harness runtime.

    Loads a configuration file, instantiates embedded participants from a
    registry, starts a WebSocket server for client participants, and runs
    the bus event loop.

    Usage::

        server = HarnessServer("harness.yaml", bus)
        server.register_participant_class("KAgent", kagent_factory)
        server.run_sync(host="localhost", port=8765)
    """

    def __init__(self, config_path: str | Path, bus: MessageBus) -> None:
        self._config = load_config(config_path)
        self._bus = bus
        self._participant_registry: dict[str, ParticipantFactory] = {}
        self._embedded_participants: dict[str, Any] = {}
        self._protocol: WebSocketProtocol | None = None
        self._ws_server: websockets.asyncio.server.Server | None = None
        self._bus_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # -- participant registry ------------------------------------------------

    def register_participant_class(
        self, name: str, factory: ParticipantFactory
    ) -> None:
        """Register *factory* as the constructor for participant class *name*.

        The factory is called as ``factory(address, bus)`` and must return an
        object with ``on_message(msg)``.
        """
        self._participant_registry[name] = factory

    # -- setup ---------------------------------------------------------------

    def _setup(self) -> None:
        """Instantiate embedded participants from the config.

        Each embedded participant's factory is looked up in the registry,
        called with ``(address, bus)``, and the result is subscribed to the
        bus for its address.
        """
        for pcfg in self._config.participants:
            if pcfg.type != "embedded":
                continue
            factory = self._participant_registry.get(pcfg.class_name)
            if factory is None:
                raise ConfigError(
                    f"no factory registered for embedded participant "
                    f"class '{pcfg.class_name}' (address '{pcfg.address}')"
                )
            participant = factory(pcfg.address, self._bus)
            self._bus.subscribe(pcfg.address, participant.on_message)
            self._embedded_participants[pcfg.address] = participant
            logger.info(
                "Embedded participant '%s' (%s) subscribed to '%s'",
                pcfg.class_name,
                id(participant),
                pcfg.address,
            )

    # -- async server --------------------------------------------------------

    async def start(self, host: str = "localhost", port: int = 8765) -> None:
        """Start the WebSocket server and instantiate embedded participants.

        Creates the ``WebSocketProtocol``, starts ``websockets.serve()``,
        and sets up embedded participants.
        """
        self._setup()

        self._protocol = WebSocketProtocol(self._bus)
        self._ws_server = await websockets.serve(
            self._protocol.handle_connection,
            host,
            port,
        )
        logger.info("WebSocket server listening on %s:%s", host, port)

    async def _async_stop(self) -> None:
        """Stop the WebSocket server gracefully."""
        if self._ws_server is not None:
            self._ws_server.close()
            await self._ws_server.wait_closed()
            logger.info("WebSocket server stopped")
            self._ws_server = None

    # -- sync runner ---------------------------------------------------------

    def run_sync(self, host: str = "localhost", port: int = 8765) -> None:
        """Run the harness server synchronously.

        Starts the bus event loop on a dedicated thread and the WebSocket
        server on the current thread.  Blocks until interrupted (SIGINT /
        SIGTERM) or ``stop()`` is called.
        """
        # Start bus in a background thread.
        self._bus_thread = threading.Thread(
            target=self._bus.run, name="harness-bus", daemon=True
        )
        self._bus_thread.start()

        # Run async WS server on the current thread.
        asyncio.run(self._run_async(host, port))

    async def _run_async(self, host: str, port: int) -> None:
        """Async entry point: start server, wait for stop signal."""
        await self.start(host, port)

        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            stop_event.set()

        loop.add_signal_handler(signal.SIGTERM, _signal_handler)
        loop.add_signal_handler(signal.SIGINT, _signal_handler)

        # Also support programmatic stop via threading.Event.
        while not stop_event.is_set():
            if self._stop_event.is_set():
                break
            # Poll every 200ms so we can check the threading event.
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.2)
            except (TimeoutError, asyncio.TimeoutError):
                continue

        await self._async_stop()

    def stop(self) -> None:
        """Signal the server to stop (safe to call from any thread)."""
        self._stop_event.set()
        self._bus.stop()
        logger.info("Harness server stop requested")
