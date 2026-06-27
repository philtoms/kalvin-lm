"""CLI Supervisor — headless WebSocket client for auto-tune.

Connects to the harness, receives events one at a time, enriches them,
writes them to ``events.jsonl``, and blocks per-event waiting for commands
from ``cmd.json``.  This is the core runtime that lets an LLM agent observe
and control auto-tune training sessions via the file-based command/status
protocol.

Spec ref: specs/auto-tune.md §Per-Event Blocking Model, §Command Processing,
          §Run Completion, §Error Handling
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import websockets

from training.participants.auto_tune.events import enrich_event
from training.participants.commands import parse_command
from training.harness.constants import TRAINER_ROLE

logger = logging.getLogger(__name__)


class CLISupervisor:
    """Headless WebSocket client participant for auto-tune.

    Parameters
    ----------
    session_dir:
        Path to the session directory containing ``config.json``,
        ``cmd.json``, ``status.json``, and ``events.jsonl``.
    """

    def __init__(self, session_dir: str) -> None:
        self._session_dir = Path(session_dir)

        # Load harness URL from session config
        config_path = self._session_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self._url: str = config["harness_url"]

        # Internal state
        self._seq: int = 0
        self._connected: bool = False
        self._ws: websockets.asyncio.client.ClientConnection | None = None
        self._latest_ratify_proposal: Any = None
        self._started_at: str | None = None
        self._last_command: dict | None = None

    # -- public API ----------------------------------------------------------

    async def connect(self) -> None:
        """Open WebSocket, send registration frame, write connected event."""
        self._ws = await websockets.connect(self._url)
        await self._ws.send(json.dumps({"register": "supervisor"}))
        logger.info("CLISupervisor registered as 'supervisor'")

        self._connected = True
        self._started_at = datetime.now().isoformat()

        # Write connected event (seq 1)
        self._seq = 1
        self._append_event({"seq": 1, "type": "connected"})
        self._write_status(state="waiting_for_event")

    async def disconnect(self) -> None:
        """Close the WebSocket and mark as disconnected."""
        self._connected = False
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def run(self) -> None:
        """Self-contained main entry point.

        Connects to the harness, polls for an initial command (e.g. start),
        then enters the per-event blocking loop.
        """
        await self.connect()

        try:
            # Poll for an initial command before waiting for harness events,
            # so pi can send "start" before any arrive. Without this the
            # supervisor deadlocks: it blocks on ws.recv() but the harness
            # won't send frames until it gets a command.
            self._write_status(state="waiting_for_command")
            cmd = await self._poll_command()
            if cmd is None:
                return
            should_exit = await self._process_command(cmd)
            if should_exit:
                return
            self._write_status(state="waiting_for_event")

            while self._connected:
                try:
                    raw = await self._ws.recv()
                except websockets.ConnectionClosed:
                    await self._handle_disconnect()
                    return

                try:
                    frame = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Malformed frame from harness: %s", raw[:200])
                    continue

                self._seq += 1
                event = enrich_event(frame, self._seq)

                if event.get("type") == "ratify_request":
                    # Buffer the canonical KLine wire dict ({"signature",
                    # "nodes"}) straight from the inbound frame, not the
                    # enriched display object produced by ``enrich_event``
                    # (which wraps the proposal as {"raw": {...}, "source": ...}).
                    # The emitted countersign frame must carry the wire
                    # contract's KLine shape (see ``_domain_json_default`` /
                    # ``events._build_kline_display``); the raw frame's
                    # proposal is exactly that shape, since the harness
                    # encoded the KLine outbound before it reached us.
                    self._latest_ratify_proposal = frame.get("message", {}).get("proposal")

                self._append_event(event)

                if event.get("type") == "progress" and event.get("status") == "complete":
                    self._write_status(state="run_complete")
                else:
                    self._write_status(state="waiting_for_command")

                cmd = await self._poll_command()
                if cmd is None:
                    return

                should_exit = await self._process_command(cmd)
                if should_exit:
                    return

                self._write_status(state="waiting_for_event")
        finally:
            self._connected = False

    # -- file I/O helpers ----------------------------------------------------

    def _write_status(self, state: str) -> None:
        """Write the status object atomically to ``status.json``."""
        status = {
            "pid": os.getpid(),
            "connected": self._connected,
            "last_event_seq": self._seq,
            "last_command": self._last_command,
            "state": state,
            "started_at": self._started_at,
        }
        tmp_path = self._session_dir / "status.json.tmp"
        final_path = self._session_dir / "status.json"
        tmp_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        os.replace(str(tmp_path), str(final_path))

    def _append_event(self, event: dict) -> None:
        """Append one JSON line to ``events.jsonl``."""
        events_path = self._session_dir / "events.jsonl"
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
            f.flush()

    async def _poll_command(self) -> dict | None:
        """Poll ``cmd.json`` for existence every 100ms.

        Returns the command dict when found, or ``None`` if disconnected.
        """
        cmd_path = self._session_dir / "cmd.json"
        while self._connected:
            if cmd_path.exists():
                try:
                    raw = cmd_path.read_text(encoding="utf-8")
                    cmd = json.loads(raw)
                    os.unlink(str(cmd_path))
                    self._last_command = cmd
                    self._write_status(state=self._current_state())
                    return cmd
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Failed to read cmd.json: %s", exc)
                    await asyncio.sleep(0.1)
                    continue
            await asyncio.sleep(0.1)
        return None

    def _current_state(self) -> str:
        """Return the last state written to status.json.

        Reads the status file to avoid needing to store state separately.
        """
        status_path = self._session_dir / "status.json"
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
            return data.get("state", "unknown")
        except (OSError, json.JSONDecodeError):
            return "unknown"

    # -- command dispatch ----------------------------------------------------

    async def _process_command(self, cmd: dict) -> bool:
        """Process a command from ``cmd.json``.

        Returns ``True`` if the supervisor should exit, ``False`` to continue.
        """
        action = cmd.get("action", "")

        if action == "shutdown":
            self._seq += 1
            self._append_event({"seq": self._seq, "type": "disconnected"})
            self._write_status(state="shutting_down")
            self._connected = False
            if self._ws is not None:
                await self._ws.close()
                self._ws = None
            return True

        if action == "continue":
            if self._latest_ratify_proposal is not None:
                await self._route_delegated_decision("continue")
            return False

        if action == "ratify":
            if self._latest_ratify_proposal is None:
                logger.warning("ratify command with no pending proposal")
                return False
            await self._route_delegated_decision("ratify")
            return False

        if action == "goal":
            text = cmd.get("text", "")
            input_text = f"goal: {text}"
            command = parse_command(input_text)
            for role, act, message in command.to_messages(self._latest_ratify_proposal):
                await self._send_frame(role, act, message)
            return False

        if action == "scaffold":
            text = cmd.get("text", "")
            if self._latest_ratify_proposal is not None:
                await self._route_delegated_decision("scaffold", text)
            else:
                # Non-gated scaffold (no pending decision): submit to trainee
                # directly, preserving the existing behaviour.
                command = parse_command(f"scaffold:{text}")
                for role, act, message in command.to_messages(self._latest_ratify_proposal):
                    await self._send_frame(role, act, message)
            return False

        if action == "guidance":
            text = cmd.get("text", "")
            command = parse_command(text)
            for role, act, message in command.to_messages(self._latest_ratify_proposal):
                await self._send_frame(role, act, message)
            return False

        # All other actions: start, stop, pause, resume, restart, save, load
        command = parse_command(action)
        for role, act, message in command.to_messages(self._latest_ratify_proposal):
            await self._send_frame(role, act, message)
        return False

    async def _route_delegated_decision(self, decision: str, text: str = "") -> None:
        """Route a supervisor decision to the Trainer for a pending ratify_request.

        In delegated mode the Trainer gates the run on each ratify_request
        and holds further events until it receives this decision
        (``supervisor_decision`` action). The Trainer applies the
        countersign/submit itself, so we address the TRAINER role (not the
        trainee) and clear the pending proposal afterward. Spec ref:
        specs/reactive-delegation.md §Supervisor Answers (RD-9/10/11).
        """
        proposal = self._latest_ratify_proposal
        payload: dict[str, Any] = {"decision": decision, "proposal": proposal}
        if text:
            payload["text"] = text
        await self._send_frame(TRAINER_ROLE, "supervisor_decision", payload)
        self._latest_ratify_proposal = None

    async def _send_frame(self, role: str, action: str, message: Any) -> None:
        """Send a single JSON frame via the WebSocket."""
        if not self._connected or self._ws is None:
            return
        frame = {"role": role, "action": action, "message": message}
        raw = json.dumps(frame)
        await self._ws.send(raw)

    async def _handle_disconnect(self) -> None:
        """Handle unexpected WebSocket disconnect."""
        self._seq += 1
        self._append_event({"seq": self._seq, "type": "disconnected"})
        self._write_status(state="errored")
        self._connected = False


# -- Standalone entry point -------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-tune CLI Supervisor")
    parser.add_argument("--session-dir", required=True, help="Path to session directory")
    args = parser.parse_args()

    supervisor = CLISupervisor(args.session_dir)
    asyncio.run(supervisor.run())
