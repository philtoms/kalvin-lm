"""Argparse-based CLI for auto-tune workflow.

Provides 12 subcommands for session management, process lifecycle,
and the send/events/step interaction pattern.  Each subcommand resolves
``--session`` to a :class:`SessionDir` instance, then delegates to the
appropriate handler module.

Usage::

    python -m participants.auto_tune init --session exp-1 --curriculum curricula/topic.md
    python -m participants.auto_tune status --session exp-1
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from participants.auto_tune import lifecycle
from participants.auto_tune import orchestrate
from participants.auto_tune import snapshots
from participants.auto_tune.session import SessionDir


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------


def _handle_init(args: argparse.Namespace) -> None:
    """Create a new auto-tune session via :meth:`SessionDir.init`."""
    sd = SessionDir.init(
        session=args.session,
        curriculum=args.curriculum,
        host=args.host,
        port=args.port,
    )
    print(f"Session '{args.session}' initialised at {sd.config_path.parent}")


def _handle_start_harness(args: argparse.Namespace) -> None:
    """Start the harness server as a background process."""
    sd: SessionDir = args._session_dir
    lifecycle.start_harness(sd.config_path.parent)


def _handle_stop_harness(args: argparse.Namespace) -> None:
    """Gracefully stop the harness server."""
    sd: SessionDir = args._session_dir
    lifecycle.stop_harness(sd.config_path.parent)


def _handle_start_supervisor(args: argparse.Namespace) -> None:
    """Start the CLI supervisor as a background process."""
    sd: SessionDir = args._session_dir
    lifecycle.start_supervisor(sd.config_path.parent)


def _handle_stop_supervisor(args: argparse.Namespace) -> None:
    """Stop the CLI supervisor process."""
    sd: SessionDir = args._session_dir
    lifecycle.stop_supervisor(sd.config_path.parent)


def _handle_send(args: argparse.Namespace) -> None:
    """Write a command to cmd.json and return immediately."""
    sd: SessionDir = args._session_dir
    try:
        command_json = json.loads(args.command_json)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in --command: {exc}", file=sys.stderr)
        sys.exit(1)
    orchestrate.send_command(sd, command_json)


def _handle_events(args: argparse.Namespace) -> None:
    """Print events from events.jsonl, optionally filtered by seq."""
    sd: SessionDir = args._session_dir
    events_list = orchestrate.read_events(sd, args.after)
    for event in events_list:
        print(json.dumps(event))


def _handle_step(args: argparse.Namespace) -> None:
    """Write command, block until next event appears, print it."""
    sd: SessionDir = args._session_dir
    try:
        command_json = json.loads(args.command_json)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in --command: {exc}", file=sys.stderr)
        sys.exit(1)
    new_events = orchestrate.step(sd, command_json)
    for event in new_events:
        print(json.dumps(event))


def _handle_status(args: argparse.Namespace) -> None:
    """Print the session status as formatted JSON."""
    sd: SessionDir = args._session_dir
    status_data = orchestrate.read_status(sd)
    print(json.dumps(status_data, indent=2))


def _handle_snapshot(args: argparse.Namespace) -> None:
    """Capture session state to runs/<n>/."""
    sd: SessionDir = args._session_dir
    snapshots.snapshot(sd)


def _handle_restore(args: argparse.Namespace) -> None:
    """Restore session state from a named run snapshot."""
    sd: SessionDir = args._session_dir
    snapshots.restore(sd, args.run)


def _handle_reset(args: argparse.Namespace) -> None:
    """Delete curriculum state, truncate events, optionally delete model."""
    sd: SessionDir = args._session_dir
    snapshots.reset(sd, fresh_model=args.fresh_model)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with 12 subcommands."""
    parser = argparse.ArgumentParser(
        prog="auto-tune",
        description="Auto-tune CLI — session management for autonomous training",
    )
    sub = parser.add_subparsers(dest="command")

    # -- init ---------------------------------------------------------------
    p_init = sub.add_parser("init", help="Create session directory, config, git branch")
    p_init.add_argument("--session", required=True, help="Session codename")
    p_init.add_argument("--curriculum", required=True, help="Path to curriculum markdown")
    p_init.add_argument("--host", default=None, help="Harness server host override")
    p_init.add_argument("--port", type=int, default=None, help="Harness server port override")
    p_init.set_defaults(func=_handle_init)

    # -- start-harness -------------------------------------------------------
    p_sh = sub.add_parser("start-harness", help="Start harness server")
    p_sh.add_argument("--session", required=True, help="Session codename")
    p_sh.set_defaults(func=_handle_start_harness)

    # -- stop-harness --------------------------------------------------------
    p_xh = sub.add_parser("stop-harness", help="Stop harness server")
    p_xh.add_argument("--session", required=True, help="Session codename")
    p_xh.set_defaults(func=_handle_stop_harness)

    # -- start-supervisor ----------------------------------------------------
    p_ss = sub.add_parser("start-supervisor", help="Start CLI supervisor")
    p_ss.add_argument("--session", required=True, help="Session codename")
    p_ss.set_defaults(func=_handle_start_supervisor)

    # -- stop-supervisor -----------------------------------------------------
    p_xs = sub.add_parser("stop-supervisor", help="Stop CLI supervisor")
    p_xs.add_argument("--session", required=True, help="Session codename")
    p_xs.set_defaults(func=_handle_stop_supervisor)

    # -- send ----------------------------------------------------------------
    p_send = sub.add_parser("send", help="Write command to cmd.json")
    p_send.add_argument("--session", required=True, help="Session codename")
    p_send.add_argument("--command", required=True, dest="command_json", help="JSON command string")
    p_send.set_defaults(func=_handle_send)

    # -- events --------------------------------------------------------------
    p_ev = sub.add_parser("events", help="Print events after given sequence")
    p_ev.add_argument("--session", required=True, help="Session codename")
    p_ev.add_argument("--after", type=int, default=0, help="Return events with seq > N (default: 0)")
    p_ev.set_defaults(func=_handle_events)

    # -- step ----------------------------------------------------------------
    p_step = sub.add_parser("step", help="Write command, block until next event")
    p_step.add_argument("--session", required=True, help="Session codename")
    p_step.add_argument("--command", required=True, dest="command_json", help="JSON command string")
    p_step.set_defaults(func=_handle_step)

    # -- status --------------------------------------------------------------
    p_stat = sub.add_parser("status", help="Print session status")
    p_stat.add_argument("--session", required=True, help="Session codename")
    p_stat.set_defaults(func=_handle_status)

    # -- snapshot ------------------------------------------------------------
    p_snap = sub.add_parser("snapshot", help="Capture state to runs/<n>/")
    p_snap.add_argument("--session", required=True, help="Session codename")
    p_snap.set_defaults(func=_handle_snapshot)

    # -- restore -------------------------------------------------------------
    p_rest = sub.add_parser("restore", help="Restore state from a run snapshot")
    p_rest.add_argument("--session", required=True, help="Session codename")
    p_rest.add_argument("--run", required=True, type=int, help="Run number to restore")
    p_rest.set_defaults(func=_handle_restore)

    # -- reset ---------------------------------------------------------------
    p_rst = sub.add_parser("reset", help="Delete curriculum state, truncate events")
    p_rst.add_argument("--session", required=True, help="Session codename")
    p_rst.add_argument("--fresh-model", action="store_true", default=False,
                        help="Also delete the Kalvin model file")
    p_rst.set_defaults(func=_handle_reset)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Parse args, validate session, and dispatch to handler."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Non-init commands require an existing session directory
    if args.command != "init":
        try:
            args._session_dir = SessionDir.load(args.session)
        except FileNotFoundError:
            print(
                f"Error: session '{args.session}' not found. "
                f"Run 'auto-tune init --session {args.session}' first.",
                file=sys.stderr,
            )
            sys.exit(1)

    args.func(args)
