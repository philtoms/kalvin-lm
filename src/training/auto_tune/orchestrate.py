"""Orchestration functions for driving an auto-tune session.

Pure functions that let pi drive an auto-tune session through its file-based
protocol: write commands to ``cmd.json``, read events from ``events.jsonl``,
and poll for new events.  These are the core operations behind the ``send``,
``events``, ``step``, and ``status`` CLI subcommands.

Spec ref: specs/auto-tune.md §CLI Subcommands, §Command Frame, §Event Frame,
          §Status Object
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from training.auto_tune.session import SessionDir

# Run summary
#
# A run produces two artefacts: the event stream (``events.jsonl``) and the
# persisted trainer state (``<curriculum>.json``). ``summarize`` aggregates
# these into a single verdict — ``run-summary.json`` — that tells the agent
# whether the run achieved its goal and, if not, where to look. It is the
# auto-tune arbiter: the signal the agent reads instead of being told to
# "reason carefully" (see SKILL.md §Pi-in-the-Loop Model).
#
# The summary is produced from file state only, so it can be run after any
# run (live or deadlocked) without touching the harness process.

# send_command


def send_command(session_dir: SessionDir, command_json: dict) -> None:
    """Write a command to the session's ``cmd.json`` file.

    Overwrites any existing file and returns immediately (no blocking).

    Args:
        session_dir: Bound session directory.
        command_json: Command payload to write (e.g. ``{"action": "start"}``).
    """
    session_dir.cmd_path.write_text(
        json.dumps(command_json, indent=2) + "\n",
        encoding="utf-8",
    )


# read_events


def read_events(session_dir: SessionDir, after_seq: int = -1) -> list[dict]:
    """Read events from the session's ``events.jsonl``, filtered by seq.

    Parses each line as JSON and returns entries where ``seq > after_seq``,
    preserving file order.  Blank and malformed lines are silently skipped.
    If ``events.jsonl`` does not exist, returns an empty list.

    Args:
        session_dir: Bound session directory.
        after_seq: Only return events with ``seq`` greater than this value.
            Defaults to ``-1`` (return all events).

    Returns:
        List of event dicts ordered by file position.
    """
    events_path: Path = session_dir.events_path
    if not events_path.exists():
        return []

    results: list[dict] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("seq", -1) > after_seq:
            results.append(entry)
    return results


# read_status


def read_status(session_dir: SessionDir) -> dict:
    """Read and parse the session's ``status.json``.

    Args:
        session_dir: Bound session directory.

    Returns:
        The parsed status dict.

    Raises:
        FileNotFoundError: If ``status.json`` does not exist.
    """
    status_path: Path = session_dir.status_path
    return json.loads(status_path.read_text(encoding="utf-8"))


# step

_POLL_INTERVAL = 0.1  # seconds between polls

# A connected run idle longer than this while holding unsatisfied work is
# treated as stalled (frozen), not incomplete (busy). Long enough to ride
# out a cogitation drain or a slow rationalise; short enough that the agent
# does not waste a full ``step`` timeout churning a dead stream.
STALL_THRESHOLD = 15.0  # seconds


def step(
    session_dir: SessionDir,
    command_json: dict,
    *,
    timeout: float = 30.0,
) -> list[dict]:
    """Send a command and block until at least one new event appears.

    Writes *command_json* to ``cmd.json``, reads ``last_event_seq`` from
    ``status.json``, then polls ``events.jsonl`` until at least one event
    with ``seq > last_event_seq`` is found.

    Args:
        session_dir: Bound session directory.
        command_json: Command payload to send.
        timeout: Maximum seconds to wait for a new event (default 30).

    Returns:
        List of new event dicts (``seq > last_event_seq``).

    Raises:
        FileNotFoundError: If ``status.json`` does not exist on first read.
        TimeoutError: If no new event appears within *timeout* seconds.
    """
    send_command(session_dir, command_json)

    # Raises FileNotFoundError if status.json is missing.
    status = read_status(session_dir)
    last_event_seq: int = status.get("last_event_seq", -1)

    deadline = time.monotonic() + timeout
    while True:
        new_events = read_events(session_dir, after_seq=last_event_seq)
        if new_events:
            return new_events
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"No new event after {timeout}s (waiting for seq > {last_event_seq})"
            )
        time.sleep(_POLL_INTERVAL)


# summarize


def _read_trainer_state(session_dir: SessionDir) -> dict | None:
    """Best-effort read of the persisted trainer state (``<curriculum>.json``).

    The state file lives at the curriculum path with ``.json`` swapped in
    (see ``trainer_factory`` in ``training.harness.__main__``). Returns
    ``None`` when absent or unreadable — the summary degrades to
    event-derived signals only.
    """
    try:
        config = json.loads(session_dir.config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    curriculum = config.get("curriculum")
    if not curriculum:
        return None
    state_path = session_dir.root / Path(curriculum).with_suffix(".json")
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _significance_level(raw: int) -> str:
    """Classify a raw significance byte into an S1–S4 band.

    Mirrors ``BandLayout.classify`` (``kalvin.significance``) without
    importing the kernel — the summary must stay pure/file-based. The
    terminal byte (``0x00``) is S4; otherwise the band is the high nibble
    of the upper byte.
    """
    if raw == 0:
        return "S4"
    # Significance bytes are laid out as S1>S2>S3>S4 from high to low
    # within the upper byte; classify by the most-significant set band.
    if raw >= 0xC0:
        return "S1"
    if raw >= 0x80:
        return "S2"
    if raw >= 0x40:
        return "S3"
    return "S4"


def _infer_outcome(
    events: list[dict],
    status: dict | None,
    trainer_state: dict | None,
    last_event_age: float | None = None,
) -> tuple[str, str]:
    """Infer the run outcome and a one-line diagnosis pointer.

    Returns ``(outcome, diagnosis)``. Outcomes:
      - ``crashed``           — an error/traceback surfaced in the stream.
      - ``supervisor-stalled`` — a ratify_request sat unanswered (the
        supervisor role did not participate).
      - ``stalled``            — the run is connected but frozen: it holds
        submitted-but-unsatisfied work and has produced no event for longer
        than ``STALL_THRESHOLD``. The trainer's satisfaction accounting has
        deadlocked; driving it further will not move it.
      - ``deadlocked``         — the run ended with unsatisfied entries and
        no terminal completion.
      - ``completed``          — a terminal progress/complete event fired.
      - ``incomplete``        — the run has not ended and is still moving;
        caller should not yet summarise.
    """
    # Walk the stream once, gathering the signals that decide outcome.
    saw_complete = False
    last_progress_status: str | None = None
    last_lesson_complete: int = 0  # lessons_completed from the last such event
    ratify_requests = 0
    last_was_ratify = False
    saw_error = False

    for ev in events:
        etype = ev.get("type")
        if etype == "error":
            saw_error = True
        if etype == "progress":
            last_progress_status = ev.get("status")
            if ev.get("status") == "complete":
                saw_complete = True
            if ev.get("status") in ("complete", "lesson_complete"):
                last_lesson_complete = ev.get(
                    "lessons_completed", last_lesson_complete
                )
        if etype == "ratify_request":
            ratify_requests += 1
            last_was_ratify = True
        elif etype in ("rationalise", "connected", "disconnected"):
            last_was_ratify = False

    # A live run (supervisor still connected, no terminal event) is not
    # summarisable — unless it has stalled on an unanswered decision, which
    # the check below catches first.
    connected = bool(status and status.get("connected"))
    terminal_stream = bool(events and events[-1].get("type") == "disconnected")

    if saw_error:
        return (
            "crashed",
            "An error surfaced in the event stream. Reproduce with a minimal "
            "test, fix, and re-run before interpreting anything else.",
        )

    # Supervisor stall: the last decision event was a ratify_request that
    # never received a non-continue answer (status still awaiting a command,
    # or the stream ended on one). This is the agent-abdication signal.
    state = status.get("state") if status else None
    if ratify_requests > 0 and (
        last_was_ratify or state == "waiting_for_command"
    ):
        return (
            "supervisor-stalled",
            f"{ratify_requests} ratify_request(s) emitted; the last decision event "
            "was unanswered. The supervisor role did not enact a decision — see "
            "SKILL.md §Pi-in-the-Loop Model.",
        )

    # A live run (still connected, no terminal event, no pending decision)
    # is either genuinely in progress or frozen. Distinguish them: a run
    # that has submitted work it cannot satisfy and has gone quiet is
    # *stalled*, not *incomplete* — the trainer's satisfaction accounting
    # has deadlocked (e.g. entries whose rationalise events never arrive),
    # and driving it further (``step``) only re-polls a stream that will
    # never move. Surface the freeze so the agent stops churning and
    # diagnoses the model gap instead.
    if connected and not terminal_stream and not saw_complete:
        return _classify_live_run(status, trainer_state, last_event_age)

    if saw_complete:
        return (
            "completed",
            f"Run completed: {last_lesson_complete} lesson(s) satisfied. "
            "Read the significance profile to judge whether the goal was met.",
        )

    # No terminal completion and the stream ended (disconnected) or went
    # idle. If the trainer state shows unsatisfied entries, this is the
    # deadlock outcome — the satisfaction model has no path for them.
    submitted: list = (trainer_state or {}).get("submitted", [])
    satisfied: list = (trainer_state or {}).get("satisfied", [])
    if trainer_state is not None and len(submitted) > len(satisfied):
        return (
            "deadlocked",
            f"{len(submitted) - len(satisfied)} of {len(submitted)} entries were "
            "submitted but never satisfied, and the run ended without completion. "
            "The satisfaction model has no path for these entries — see "
            "``src/training/trainer/reactor.py`` (``_auto_countersign``) and the "
            "owning spec for what \"resolved\" means for this entry type.",
        )

    # No trainer state to confirm deadlock, but no completion either: the
    # run ended inconclusively. Surface as deadlocked with a softer pointer.
    return (
        "deadlocked",
        "The run ended without a completion event and with S2/S3 proposals "
        "that were never resolved. Inspect events.jsonl for the last lesson's "
        "entry type and whether any proposal could satisfy it.",
    )


def _parse_iso(value: object) -> datetime | None:
    """Parse an ISO-8601 timestamp (naive or aware) into an aware datetime.

    Returns ``None`` on any parse failure — callers treat a missing
    timestamp as "unknown liveness" rather than a stall.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _classify_live_run(
    status: dict | None,
    trainer_state: dict | None,
    last_event_age: float | None,
) -> tuple[str, str]:
    """Classify a connected, non-terminal run as ``stalled`` or ``incomplete``.

    A run is *stalled* when it holds submitted-but-unsatisfied work **and**
    has been idle longer than ``STALL_THRESHOLD`` (or liveness is unknown —
    an older supervisor that never wrote ``last_event_at``). The satisfaction
    gap is read from trainer state when available; when the state file is
    absent or unreadable, any connected run idle past the threshold is
    treated as stalled, since a busy run produces events.

    Otherwise the run is *incomplete* — still moving, keep driving.
    """
    submitted: list = (trainer_state or {}).get("submitted", [])
    satisfied: list = (trainer_state or {}).get("satisfied", [])
    has_gap = trainer_state is not None and len(submitted) > len(satisfied)
    idle = last_event_age is None or last_event_age >= STALL_THRESHOLD

    if idle and (has_gap or trainer_state is None):
        gap = (
            f"{len(submitted) - len(satisfied)} of {len(submitted)} entries "
            "submitted but unsatisfied, "
            if has_gap
            else ""
        )
        age = (
            f"{last_event_age:.0f}s"
            if isinstance(last_event_age, (int, float))
            else "unknown liveness"
        )
        return (
            "stalled",
            f"Connected but frozen — no event for {age}. {gap}"
            "The trainer's satisfaction accounting has deadlocked: driving it "
            "further will not produce events. Stop the run, read the harness "
            "log and the last lesson's compiled entries, and diagnose why "
            "submitted work is not being rationalised. See SKILL.md §Stalled runs.",
        )
    return "incomplete", "Run is still in progress — no verdict yet."


def summarize(session_dir: SessionDir, *, write: bool = True) -> dict:
    """Aggregate a run into a verdict the agent reads to decide what's next.

    Produces ``run-summary.json`` in the session directory (unless
    ``write=False``) and returns the summary dict. Pure file aggregation —
    safe to run after any run, live or deadlocked.

    The summary is the auto-tune arbiter: it carries the *termination*
    signal (``outcome``) and the *diagnosis* signal (``significance``
    profile + ``diagnosis`` pointer). The agent chases it instead of being
    exhorted to "reason carefully" each turn.
    """
    events = read_events(session_dir)
    try:
        status = read_status(session_dir)
    except FileNotFoundError:
        status = None
    trainer_state = _read_trainer_state(session_dir)

    # Liveness: seconds since the supervisor appended its last event. A
    # connected run that stops producing events while holding unsatisfied
    # work is frozen, not busy (see ``_classify_live_run``).
    last_event_at = _parse_iso(status.get("last_event_at")) if status else None
    last_event_age = (
        (datetime.now(timezone.utc) - last_event_at).total_seconds()
        if last_event_at is not None
        else None
    )

    # Significance profile from rationalise events.
    bands: dict[str, int] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0}
    ground = 0
    frames = 0
    proposals_emitted = 0
    ratify_requests = 0
    for ev in events:
        if ev.get("type") != "rationalise":
            if ev.get("type") == "ratify_request":
                ratify_requests += 1
            continue
        kind = ev.get("kind")
        if kind == "ground":
            ground += 1
        elif kind == "frame":
            frames += 1
        sig = ev.get("significance")
        raw = sig.get("raw", 0) if isinstance(sig, dict) else 0
        bands[_significance_level(int(raw))] += 1
        # A frame is a proposal; grounds are auto-resolutions, not proposals.
        if kind == "frame":
            proposals_emitted += 1

    # Entries resolved, from trainer state when available.
    submitted: list = (trainer_state or {}).get("submitted", [])
    satisfied: list = (trainer_state or {}).get("satisfied", [])
    entries_total = len(submitted) if trainer_state is not None else None
    entries_satisfied = len(satisfied) if trainer_state is not None else None
    entries_deadlocked = (
        entries_total - entries_satisfied
        if entries_total is not None
        else None
    )

    outcome, diagnosis = _infer_outcome(events, status, trainer_state, last_event_age)

    # Ratifications: ratify_requests that received a non-continue answer are
    # not distinguishable from the stream alone (the answer is a bus message,
    # not an event). We approximate depth as proposals_emitted vs the count
    # of entries satisfied: a wide gap means proposals surfaced but nothing
    # closed them.
    proposals_ratified = None
    if entries_satisfied is not None:
        # Each satisfied non-S1 entry corresponds to one ratified proposal;
        # S1 entries (grounds) were auto-resolved, not ratified.
        proposals_ratified = max(0, entries_satisfied - ground)

    summary = {
        "outcome": outcome,
        "diagnosis": diagnosis,
        "significance": {
            "S1": bands["S1"],
            "S2": bands["S2"],
            "S3": bands["S3"],
            "S4": bands["S4"],
            "ground": ground,
            "frame": frames,
        },
        "proposals_emitted": proposals_emitted,
        "proposals_ratified": proposals_ratified,
        "ratify_requests": ratify_requests,
        "entries_total": entries_total,
        "entries_satisfied": entries_satisfied,
        "entries_deadlocked": entries_deadlocked,
        "events_total": len(events),
    }

    if write:
        summary_path = session_dir.summary_path
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    return summary
