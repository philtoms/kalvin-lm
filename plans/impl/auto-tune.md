# Auto-Tune Implementation Plan

**Status:** not started
**Spec refs:** `specs/auto-tune.md`

## Spec References

- `@specs/auto-tune.md` — AT-1 through AT-20
- `@specs/harness-server.md` — harness configuration, participant architecture
- `@specs/curriculum.md` — curriculum state persistence format
- `@specs/kscript.md` — decompiler for event enrichment

## Build Phases

### Phase 1: Session Management (CLI skeleton)

The session directory, config, and CLI framework. No supervisor yet — just `init` and the file management commands.

### Phase 2: CLI Supervisor

The WebSocket client participant. Connects to harness, reads cmd.json, writes events.jsonl, per-event blocking.

### Phase 3: Process Lifecycle and Orchestration

`start-harness`, `stop-harness`, `start-supervisor`, `stop-supervisor`, `step`.

### Phase 4: Snapshot, Restore, Reset

Run management commands for the tuning loop.

## Implementation Tasks

### Task 1: Session module (src/participants/auto_tune/session.py)

- **Spec ref:** @specs/auto-tune §Session Configuration, §Session Initialisation, §Directory Structure
- **Test mapping:** AT-1, AT-2, AT-3
- **Details:**
  - `SessionConfig` dataclass: session, curriculum, harness_url, model_path, run_counter, created_from_branch, created_from_commit
  - `SessionDir` class managing the session directory:
    - `init(session, curriculum, host, port)` — create directory, write config.json, create git branch, checkout
    - `load(session)` — load config from existing directory
    - Properties: `config_path`, `cmd_path`, `status_path`, `events_path`, `runs_dir`
  - Read `harness.yaml` for default host/port
  - Derive model_path from harness config (same logic as `__main__.py`)
  - Record `git rev-parse HEAD` and current branch at init time

### Task 2: CLI entry point (src/participants/auto_tune/__main__.py, cli.py)

- **Spec ref:** @specs/auto-tune §CLI Subcommands
- **Test mapping:** AT-1
- **Details:**
  - `argparse` with subcommands: `init`, `start-harness`, `stop-harness`, `start-supervisor`, `stop-supervisor`, `send`, `events`, `step`, `status`, `snapshot`, `restore`, `reset`
  - Each subcommand delegates to a handler function
  - All subcommands take `--session <name>` as required argument
  - `__main__.py` is a thin wrapper: `from .cli import main; main()`

### Task 3: Event enrichment (src/participants/auto_tune/events.py)

- **Spec ref:** @specs/auto-tune §Event Frame, §Significance Object, §KLine Display Object, §Event Enrichment
- **Test mapping:** AT-11
- **Details:**
  - `enrich_event(raw_frame: dict) -> dict` — transform raw harness WebSocket frame into auto-tune event format
  - Significance calculation: `raw / D_MAX` → normalised, classify S1–S4
  - KLine decompilation: use `src/kscript/decompiler.py` to produce source strings
  - Map harness action names to auto-tune event types:
    - `"progress"` → `"progress"` (with field renaming)
    - `"event"` with `kind: "ground"/"frame"` → `"rationalise"`
    - `"ratify_request"` → `"ratify_request"`
    - `"notify"` → `"escalation"`
  - Add monotonic `seq` counter

### Task 4: CLI Supervisor (src/participants/auto_tune/supervisor.py)

- **Spec ref:** @specs/auto-tune §Per-Event Blocking Model, §Command Processing, §Run Completion, §Error Handling
- **Test mapping:** AT-6, AT-7, AT-8, AT-9, AT-10, AT-12, AT-13
- **Details:**
  - `CLISupervisor` class:
    - Constructor: takes session dir path, connects WebSocket, registers as `supervisor` role
    - Main loop: `run()` — receive one WS message → enrich → write to events.jsonl → update status → poll cmd.json → process command → repeat
    - Command processing: map simplified command to `parse_command()` result, send via WebSocket
    - `continue` command: no-op, resume loop
    - `ratify` command: send countersign for latest buffered proposal
    - `shutdown` command: write disconnected event, close WebSocket, exit
    - Buffer latest `ratify_request` proposal
    - On run complete: set status to `run_complete`, continue blocking for commands
    - On unexpected disconnect: write disconnected event, set status to `errored`, exit
  - Uses `HarnessClient`-style WebSocket connection (adapted from `tui_client.py`)
  - `status.json` updated atomically on every state change
  - `cmd.json` polling: check file existence, read, delete, process. Poll interval ~100ms.

### Task 5: Process lifecycle commands (src/participants/auto_tune/lifecycle.py)

- **Spec ref:** @specs/auto-tune §Harness Lifecycle, §Supervisor Lifecycle, §Error Handling
- **Test mapping:** AT-4, AT-5, AT-14, AT-20
- **Details:**
  - `start_harness(session_dir)`:
    - Run `python -m harness --config harness.yaml` as background process
    - Record PID in session dir
    - Poll WebSocket port (from config) until connection accepted
    - Return
  - `stop_harness(session_dir)`:
    - Read PID, send SIGTERM
    - Wait up to 5s, SIGKILL if still alive
    - Clean up PID file
  - `start_supervisor(session_dir)`:
    - Run `python -m participants.auto_tune.supervisor --session <path>` as background process
    - Record PID in session dir
    - Poll `status.json` until `connected: true`
    - Return
  - `stop_supervisor(session_dir)`:
    - Write `{"action": "shutdown"}` to `cmd.json`
    - Wait up to 5s for process exit, SIGKILL if needed
    - Clean up PID file

### Task 6: Orchestration commands (src/participants/auto_tune/orchestrate.py)

- **Spec ref:** @specs/auto-tune §CLI Subcommands
- **Test mapping:** AT-14, AT-15
- **Details:**
  - `send_command(session_dir, command_json)`:
    - Write command to `cmd.json`
    - Return immediately
  - `read_events(session_dir, after_seq)`:
    - Read `events.jsonl`, parse lines, filter `seq > after_seq`
    - Print to stdout as JSONL
  - `step(session_dir, command_json)`:
    - Write command to `cmd.json`
    - Read current `last_event_seq` from `status.json`
    - Poll `events.jsonl` until a line with `seq > last_event_seq` appears
    - Print the new event(s) to stdout
  - `read_status(session_dir)`:
    - Print `status.json` to stdout

### Task 7: Snapshot and restore (src/participants/auto_tune/snapshots.py)

- **Spec ref:** @specs/auto-tune §Snapshot and Restore
- **Test mapping:** AT-16, AT-17
- **Details:**
  - `snapshot(session_dir)`:
    - Increment `run_counter` in `config.json`
    - Create `runs/<n>/` directory
    - Copy curriculum state file → `runs/<n>/state.json`
    - Copy `events.jsonl` → `runs/<n>/events.jsonl`
    - Copy model file (if exists) → `runs/<n>/model.bin`
    - Write `runs/<n>/meta.json` with git HEAD, branch, dirty status, timestamp
  - `restore(session_dir, run_number)`:
    - Verify harness and supervisor are stopped (check PIDs/status)
    - Copy `runs/<n>/state.json` back to curriculum state file location
    - Copy `runs/<n>/model.bin` back to model file location (if exists)

### Task 8: Reset (src/participants/auto_tune/snapshots.py or session.py)

- **Spec ref:** @specs/auto-tune §Reset
- **Test mapping:** AT-18, AT-19
- **Details:**
  - `reset(session_dir, fresh_model=False)`:
    - Derive curriculum state file path from `config.json` curriculum path (change extension to `.json`)
    - Delete curriculum state file
    - Truncate `events.jsonl` to empty
    - If `fresh_model`: delete model file from path in `config.json`

### Task 9: Supervisor standalone entry (src/participants/auto_tune/supervisor.py)

- **Spec ref:** @specs/auto-tune §Supervisor Lifecycle
- **Test mapping:** AT-6
- **Details:**
  - `if __name__` block or `__main__` support for running supervisor directly:
    `python -m participants.auto_tune.supervisor --session-dir <path>`
  - Reads config from session dir
  - Constructs `CLISupervisor` and calls `run()`

### Task 10: Wire CLI subcommands (src/participants/auto_tune/cli.py)

- **Test mapping:** all AT IDs
- **Details:**
  - Map each argparse subcommand to its handler:
    - `init` → `SessionDir.init()`
    - `start-harness` → `start_harness()`
    - `stop-harness` → `stop_harness()`
    - `start-supervisor` → `start_supervisor()`
    - `stop-supervisor` → `stop_supervisor()`
    - `send` → `send_command()`
    - `events` → `read_events()`
    - `step` → `step()`
    - `status` → `read_status()`
    - `snapshot` → `snapshot()`
    - `restore` → `restore()`
    - `reset` → `reset()`

## Test Mapping

| Spec ID | Test file | Test function | Status |
|---------|-----------|---------------|--------|
| AT-1 | test_auto_tune.py | test_init_creates_session_directory | ☐ |
| AT-2 | test_auto_tune.py | test_init_creates_git_branch | ☐ |
| AT-3 | test_auto_tune.py | test_init_records_config | ☐ |
| AT-4 | test_auto_tune.py | test_start_harness_starts_and_waits | ☐ |
| AT-5 | test_auto_tune.py | test_stop_harness_graceful_shutdown | ☐ |
| AT-6 | test_auto_tune.py | test_supervisor_connects_and_registers | ☐ |
| AT-7 | test_auto_tune.py | test_supervisor_per_event_blocking | ☐ |
| AT-8 | test_auto_tune.py | test_continue_command_noop | ☐ |
| AT-9 | test_auto_tune.py | test_ratify_sends_countersign | ☐ |
| AT-10 | test_auto_tune.py | test_shutdown_disconnects_cleanly | ☐ |
| AT-11 | test_auto_tune.py | test_event_enrichment_decompiled_source | ☐ |
| AT-12 | test_auto_tune.py | test_run_complete_does_not_exit | ☐ |
| AT-13 | test_auto_tune.py | test_unexpected_disconnect_errored | ☐ |
| AT-14 | test_auto_tune.py | test_step_blocks_until_event | ☐ |
| AT-15 | test_auto_tune.py | test_events_after_seq_filter | ☐ |
| AT-16 | test_auto_tune.py | test_snapshot_captures_state_events_model_git | ☐ |
| AT-17 | test_auto_tune.py | test_restore_reinstates_state_and_model | ☐ |
| AT-18 | test_auto_tune.py | test_reset_deletes_state_truncates_events | ☐ |
| AT-19 | test_auto_tune.py | test_reset_fresh_model_deletes_model | ☐ |
| AT-20 | test_auto_tune.py | test_lifecycle_pid_management_and_timeouts | ☐ |

## Design Decisions

1. **File-mediated over stdin/stdout** — pi's bash tool cannot maintain persistent conversations. See ADR-0003.
2. **Per-event blocking over batched events** — gives pi maximum granularity of training observability.
3. **WebSocket client, not embedded** — pi owns process lifecycle. Supervisor is just another participant on the bus.
4. **Reuse `parse_command()`** — the CLI supervisor uses the same command parser as TUI and Slack, ensuring consistent behaviour.
5. **Supervisor exits on error, does not auto-reconnect** — pi decides recovery strategy.
6. **Run numbering is auto-incremented** — simpler than manual naming, less room for error.

## Status

Plan complete. No implementation started.
