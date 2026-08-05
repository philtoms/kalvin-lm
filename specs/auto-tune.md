# Auto-Tune Specification

## Overview

Auto-tune is a CLI tool and supervisor participant that enables an LLM coding agent (pi) to autonomously control training sessions, observe results, modify the codebase, and re-run — converging on a behavioural goal for Kalvin's rationalisation. The thing being tuned is Kalvin's significance model: pi observes how the reactor/cogitator/rationaliser actually behave under a curriculum, then changes the model code (`expand()`, `significance.py`, the rationaliser) together with the owning spec that defines its intended semantics, and re-runs to confirm. Incidental codebase goals are also valid. Pi owns the full lifecycle: it starts the harness server, starts the CLI supervisor, drives training via commands, reads events, edits code, snapshots state, resets, and repeats.

The system has two components:

- **Auto-tune CLI** (`python -m training.auto_tune`) — session management, process lifecycle, and the `step`/`send`/`events` commands.
- **CLI Supervisor** — a WebSocket client participant that connects to the harness, reads commands from a file, writes events to a file, and blocks per-event for maximum observability.

## Dependencies

- `src/training/harness/` — message bus, WebSocket protocol, harness server
- `src/training/supervisors/commands.py` — `parse_command()` for mapping simplified commands to bus messages
- `src/kalvin/significance.py` — `SIG8_MAX` for the 8-bit significance grade
- `specs/harness-server.md` — harness configuration and participant architecture
- `specs/supervisor-decision.md` — the decision contract the CLI supervisor participates in
- `specs/curriculum.md` — curriculum state persistence format

## Definitions

### Session Configuration

| Field               | Type          | Description                                      |
| ------------------- | ------------- | ------------------------------------------------ |
| session             | `str`         | Codename for the tuning session                  |
| curriculum          | `str`         | Path to curriculum markdown file                 |
| harness_url         | `str`         | WebSocket URL (default `ws://localhost:8765`)    |
| model_path          | `str \| None` | Path to Kalvin model file (for snapshot/restore) |
| run_counter         | `int`         | Number of snapshots taken                        |
| created_from_branch | `str`         | Branch that was current at `init` time           |
| created_from_commit | `str`         | Commit hash at `init` time                       |
| worktree_path       | `str`         | Absolute path to the session's git worktree      |

### Event Frame

Each line is a JSON object with a monotonic `seq` counter.

| Type             | Fields                                                                       | Description                                                                                                                                                                                                                       |
| ---------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `connected`      | `seq`                                                                        | Supervisor connected to harness                                                                                                                                                                                                   |
| `disconnected`   | `seq`                                                                        | Supervisor disconnected (planned or error)                                                                                                                                                                                        |
| `progress`       | `seq`, `status`, `lesson`, `lessons_total`, `lessons_completed`              | Training progress: `started`, `lesson_complete`, `complete`, `amended`, `polling_for_goal`, `ready`                                                                                                                               |
| `rationalise`    | `seq`, `kind`, `significance`, `query`, `proposal`                           | Kalvin rationalisation event (`ground` or `frame`), with decompiled source and significance breakdown                                                                                                                             |
| `ratify_request` | `seq`, `query`, `proposal`, `significance`, `misfit`, `curriculum_context` | A proposal the Trainer cannot auto-ratify, requiring a supervisor decision. Enriched with the misfit diagnosis and curriculum context (`@specs/supervisor-decision.md`). |
| `escalation`     | `seq`, `reason`, `detail`, `lesson_position`                                 | Trainer cannot make progress (`budget_exhaustion` or `low_confidence`) → [removed] — no escalation mechanism (`@specs/supervisor-decision.md` SD-3)                                                                                                                                                            |

### Significance Object

| Field      | Type    | Description                              |
| ---------- | ------- | ---------------------------------------- |
| raw        | `int`   | 8-bit significance byte (low byte of an int) |
| normalised | `float` | `byte / 0xFF` (0.0 S4 — 1.0 S1; trivial rescale for display) |
| level      | `str`   | `S1`, `S2`, `S3`, or `S4` (via `BandLayout.classify`) |

### KLine Display Object

| Field       | Type                                 | Description                                  |
| ----------- | ------------------------------------ | -------------------------------------------- |
| values      | `{signature: int, nodes: list[int]}` | The kline the model stored — judgement basis |
| for_display | `str`                                | Decompiled KScript label — display only      |

### Command Frame

A single JSON object, written by pi, consumed and deleted by the supervisor.

| Action     | Fields           | Description                                                                                                   |
| ---------- | ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `start`    | `action`         | Begin training session                                                                                        |
| `stop`     | `action`         | End training session                                                                                          |
| `pause`    | `action`         | Pause training                                                                                                |
| `resume`   | `action`         | Resume training                                                                                               |
| `restart`  | `action`         | Reset training state and restart                                                                              |
| `ratify`   | `action`         | Countersign latest pending proposal                                                                           |
| `save`     | `action`         | Persist Kalvin model                                                                                          |
| `load`     | `action`         | Load Kalvin model                                                                                             |
| `goal`     | `action`, `text` | Set training goal                                                                                             |
| `guidance` | `action`, `text` | Freeform guidance for the trainer                                                                             |
| `scaffold` | `action`, `text` | Submit reactive scaffolding KScript to Kalvin (supervisor decision — `@specs/supervisor-decision.md`) |
| `continue` | `action`         | No-op: acknowledge event, wait for next                                                                       |
| `shutdown` | `action`         | Graceful supervisor shutdown                                                                                  |

### Status Object

| Field          | Type             | Description                                                                                          |
| -------------- | ---------------- | ---------------------------------------------------------------------------------------------------- |
| pid            | `int`            | Supervisor process ID                                                                                |
| connected      | `bool`           | WebSocket connection state                                                                           |
| last_event_seq | `int`            | Sequence number of last written event                                                                |
| last_event_at  | `str`            | ISO timestamp of the last written event; the liveness heartbeat for stall detection (see §Run Summary) |
| last_command   | `object \| null` | Last consumed command                                                                                |
| state          | `str`            | `connecting`, `waiting_for_event`, `waiting_for_command`, `run_complete`, `shutting_down`, `errored` |
| started_at     | `str`            | ISO timestamp                                                                                        |

### Snapshot Metadata

| Field      | Type   | Description                                  |
| ---------- | ------ | -------------------------------------------- |
| run        | `int`  | Run number                                   |
| timestamp  | `str`  | ISO timestamp                                |
| git_head   | `str`  | Commit hash at snapshot time                 |
| git_branch | `str`  | Branch at snapshot time                      |
| git_dirty  | `bool` | Whether working tree had uncommitted changes |

### Run Summary Object

Written to `run-summary.json` by `summary`. The auto-tune arbiter: the
machine-readable signal the agent reads to decide whether a run achieved
its goal and, if not, where to look — replacing per-turn supervision prose
(see SKILL.md §Pi-in-the-Loop Model).

| Field                | Type             | Description                                                                                     |
| -------------------- | ---------------- | ----------------------------------------------------------------------------------------------- |
| outcome              | `str`            | `completed`, `deadlocked`, `supervisor-stalled`, `stalled`, `crashed`, or `incomplete` (see §Run Summary)  |
| diagnosis            | `str`            | One- to two-sentence pointer naming the file/concept to inspect first (diagnosis, not procedure) |
| significance         | `object`         | S1–S4 band histogram over `rationalise` events, plus `ground` and `frame` counts                |
| proposals_emitted    | `int`            | Count of `frame` rationalise events (expansion proposals surfaced)                              |
| proposals_ratified   | `int \| null`    | Entries satisfied minus grounds (approximation); `null` when trainer state is absent            |
| ratify_requests      | `int`            | Count of `ratify_request` events (decisions surfaced to the supervisor)                         |
| entries_total        | `int \| null`    | Submitted entry count from trainer state; `null` when state is absent                           |
| entries_satisfied    | `int \| null`    | Satisfied entry count from trainer state                                                        |
| entries_deadlocked   | `int \| null`    | `entries_total − entries_satisfied`                                                             |
| events_total         | `int`            | Total event count in the stream                                                                 |

### Session Layout

Session artefacts are persisted to files inside the session's git worktree. The concrete directory layout and the concept→file mapping (which concept names which real file) are documented in `@plans/impl/auto-tune-session-layout.md` — file structure and code locations are Plan-owned per Structural Rule #5.

## API

### CLI Subcommands

| Command                                                                         | Description                                                            |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `auto-tune init --session <name> --curriculum <path> [--host <h>] [--port <p>]` | Create worktree, session directory, session configuration, git branch  |
| `auto-tune teardown --session <name>`                                           | Remove worktree and delete branch                                      |
| `auto-tune start-harness --session <name>`                                      | Start harness server, record PID, wait for ready                       |
| `auto-tune stop-harness --session <name>`                                       | Graceful SIGTERM to harness, cleanup PID                               |
| `auto-tune start-supervisor --session <name>`                                   | Start CLI supervisor, record PID, wait for connected                   |
| `auto-tune stop-supervisor --session <name>`                                    | Send shutdown command, wait for exit (SIGKILL on timeout)              |
| `auto-tune send --session <name> --command <json>`                              | Write the command to the command file, return immediately              |
| `auto-tune events --session <name> [--after <seq>]`                             | Print event-stream records after the given sequence (default: all)     |
| `auto-tune step --session <name> --command <json>`                              | Write command, block until next event appears, print it                |
| `auto-tune status --session <name>`                                             | Print the status object                                                |
| `auto-tune summary --session <name>`                                          | Aggregate the current run into a verdict (`run-summary.json`) — the arbiter |
| `auto-tune snapshot --session <name>`                                           | Capture state, events, model, git metadata to the run directory        |
| `auto-tune restore --session <name> --run <n>`                                  | Restore curriculum state and model from run snapshot                   |
| `auto-tune reset --session <name> [--fresh-model]`                              | Delete curriculum state file, truncate events, optionally delete model |

## Behavioural Rules

### Session Initialisation

1. `init` creates the session directory and all supporting files inside a git worktree.
2. `init` creates a git worktree at `.worktrees/auto-tune/<session>/` with branch `auto-tune/<session>` from the current HEAD. The main repo stays on its current branch.
3. `init` records the current branch name, commit hash, and worktree path in the session configuration.
4. `init` reads the project harness config to derive default host/port, overridable via `--host`/`--port`.
5. The session configuration stores the resolved `harness_url` as `ws://<host>:<port>`.
6. `init` records the Kalvin model path (derived from harness config) in the session configuration.

### Harness Lifecycle

7. `start-harness` starts the harness server as a background process, configured from the per-session harness config.
   7a. `start-harness` generates that per-session harness config from the project harness config. Auto-tune does not configure an LLMSupervisor participant, so pi (the CLI supervisor) is the sole decider for the session (`@specs/supervisor-decision.md`).
8. `start-harness` records the PID in the session directory.
9. `start-harness` polls the WebSocket port until it accepts connections, then returns.
   9a. Before polling, `start-harness` kills any orphan process bound to the WebSocket port (not just the PID-file process). This closes the gap where a stale or missing PID file leaves an orphaned harness holding the port: without it the new harness fails to bind with `EADDRINUSE` and the readiness poll masks the failure by connecting to the orphan.
   9b. The readiness poll fails fast if the spawned harness exits before becoming ready, rather than polling until the timeout (which would again mask a bind crash behind a pre-existing listener).
10. `stop-harness` sends SIGTERM to the harness PID, waits for exit (SIGKILL on 5s timeout).

### Supervisor Lifecycle

11. `start-supervisor` starts the CLI supervisor as a background process.
12. The supervisor connects to the harness WebSocket, sends registration frame `{"register": "supervisor"}`.
13. On successful connection, writes `{"seq": 1, "type": "connected"}` to the event stream and sets the status object's state to `waiting_for_event`.
14. `start-supervisor` polls the status object until `connected` is `true`, then returns.
   14a. The readiness poll fails fast if the spawned supervisor exits before connecting, rather than polling until the timeout (which would mask an early crash — e.g. import error or missing data dir — behind a never-appearing status object). Mirrors rule 9b.
15. `stop-supervisor` writes `{"action": "shutdown"}` to the command file, waits for process exit (SIGKILL on 5s timeout).

### Per-Event Blocking Model

16. The supervisor's main loop: receive one WebSocket message → write event to the event stream → update the status object's state to `waiting_for_command` → poll the command file until a command appears → consume and delete it → process command → return to waiting for next WebSocket message.
17. The supervisor buffers the latest `ratify_request` proposal for use when `ratify` command arrives.
18. On `continue` command, the supervisor sends nothing to the harness and returns to waiting for the next event.
19. On `shutdown` command, the supervisor writes `{"type": "disconnected"}` event, disconnects WebSocket, and exits.

### Command Processing

20. The supervisor maps simplified commands to harness bus messages using `parse_command()` from `src/training/supervisors/commands.py`.
21. Commands that map to multiple bus messages (e.g., `ratify` → `countersign`) send all messages sequentially.
22. The supervisor sends commands via the WebSocket using the same JSON frame protocol as the TUI.

### Event Enrichment

23. Raw `RationaliseEvent` payloads are enriched with decompiled source before writing to the event stream.
24. KLine objects are converted to the KLine Display Object format: `{values: {signature, nodes}, for_display: <decompiled>}`.
25. Significance values are converted to the Significance Object format: `{raw, normalised, level}`.
26. The significance level (`S1`–`S4`) is derived from the raw byte via `BandLayout.classify`.
27. Progress events are passed through with field renaming only (`lesson_label` → `lesson`, etc.).

### Run Completion

28. When the supervisor receives a progress event with `status: "complete"`, it writes the event and sets the status object's state to `run_complete`.
29. The supervisor does not exit on run completion — it remains connected, waiting for pi to send `restart`, `shutdown`, or another command.

### Error Handling

30. On WebSocket disconnect (unexpected), the supervisor writes `{"type": "disconnected"}` and sets state to `errored`, then exits.
31. `stop-supervisor` and `stop-harness` use SIGKILL after a 5-second timeout as a safety net.

### Snapshot and Restore

32. `snapshot` increments the run counter in the session configuration and creates the run directory.
33. `snapshot` copies the curriculum state file to the run's state snapshot.
34. `snapshot` copies the event stream to the run's event log.
35. `snapshot` copies the Kalvin model file (if it exists) to the run's model snapshot.
36. `snapshot` writes the run's metadata with current git HEAD, branch, dirty status, and timestamp.
37. `restore` copies state and model files from the specified run back to their working locations.
38. `restore` requires the harness and supervisor to be stopped.

### Reset

39. `reset` deletes the curriculum state file, resolved from the worktree root.
40. `reset` truncates the event stream to empty.
41. `reset` does not modify the run counter or existing snapshots.
42. `reset --fresh-model` also deletes the Kalvin model file, resolved from the worktree root.

### Teardown

43. `teardown` removes the session's git worktree directory.
44. `teardown` deletes the associated `auto-tune/<session>` branch.
45. `teardown` must be run from the main repo (not from inside the worktree).

### Run Summary

46. `summary` aggregates the current run into a verdict written to `run-summary.json` in the session directory, and prints it. It is pure file aggregation (event stream + status + persisted trainer state) and is safe to run after any run, live or deadlocked — it does not touch the harness process.
47. The summary's `outcome` field is one of: `completed` (terminal completion event fired), `deadlocked` (run ended with submitted-but-unsatisfied entries and no completion), `supervisor-stalled` (a `ratify_request` sat unanswered — the supervisor role did not enact a decision), `stalled` (the run is connected but frozen — submitted-but-unsatisfied work and no event for longer than the stall threshold, or liveness unknown), `crashed` (an error surfaced in the stream), or `incomplete` (the run is still in progress and still moving).
   47a. A connected run is `stalled` rather than `incomplete` when it holds submitted-but-unsatisfied entries (or trainer state is absent) and has produced no event for longer than the stall threshold. This distinguishes a frozen run (the trainer's satisfaction accounting has deadlocked; driving it will not move it) from a busy one, so the agent stops churning `step` and diagnoses the model gap instead of being misdirected to "keep driving".
48. The summary's `significance` field is the S1–S4 band histogram over `rationalise` events, plus `ground`/`frame` counts. It is the analogue of dialogue-dev's displacement/escalation signals: the agent reads it to judge whether the run achieved the curriculum's goal.
49. The summary's `entries_total`/`entries_satisfied`/`entries_deadlocked` are derived from the persisted trainer state when available; they are `null` when the state file is absent (the outcome is still inferred from the event stream alone).
50. The summary's `diagnosis` field is a one- or two-sentence pointer naming the file or concept to inspect first. It is diagnosis, not procedure — it does not prescribe what to edit.
51. `summary` is the auto-tune arbiter: the signal the agent reads to decide whether to stop, diagnose, or continue, replacing per-turn supervision prose (see SKILL.md §Pi-in-the-Loop Model).

## Test Matrix

| ID    | Criterion                                                                                                                    | Origin ref              |
| ----- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| AT-1  | `init` creates session directory with all supporting files                                                                   | §Session Initialisation |
| AT-2  | `init` creates git worktree at `.worktrees/auto-tune/<session>` with branch `auto-tune/<session>`, main repo stays unchanged | §Session Initialisation |
| AT-3  | `init` records source branch, commit, harness URL, model path in the session configuration                                   | §Session Initialisation |
| AT-4  | `start-harness` starts harness and waits for WebSocket readiness                                                             | §Harness Lifecycle      |
| AT-5  | `stop-harness` gracefully terminates harness process                                                                         | §Harness Lifecycle      |
| AT-6  | Supervisor connects, registers as supervisor role, writes connected event                                                    | §Supervisor Lifecycle   |
| AT-7  | Supervisor writes one event per WebSocket message and blocks for command                                                     | §Per-Event Blocking     |
| AT-8  | `continue` command produces no harness message, resumes event loop                                                           | §Per-Event Blocking     |
| AT-9  | `ratify` command sends countersign for latest buffered proposal                                                              | §Per-Event Blocking     |
| AT-10 | `shutdown` command disconnects and exits cleanly                                                                             | §Per-Event Blocking     |
| AT-11 | Events include decompiled KLine source and significance breakdown                                                            | §Event Enrichment       |
| AT-12 | Run completion sets state to `run_complete` without exiting                                                                  | §Run Completion         |
| AT-13 | Unexpected disconnect writes disconnected event and sets errored state                                                       | §Error Handling         |
| AT-14 | `step` writes command, blocks until next event, prints it                                                                    | §CLI Subcommands        |
| AT-15 | `events --after N` returns events with seq > N                                                                               | §CLI Subcommands        |
| AT-16 | `snapshot` captures state, events, model, and git metadata                                                                   | §Snapshot and Restore   |
| AT-17 | `restore` reinstates state and model from a named run                                                                        | §Snapshot and Restore   |
| AT-18 | `reset` deletes curriculum state and truncates events                                                                        | §Reset                  |
| AT-19 | `reset --fresh-model` also deletes Kalvin model file                                                                         | §Reset                  |
| AT-20 | Process lifecycle commands manage PIDs and enforce timeouts                                                                  | §Error Handling         |
| AT-21 | `start-harness` kills orphan processes bound to the port (not just the PID-file process) and fails fast on a spawned-harness crash | §Harness Lifecycle     |
| AT-22 | `summary` writes `run-summary.json` and prints the verdict                                                                  | §Run Summary            |
| AT-23 | `summary` classifies `completed` / `deadlocked` / `supervisor-stalled` / `stalled` / `crashed` / `incomplete` from the event stream   | §Run Summary            |
| AT-24 | `summary` aggregates the S1–S4 significance histogram and ground/frame counts                                              | §Run Summary            |
| AT-25 | `summary` reports `entries_total`/`entries_satisfied`/`entries_deadlocked` from trainer state (null when absent)           | §Run Summary            |
| AT-26 | `summary` degrades gracefully when the trainer state file is missing (outcome still inferred)                              | §Run Summary            |
| AT-28 | A connected run with submitted-but-unsatisfied entries and no event for longer than the stall threshold reports `stalled`, not `incomplete` | §Run Summary            |
| AT-27 | `start-supervisor` fails fast if the spawned supervisor exits before connecting (does not poll status.json until timeout)  | §Supervisor Lifecycle   |

## Out of Scope

- Pi extension integration — the CLI is sufficient via bash
- Auto-tune running multiple sessions concurrently
- Automatic code editing by the auto-tune tool itself — pi edits code, auto-tune just runs training
- Performance benchmarking or timing comparisons between runs
- Integration with CI/CD pipelines
