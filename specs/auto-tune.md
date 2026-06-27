# Auto-Tune Specification

## Overview

Auto-tune is a CLI tool and supervisor participant that enables an LLM coding agent (pi) to autonomously control training sessions, observe results, modify the codebase, and re-run — converging on a code quality goal. Pi owns the full lifecycle: it starts the harness server, starts the CLI supervisor, drives training via commands, reads events, edits code, snapshots state, resets, and repeats.

The system has two components:

- **Auto-tune CLI** (`python -m training.participants.auto_tune`) — session management, process lifecycle, and the `step`/`send`/`events` commands.
- **CLI Supervisor** — a WebSocket client participant that connects to the harness, reads commands from a file, writes events to a file, and blocks per-event for maximum observability.

## Dependencies

- `src/training/harness/` — message bus, WebSocket protocol, harness server
- `src/training/participants/commands.py` — `parse_command()` for mapping simplified commands to bus messages
- `src/kalvin/expand.py` — `D_MAX` for significance normalisation
- `specs/harness-server.md` — harness configuration and participant architecture
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
| `ratify_request` | `seq`, `query`, `proposal`, `significance`, (`misfit`, `curriculum_context`) | S2/S3 proposal requiring ratification decision. When reactive delegation is active (`@specs/reactive-delegation.md`), enriched with the misfit diagnosis and curriculum context so the supervisor can write reactive scaffolding. |
| `escalation`     | `seq`, `reason`, `detail`, `lesson_position`                                 | Trainer cannot make progress (`budget_exhaustion` or `low_confidence`)                                                                                                                                                            |

### Significance Object

| Field      | Type    | Description                              |
| ---------- | ------- | ---------------------------------------- |
| raw        | `int`   | 64-bit significance integer              |
| normalised | `float` | `significance / D_MAX` (0.0 S4 — 1.0 S1) |
| level      | `str`   | `S1`, `S2`, `S3`, or `S4`                |

### KLine Display Object

| Field  | Type                                 | Description               |
| ------ | ------------------------------------ | ------------------------- |
| raw    | `{signature: int, nodes: list[int]}` | Raw KLine data            |
| source | `str`                                | Decompiled KScript source |

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
| `scaffold` | `action`, `text` | Submit reactive scaffolding KScript to Kalvin (delegated reactive decision — `@specs/reactive-delegation.md`) |
| `continue` | `action`         | No-op: acknowledge event, wait for next                                                                       |
| `shutdown` | `action`         | Graceful supervisor shutdown                                                                                  |

### Status Object

| Field          | Type             | Description                                                                                          |
| -------------- | ---------------- | ---------------------------------------------------------------------------------------------------- |
| pid            | `int`            | Supervisor process ID                                                                                |
| connected      | `bool`           | WebSocket connection state                                                                           |
| last_event_seq | `int`            | Sequence number of last written event                                                                |
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
   7a. `start-harness` generates that per-session harness config (from the project harness config) setting `trainer.llm.enabled: false`, placing the session in delegated mode so pi acts as the reactive decision-maker (`@specs/reactive-delegation.md`).
8. `start-harness` records the PID in the session directory.
9. `start-harness` polls the WebSocket port until it accepts connections, then returns.
10. `stop-harness` sends SIGTERM to the harness PID, waits for exit (SIGKILL on 5s timeout).

### Supervisor Lifecycle

11. `start-supervisor` starts the CLI supervisor as a background process.
12. The supervisor connects to the harness WebSocket, sends registration frame `{"register": "supervisor"}`.
13. On successful connection, writes `{"seq": 1, "type": "connected"}` to the event stream and sets the status object's state to `waiting_for_event`.
14. `start-supervisor` polls the status object until `connected` is `true`, then returns.
15. `stop-supervisor` writes `{"action": "shutdown"}` to the command file, waits for process exit (SIGKILL on 5s timeout).

### Per-Event Blocking Model

16. The supervisor's main loop: receive one WebSocket message → write event to the event stream → update the status object's state to `waiting_for_command` → poll the command file until a command appears → consume and delete it → process command → return to waiting for next WebSocket message.
17. The supervisor buffers the latest `ratify_request` proposal for use when `ratify` command arrives.
18. On `continue` command, the supervisor sends nothing to the harness and returns to waiting for the next event.
19. On `shutdown` command, the supervisor writes `{"type": "disconnected"}` event, disconnects WebSocket, and exits.

### Command Processing

20. The supervisor maps simplified commands to harness bus messages using `parse_command()` from `src/training/participants/commands.py`.
21. Commands that map to multiple bus messages (e.g., `ratify` → `countersign`) send all messages sequentially.
22. The supervisor sends commands via the WebSocket using the same JSON frame protocol as the TUI.

### Event Enrichment

23. Raw `RationaliseEvent` payloads are enriched with decompiled source before writing to the event stream.
24. KLine objects are converted to the KLine Display Object format: `{raw: {signature, nodes}, source: <decompiled>}`.
25. Significance values are converted to the Significance Object format: `{raw, normalised, level}`.
26. The significance level (`S1`–`S4`) is derived from the raw significance and `D_MAX` using existing classification logic.
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

## Out of Scope

- Pi extension integration — the CLI is sufficient via bash
- Auto-tune running multiple sessions concurrently
- Automatic code editing by the auto-tune tool itself — pi edits code, auto-tune just runs training
- Performance benchmarking or timing comparisons between runs
- Integration with CI/CD pipelines
