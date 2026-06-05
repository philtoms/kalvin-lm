---
name: auto-tune
description: Drives an auto-tune session to improve the codebase using repeated training runs, observation, code edits, and documentation updates. Use when the user says "/auto-tune" or asks to auto-tune, tune, or iterate on the codebase using training runs. Establishes a goal, runs training sessions, observes results, edits code, and updates cascade documentation.
---

# Auto-Tune

Auto-tune is a hands-free loop where you (pi) control training sessions against the Kalvin codebase, observe results, edit code, and re-run to converge on a goal.

## Before Starting

Read these files for context:
- `specs/auto-tune.md` — full specification of the auto-tune system
- `CONTEXT.md` — domain glossary (especially the Auto-Tune, CLI Supervisor, and Snapshot entries)

## Phase 1: Establish the Goal

Before creating any session, discuss the goal with the user until you can state it precisely. Ask:

1. **What specific improvement are you targeting?** (e.g., "better server-side logging for training operations")
2. **Which curriculum should drive the training runs?** (default: `curricula/first-steps.md`)
3. **How will you know the goal is met?** (observable outcome — e.g., "a developer can trace the full training pipeline from harness.log")

Do NOT proceed to session creation until the goal is clear and specific. Vague goals like "make it better" need sharpening first.

## Phase 2: Create and Run

### Commands Reference

All commands require `PYTHONPATH=src` and are invoked as:
```bash
PYTHONPATH=src python -m participants.auto_tune <command> --session <name> [options]
```

| Command | Purpose |
|---------|---------|
| `init --session <name> --curriculum <path> [--host <h>] [--port <p>]` | Create session directory, config.json, git branch `auto-tune/<name>` |
| `start-harness --session <name>` | Start harness server (background), wait for ready |
| `stop-harness --session <name>` | Graceful shutdown (SIGTERM → SIGKILL on timeout) |
| `start-supervisor --session <name>` | Start CLI supervisor (background), wait for connected |
| `stop-supervisor --session <name>` | Send shutdown command, wait for exit |
| `send --session <name> --command '<json>'` | Write command, return immediately |
| `events --session <name> [--after <seq>]` | Print events after given sequence number |
| `step --session <name> --command '<json>'` | Write command, block until next event, print it |
| `status --session <name>` | Print status.json |
| `snapshot --session <name>` | Capture state, events, model, git metadata to `runs/<n>/` |
| `restore --session <name> --run <n>` | Restore state and model from a snapshot |
| `reset --session <name> [--fresh-model]` | Delete curriculum state, truncate events, optionally delete model |

### Supervisor Commands (sent via `step` or `send`)

| Command JSON | Effect |
|-------------|--------|
| `{"action": "start"}` | Begin training session |
| `{"action": "continue"}` | No-op — acknowledge event, wait for next |
| `{"action": "ratify"}` | Countersign latest pending proposal |
| `{"action": "restart"}` | Reset training state and restart |
| `{"action": "pause"}` | Pause training |
| `{"action": "resume"}` | Resume training |
| `{"action": "stop"}` | End training session |
| `{"action": "shutdown"}` | Graceful supervisor exit |
| `{"action": "guidance", "text": "..."}` | Send guidance to trainer |
| `{"action": "goal", "text": "..."}` | Set a training goal |
| `{"action": "save"}` | Persist Kalvin model |
| `{"action": "load"}` | Load Kalvin model |

### Event Types (received from supervisor)

| Type | When |
|------|------|
| `connected` | Supervisor connected to harness |
| `progress` | Training progress (started, lesson_complete, complete, amended, etc.) |
| `rationalise` | Kalvin rationalisation event (ground or frame) |
| `ratify_request` | S2/S3 proposal needing ratification |
| `escalation` | Trainer stuck (budget_exhaustion or low_confidence) |
| `disconnected` | Supervisor disconnected |

### Typical Session Lifecycle

```bash
# 1. Init
PYTHONPATH=src python -m participants.auto_tune init --session <name> --curriculum curricula/first-steps.md

# 2. Start processes
PYTHONPATH=src python -m participants.auto_tune start-harness --session <name>
PYTHONPATH=src python -m participants.auto_tune start-supervisor --session <name>

# 3. Run training (one event at a time)
PYTHONPATH=src python -m participants.auto_tune step --session <name> --command '{"action": "start"}'
PYTHONPATH=src python -m participants.auto_tune step --session <name> --command '{"action": "continue"}'
# ... repeat through all events ...

# 4. Snapshot results
PYTHONPATH=src python -m participants.auto_tune snapshot --session <name>

# 5. Observe
cat auto-tune/<name>/harness.log

# 6. Stop processes
PYTHONPATH=src python -m participants.auto_tune stop-supervisor --session <name>
PYTHONPATH=src python -m participants.auto_tune stop-harness --session <name>
```

### Between Runs (the tuning loop)

```bash
# Reset state for a fresh run
PYTHONPATH=src python -m participants.auto_tune reset --session <name>

# Restart processes
PYTHONPATH=src python -m participants.auto_tune start-harness --session <name>
PYTHONPATH=src python -m participants.auto_tune start-supervisor --session <name>

# Run again...
```

## Phase 3: Observe and Edit

After each run:

1. **Read `auto-tune/<session>/harness.log`** — the server-side training trace
2. **Read events** — `auto-tune/<session>/events.jsonl` for the full event stream
3. **Compare snapshots** — diff harness.log between runs to see impact of code changes
4. **Edit code** — make changes to address the goal
5. **Snapshot before each code change** — so you can compare before/after

### Key Files to Observe

| File | What it tells you |
|------|-------------------|
| `auto-tune/<session>/harness.log` | Full server-side training trace (Trainer, Reactor, Adapter logging) |
| `auto-tune/<session>/events.jsonl` | Structured event stream from supervisor's perspective |
| `auto-tune/<session>/status.json` | Supervisor state (connected, last_event_seq, state) |
| `auto-tune/<session>/runs/<n>/` | Snapshot directory (state, events, git metadata) |

## Phase 4: Document

Every auto-tune session must produce cascade documentation that integrates the work into main development as if it had been developed interactively.

### Required Outputs

1. **Spec** (`specs/<feature>.md`) — behavioural rules, test matrix, definitions derived from what you observed and implemented. Reference the auto-tune session evidence.

2. **Plan** (`plans/impl/<feature>.md`) — implementation tasks, test mapping, design decisions. Include:
   - A link to the auto-tune session directory as evidence
   - Test mapping with status (✅ for passing tests, ☐ for not yet written)

3. **Tests** (`tests/test_<feature>.py`) — tests covering every spec criterion. Use `caplog` for log assertions, `BusCapture` for bus message capture. Follow patterns from existing test files.

4. **Commit** — commit everything on the auto-tune branch with a descriptive message.

### Documentation Rules

- The spec must be derivable from what you observed during auto-tune runs, not invented
- The plan must reference the actual code changes made during the session
- Tests must cover spec criteria (TL-1 through TL-N style IDs)
- The auto-tune session directory stays as evidence (don't clean it up)

## Phase 5: Merge

When the goal is met and documentation is complete:

1. Verify all tests pass: `PYTHONPATH=src python -m pytest tests/test_<feature>.py -v`
2. Commit any remaining changes on the auto-tune branch
3. Merge to main: `git checkout main && git merge auto-tune/<session> --no-ff`
4. Delete the branch: `git branch -d auto-tune/<session>`

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `step` times out | Supervisor may be stuck. Check `status.json`. Kill stale processes: `lsof -ti :8765 \| xargs kill` |
| Port 8765 in use | Kill stale harness: `lsof -ti :8765 \| xargs kill` |
| `lessons_completed` already 3 on start | Stale curriculum state file. Delete `curricula/<slug>.json` and reset |
| Supervisor won't connect | Harness not ready. Check `harness.pid`, wait, retry |
| No rationalise events in log | Curriculum is all fast-path S1. That's correct but boring — consider a more complex curriculum |
| Events.jsonl shows `lesson: null` | Minor enrichment bug in progress events — the data is still valid |
