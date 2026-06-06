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

| Command                                                               | Purpose                                                              |
| --------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `init --session <name> --curriculum <path> [--host <h>] [--port <p>]` | Create session directory, config.json, git branch `auto-tune/<name>` |
| `start-harness --session <name>`                                      | Start harness server (background), wait for ready                    |
| `stop-harness --session <name>`                                       | Graceful shutdown (SIGTERM → SIGKILL on timeout)                     |
| `start-supervisor --session <name>`                                   | Start CLI supervisor (background), wait for connected                |
| `stop-supervisor --session <name>`                                    | Send shutdown command, wait for exit                                 |
| `send --session <name> --command '<json>'`                            | Write command, return immediately                                    |
| `events --session <name> [--after <seq>]`                             | Print events after given sequence number                             |
| `step --session <name> --command '<json>'`                            | Write command, block until next event, print it                      |
| `status --session <name>`                                             | Print status.json                                                    |
| `snapshot --session <name>`                                           | Capture state, events, model, git metadata to `runs/<n>/`            |
| `restore --session <name> --run <n>`                                  | Restore state and model from a snapshot                              |
| `reset --session <name> [--fresh-model]`                              | Delete curriculum state, truncate events, optionally delete model    |

### Supervisor Commands (sent via `step` or `send`)

| Command JSON                            | Effect                                   |
| --------------------------------------- | ---------------------------------------- |
| `{"action": "start"}`                   | Begin training session                   |
| `{"action": "continue"}`                | No-op — acknowledge event, wait for next |
| `{"action": "ratify"}`                  | Countersign latest pending proposal      |
| `{"action": "restart"}`                 | Reset training state and restart         |
| `{"action": "pause"}`                   | Pause training                           |
| `{"action": "resume"}`                  | Resume training                          |
| `{"action": "stop"}`                    | End training session                     |
| `{"action": "shutdown"}`                | Graceful supervisor exit                 |
| `{"action": "guidance", "text": "..."}` | Send guidance to trainer                 |
| `{"action": "goal", "text": "..."}`     | Set a training goal                      |
| `{"action": "save"}`                    | Persist Kalvin model                     |
| `{"action": "load"}`                    | Load Kalvin model                        |

### Event Types (received from supervisor)

| Type             | When                                                                  |
| ---------------- | --------------------------------------------------------------------- |
| `connected`      | Supervisor connected to harness                                       |
| `progress`       | Training progress (started, lesson_complete, complete, amended, etc.) |
| `rationalise`    | Kalvin rationalisation event (ground or frame)                        |
| `ratify_request` | S2/S3 proposal needing ratification                                   |
| `escalation`     | Trainer stuck (budget_exhaustion or low_confidence)                   |
| `disconnected`   | Supervisor disconnected                                               |

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
4. **Edit code** — make changes to address the goal. Check that you are on a branch and be aggressive
5. **Snapshot before each code change** — so you can compare before/after

### Fix Before Continuing

If a training run crashes or produces unexpected errors (exceptions in harness.log, ValueError, traceback, etc.), **stop and fix the bug before continuing the tune session**. Do not work around failures or treat them as acceptable noise. The training pipeline must run cleanly.

Steps:

1. **Identify** — read the full traceback in harness.log. Trace the code path.
2. **Reproduce** — write a minimal inline test (PYTHONPATH=src python -c "...") that triggers the failure.
3. **Fix** — edit the source code.
4. **Verify** — re-run the reproduction test and the existing test suite.
5. **Re-run** — snapshot, reset, and run the training session again to confirm the fix.

### Commit Regularly

Commit each meaningful change to the auto-tune branch as you go. Do not accumulate a large diff across multiple runs. Good commit points include:

- After fixing a bug found during a training run
- After a code change that produces an observable difference in the training trace
- After updating or adding tests
- After any documentation update (spec, plan)

Use descriptive commit messages that reference the session and the change:

```
fix(expand): cycle detection in edge_hops, null-safe _as_kline replacement

DO NOT MERGE INTO MAIN
```

### Key Files to Observe

| File                               | What it tells you                                                   |
| ---------------------------------- | ------------------------------------------------------------------- |
| `auto-tune/<session>/harness.log`  | Full server-side training trace (Trainer, Reactor, Adapter logging) |
| `auto-tune/<session>/events.jsonl` | Structured event stream from supervisor's perspective               |
| `auto-tune/<session>/status.json`  | Supervisor state (connected, last_event_seq, state)                 |
| `auto-tune/<session>/runs/<n>/`    | Snapshot directory (state, events, git metadata)                    |

## Phase 4: Auto-Review

After each run, **before** editing code for the session goal, automatically review the run artifacts for systemic issues in the auto-tune tooling itself. This is a self-improvement pass: the training pipeline and CLI are code too, and bugs or friction in them waste every subsequent run.

### Why

Auto-tune sessions produce structured evidence — logs, events, snapshots, git metadata — that reveal not just whether Kalvin is learning, but whether the auto-tune _machinery_ is working correctly. Session `llm-misfix` (runs 001–003) demonstrated six distinct tooling defects discovered only through post-run review: stale processes corrupting event data, a disconnected curriculum config, missing cmd.json cleanup, wrong git branch, absent LLM client, and null lesson labels.

Auto-review catches these defects automatically and fixes them before they compound across future runs.

### When to Run Auto-Review

Run auto-review **after every snapshot**, before any goal-directed code editing. It is cheap (reads files, checks state) and catches problems that would otherwise surface as confusing failures mid-run.

### Review Checklist

For each item, check the evidence, classify the finding, and fix it immediately. All fixes go into `src/participants/auto_tune/` and `tests/test_auto_tune_*.py`.

#### 1. Git Hygiene

**Check:** `runs/<n>/meta.json` — is `git_branch` the expected `auto-tune/<session>` branch?

```bash
PYTHONPATH=src python -c "import json; m=json.load(open('auto-tune/<session>/runs/<n>/meta.json')); print(m['git_branch'])"
```

**If wrong:** The session is committing to the wrong branch. Fix `reset()` in `snapshots.py` to check out the correct branch. Commit the fix.

**If `git_dirty: true` with unexpected files:** Check for files that should be gitignored or cleaned up by reset.

#### 2. Process Hygiene

**Check:** Were there duplicate supervisors or harnesses during the run?

Evidence:

- `harness.log` containing multiple `Client registered: supervisor` lines from different PIDs
- `events.jsonl` with out-of-order or duplicate sequence numbers
- `status.json` showing a PID that doesn't match `supervisor.pid`

**If found:** The stale process detection in `lifecycle.py` (`_kill_stale_process`) missed something. Investigate why and strengthen the detection. Add a test.

#### 3. Curriculum Integrity

**Check:** Did the harness load the curriculum from `config.json`, or from the project's `harness.yaml`?

Evidence:

- `harness.log` line `Session started — N lessons, curriculum: <path>` — does the path match the session config?
- The per-session `auto-tune/<session>/harness.yaml` should exist and have the correct `curriculum_file`

**If wrong:** The per-session config generation (`_generate_session_harness_config`) is not being called or is failing silently. Fix `start_harness()` in `lifecycle.py`. Add a test.

#### 4. Event Quality

**Check:** `runs/<n>/events.jsonl` — are all events well-formed and complete?

Evidence:

- `lesson: null` on `lesson_complete` events → the Trainer advanced the curriculum position before emitting the progress event
- `reason: ""` or `detail: ""` on escalation events → enrichment not capturing available context
- Missing event types (no `rationalise` events when curriculum has S2 lessons)

**If found:** Fix the event source. For null lesson labels, capture the label before advancing position (as done in `_check_lesson_complete`). For missing enrichment, trace the code path from harness → supervisor → events.jsonl. Add a test.

#### 5. LLM Availability

**Check:** `harness.log` for the warning `'openai' package not installed — LLM client unavailable`.

**Cross-reference:** Does the curriculum contain lessons that trigger S2/S3 (slow path)? If yes and no LLM client is available, every S2 entry will escalate as `low_confidence` with no possibility of reactive scaffolding.

**If found:** This is not a code fix — it's a session configuration issue. Either:

- Install the LLM client (`pip install kalvin[trainer]`)
- Switch to a curriculum that only uses S1 (fast path) lessons
- Warn the user that the session will produce escalation-only results

#### 6. Repeated Identical Escalations

**Check:** Count consecutive escalation events with the same `reason` and `lesson_position`.

```bash
PYTHONPATH=src python -c "
import json
events = [json.loads(l) for l in open('auto-tune/<session>/runs/<n>/events.jsonl')]
escalations = [e for e in events if e['type'] == 'escalation']
for i, e in enumerate(escalations):
    print(f'{i}: reason={e["reason"]} lesson_position={e["lesson_position"]}')
"
```

**If found (3+ identical in a row):** The reactor is re-escalating without any intervening action. Consider:

- Is the max_reactive_rounds budget being consumed uselessly?
- Should the supervisor auto-continue duplicate escalations instead of stepping through each one?
- Is there a code fix that would make escalation events more actionable (e.g., richer detail)?

#### 7. Session Startup Friction

**Check:** How many `step` commands timed out during the session? Were there immediate disconnects or shutdowns?

Evidence:

- `events.jsonl` starting with `connected` → `disconnected` (no training events)
- `cmd.json` containing `{"action": "shutdown"}` before the session starts

**If found:** `reset()` did not clean up `cmd.json`. Fix `reset()` in `snapshots.py` to delete stale command files. Add a test.

### Applying Fixes

For each finding:

1. **Reproduce** — write a minimal test that demonstrates the issue.
2. **Fix** — edit the auto-tune source code (`src/participants/auto_tune/`, `src/trainer/`, etc.).
3. **Verify** — run the test suite (`PYTHONPATH=src python -m pytest tests/test_auto_tune_*.py`).
4. **Commit** — commit with a message like:

```
fix(auto-tune): stale process detection in lifecycle start-harness

Caught by auto-review in session llm-misfix run 003.
Evidence: duplicate supervisor PIDs writing interleaved events.

DO NOT MERGE INTO MAIN
```

5. **Re-run** — snapshot, reset, and run the training session again. Confirm the finding is gone in the next run's artifacts.

### What NOT to Fix

Auto-review targets the **auto-tune tooling and training pipeline**, not the Kalvin model or training curriculum. Do not use auto-review to:

- Change the training curriculum
- Modify Kalvin's rationalisation logic
- Adjust the Trainer's pedagogical strategy

Those are Phase 3 concerns driven by the session goal.

## Phase 5: Document

Every auto-tune session must produce cascade documentation that integrates the work into main development as if it had been developed interactively.

### Required Outputs

1. **Spec** (`specs/<feature>.md`) — behavioural rules, test matrix, definitions derived from what you observed and implemented. Reference the auto-tune session evidence.

2. **Plan** (`plans/impl/<feature>.md`) — implementation tasks, test mapping, design decisions. Include:
   - A link to the auto-tune session directory as evidence
   - Test mapping with status (✅ for passing tests, ☐ for not yet written)

3. **Tests** (`tests/test_<feature>.py`) — tests covering every spec criterion. Use `caplog` for log assertions, `BusCapture` for bus message capture. Follow patterns from existing test files.

4. **Commit** — commit everything on the auto-tune branch with a descriptive message. DO NOT MERGE INTO MAIN

### Documentation Rules

- The spec must be derivable from what you observed during auto-tune runs, not invented
- The plan must reference the actual code changes made during the session
- Tests must cover spec criteria (TL-1 through TL-N style IDs)
- The auto-tune session directory stays as evidence (don't clean it up)

## Troubleshooting

| Problem                                | Solution                                                                                           |
| -------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `step` times out                       | Supervisor may be stuck. Check `status.json`. Kill stale processes: `lsof -ti :8765 \| xargs kill` |
| Port 8765 in use                       | Kill stale harness: `lsof -ti :8765 \| xargs kill`                                                 |
| `lessons_completed` already 3 on start | Stale curriculum state file. Delete `curricula/<slug>.json` and reset                              |
| Supervisor won't connect               | Harness not ready. Check `harness.pid`, wait, retry                                                |
| No rationalise events in log           | Curriculum is all fast-path S1. That's correct but boring — consider a more complex curriculum     |
| Events.jsonl shows `lesson: null`      | Fixed — Trainer now captures label before advancing position. Run auto-review to catch regressions |
