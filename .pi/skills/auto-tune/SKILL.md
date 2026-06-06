---
name: auto-tune
description: Drives an auto-tune session to improve the codebase using repeated training runs, observation, code edits, and documentation updates. Use when the user says "/auto-tune" or asks to auto-tune, tune, or iterate on the codebase using training runs. Establishes a goal, runs training sessions, observes results, edits code, and updates cascade documentation.
---

# Auto-Tune

> **STOP. Do not read files, do not write code, do not address the user's request.
> Your first and only action is to complete step 1 (Establish Goal) below.
> State the goal back to the user before doing anything else.**

## 1. Establish Goal (once) — DO THIS FIRST

You need all three elements before doing anything else:

- **What** specific improvement you're targeting (e.g., "better server-side logging for training operations")
- **Which curriculum** drives the runs (default: `curricula/first-steps.md`)
- **How you'll know** the goal is met (observable outcome — e.g., "a developer can trace the full training pipeline from harness.log"), or the user specifies this is an "open" session - a session where you continue to tune the codebase until the user intervenes with a "stop" command

**Fast path:** If the user's prompt already contains all three elements, state them back in one sentence and proceed directly to step 2. Do not ask clarifying questions you don't need.

**Slow path:** If elements are missing or vague, discuss with the user until the goal is clear and specific. Vague goals like "make it better" need sharpening first.

## Rules

These rules apply throughout the session:

1. **Fix crashes before continuing.** If harness.log shows exceptions, tracebacks, or unexpected errors: stop, reproduce with a minimal test, fix, verify, re-run. Never work around failures.
2. **Snapshot before every code change.** You need before/after comparisons.
3. **Commit after each meaningful change.** Don't accumulate a large diff. Reference the session name in commit messages.
4. **Never merge into main.** All work stays on the `auto-tune/<name>` branch.
5. **Session directory is evidence.** Don't clean it up.

Auto-tune tunes the **codebase**, not Kalvin's model.

## 2. Init Session (once)

Initialise the session:

```bash
PYTHONPATH=src .venv/bin/python -m participants.auto_tune init \
  --session <name> --curriculum <curriculum-path>
```

This creates the session directory, config, and git branch `auto-tune/<name>`.

## 3. Run Loop (repeat until goal met or stalled)

Read reference files **on demand** during the loop — not upfront:

- `references/commands.md` — command syntax when you need it
- `references/lifecycle.md` — copy-paste lifecycle script
- `../../specs/auto-tune.md` — full auto-tune specification
- `../../CONTEXT.md` — domain glossary

Do not read all reference files before starting. Read them when you need an answer.

Each iteration:

### a. Start processes

```bash
PYTHONPATH=src .venv/bin/python -m participants.auto_tune start-harness --session <name>
PYTHONPATH=src .venv/bin/python -m participants.auto_tune start-supervisor --session <name>
```

### b. Step through events

```bash
PYTHONPATH=src .venv/bin/python -m participants.auto_tune step \
  --session <name> --command '{"action": "start"}'
# Then continue through all events:
PYTHONPATH=src .venv/bin/python -m participants.auto_tune step \
  --session <name> --command '{"action": "continue"}'
# ... repeat until disconnected event ...
```

### c. Snapshot

```bash
PYTHONPATH=src .venv/bin/python -m participants.auto_tune snapshot --session <name>
```

### d. Observe

1. Read `auto-tune/<name>/harness.log` — the server-side training trace
2. Read `auto-tune/<name>/events.jsonl` — the full event stream
3. Compare snapshots — diff harness.log between runs to see impact of changes

### e. Edit or loop

- If the goal is met → go to step 4 (Document)
- If there are crashes → fix (see Rule 1), then go to f
- If there's an improvement to try → snapshot, edit code, commit, then go to f
- If 3 runs pass with no meaningful improvement → surface this to the user and ask whether to continue

### f. Reset for next run

```bash
PYTHONPATH=src .venv/bin/python -m participants.auto_tune stop-supervisor --session <name>
PYTHONPATH=src .venv/bin/python -m participants.auto_tune stop-harness --session <name>
PYTHONPATH=src .venv/bin/python -m participants.auto_tune reset --session <name>
```

Then return to step a.

For common problems, read `references/troubleshooting.md`.

## 4. Document (once, at end)

Every auto-tune session must produce cascade documentation:

1. **Spec** (`specs/<feature>.md`) — behavioural rules, test matrix, definitions derived from what you observed. Reference the auto-tune session directory as evidence.

2. **Plan** (`plans/impl/<feature>.md`) — implementation tasks, test mapping, design decisions. Include:
   - A link to the auto-tune session directory as evidence
   - Test mapping with status (✅ for passing tests, ☐ for not yet written)

3. **Tests** (`tests/test_<feature>.py`) — tests covering every spec criterion. Use `caplog` for log assertions, `BusCapture` for bus message capture. Follow patterns from existing test files.

4. **Commit** — commit everything on the auto-tune branch with a descriptive message. DO NOT MERGE INTO MAIN.

### Documentation Rules

- The spec must be derivable from what you observed during auto-tune runs, not invented
- The plan must reference the actual code changes made during the session
- Tests must cover spec criteria (TL-1 through TL-N style IDs)
