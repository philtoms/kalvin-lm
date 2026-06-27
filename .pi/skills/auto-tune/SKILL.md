---
name: auto-tune
description: Drives an auto-tune session to improve the codebase using repeated training runs, observation, code edits, and documentation updates. Use when the user says "/auto-tune" or asks to auto-tune, tune, or iterate on the codebase using training runs. Establishes a goal, runs training sessions, observes results, edits code, and updates cascade documentation.
---

# Auto-Tune

> **STOP. Do not read files, do not write code, do not address the user's request.
> Your first and only action is to determine which entry point applies (New Session or Resume Session) and complete step 1.**

## Entry Points

### New Session

The user wants to start a fresh auto-tune session. No `auto-tune/<name>/session-state.md` exists yet. → Complete step 1, then step 2.

### Resume Session

The user says "resume", "continue", or points at an existing session. A `session-state.md` file exists in the session directory. → Read `auto-tune/<name>/session-state.md`, confirm the goal and next action with the user, then jump directly to the phase indicated by **Current Phase** in the state file.

**Resuming from the main repo:** The session lives in a git worktree at `.worktrees/auto-tune/<name>/`. You can find the session state at `.worktrees/auto-tune/<name>/auto-tune/<name>/session-state.md`. Cd into `.worktrees/auto-tune/<name>/` before continuing.

**Automatic handoff:** If the `auto-tune-handoff` extension is loaded, it monitors context usage and, when you cross the ceiling (default 80%), injects a steering message telling you to stop and update session-state.md, then shows a notification prompting you to run `/auto-tune-handoff`. The command creates a fresh session that starts with "Resume auto-tune <name>".

**Manual handoff:** If you notice context is getting large, update `session-state.md` and tell the user: "Context is getting large. Run `/auto-tune-handoff` or start a fresh conversation and say 'resume auto-tune <name>'."

## Pi-in-the-Loop Model

Auto-tune has two supervisor models. Both fill the same `supervisor` role on
the harness bus and produce identical logs; they differ only in **who
reasons**.

- **Human-in-the-loop** (Slack / TUI participant): a human is the supervisor.
  Their reasoning happens in their head or Slack and is **never in the logs** —
  the logs capture protocol only (events in, commands out).
- **Pi-in-the-loop** (auto-tune, this skill): **you (pi) are the supervisor
  participant** — the agent-in-the-loop. The CLI Supervisor process is merely a
  protocol relay (WebSocket ↔ file: `events.jsonl` out, `cmd.json` in); it does
  not think. You drive it by writing commands via `step`/`send`.

**Your reasoning will never be in the logs — by design, exactly as a human
supervisor's reasoning never is.** The record of your supervisor reasoning
lives in the **pi session window**, which is your native surface the way a
human's head/Slack is theirs. The session window is the *only* place evidence
that "pi is performing supervisor mode" can exist.

**Consequence for the run loop:** supervisor-decision events (`ratify_request`,
`escalation`, and any event that needs a non-`continue` answer) **must be
handled turn-by-turn, with your reasoning shown in the session window before
each command is emitted.** One decision event = one turn of reasoning = one
emitted command. Routine observation events (`rationalise`, `progress`,
`ground`) can be driven efficiently — you observe them but do not act.

**Never respond to a decision event with a hardcoded or looped command.**
Driving a `ratify_request` with a scripted `ratify` abdicates the supervisor
role just as surely as a human pressing a button without reading the prompt.
When `commands.jsonl` shows a run of identical decisions with no reasoning
between them in the window, that is a process failure, not a result.

## 1. Establish Goal (once, new sessions only) — DO THIS FIRST

You need all three elements before doing anything else:

- **What** specific improvement you're targeting (e.g., "better server-side logging for training operations")
- **Which curriculum** drives the runs (default: `curricula/first-steps.md`)
- **How you'll know** the goal is met (observable outcome — e.g., "a developer can trace the full training pipeline from harness.log"), or the user specifies this is an "open" session

**Fast path:** If the user's prompt already contains all three elements, state them back in one sentence and proceed directly to step 2.

**Slow path:** If elements are missing or vague, discuss with the user until the goal is clear and specific.

## Rules

These rules apply throughout the session:

1. **Fix crashes before continuing.** If harness.log shows exceptions, tracebacks, or unexpected errors: stop, reproduce with a minimal test, fix, verify, re-run. Never work around failures.
2. **Snapshot before every code change.** You need before/after comparisons.
3. **Commit after each meaningful change.** Don't accumulate a large diff. Reference the session name in commit messages.
4. **Never merge into main.** All work stays on the `auto-tune/<name>` branch inside the worktree.
5. **Session state is sacred.** Update `session-state.md` after every observation (step 3d). This file is your lifeline for resuming after context resets.
6. **Keep context lean.** Don't re-read old harness logs or events from previous runs. The state file has the summary. Only read the current run's artifacts. (This concerns *historical* runs — it does **not** excuse skipping per-event reasoning on decision events; see §Pi-in-the-Loop Model.)
7. **Work inside the worktree.** After init, all commands and file operations happen inside `.worktrees/auto-tune/<name>/`. Never modify files in the main repo.

Auto-tune tunes the **codebase** AND the **project documentation**.

## 2. Init Session (once, new sessions only)

Capture the Python interpreter path from the main repo, then initialise:

```bash
# From the main project repo:
AT_PYTHON="$(pwd)/.venv/bin/python"

PYTHONPATH=src $AT_PYTHON -m participants.auto_tune init \
  --session <name> --curriculum <curriculum-path>
```

This creates a git worktree at `.worktrees/auto-tune/<name>/` with branch `auto-tune/<name>`. The main repo stays on its current branch — full isolation.

After init, cd into the worktree:

```bash
cd .worktrees/auto-tune/<name>
```

All subsequent commands run from inside the worktree using `$AT_PYTHON`:

```bash
PYTHONPATH=src $AT_PYTHON -m participants.auto_tune <command> --session <name>
```

Then create the initial `auto-tune/<name>/session-state.md` using the template from `references/session-state-format.md`. Fill in the goal, done criteria, session details. Set **Current Phase** to `running`, **Next Action** to `start run 1`. Include the **Worktree** field.

## 3. Run Loop (repeat until goal met or stalled)

**At the top of every loop iteration, read `auto-tune/<name>/session-state.md` to re-anchor.** This takes ~50 lines instead of the full chat history.

Read reference files **on demand** — not upfront:

- `references/commands.md` — command syntax when you need it
- `references/lifecycle.md` — copy-paste lifecycle script
- `references/session-state-format.md` — state file template (only when creating or restructuring)
- `references/troubleshooting.md` — when something goes wrong
- `../../specs/auto-tune.md` — full auto-tune specification (relative from worktree root)
- `../../CONTEXT.md` — domain glossary (relative from worktree root)

Do not read all reference files before starting. Read them when you need an answer.

Each iteration:

### a. Start processes

```bash
PYTHONPATH=src $AT_PYTHON -m participants.auto_tune start-harness --session <name>
PYTHONPATH=src $AT_PYTHON -m participants.auto_tune start-supervisor --session <name>
```

Update state: **Current Phase** → `running`.

### b. Step through events

There are two classes of event, handled differently (see **§Pi-in-the-Loop
Model**):

**Routine observation events** — `rationalise`, `progress`, `ground`,
`connected`, `disconnected`: the supervisor role observes but does not act.
Drive these with `continue`. You may batch them in a loop and do **not** need to
hold each in context — note only the highlights that inform your next edit.

```bash
PYTHONPATH=src $AT_PYTHON -m training.participants.auto_tune step \
  --session <name> --command '{"action": "start"}'
# Loop continue until a supervisor-decision event or run completion:
PYTHONPATH=src $AT_PYTHON -m training.participants.auto_tune step \
  --session <name> --command '{"action": "continue"}'
```

**Supervisor-decision events** — `ratify_request`, `escalation`, or any event
requiring `ratify` / `scaffold` / `goal` / any non-`continue` answer: **stop
the loop.** This is you acting as supervisor. In the session window: read the
event (for `ratify_request`, use its `misfit` + `curriculum_context`
enrichment), state which command is correct and **why**, then emit exactly one
command. Repeat, turn by turn, per decision event.

Your decisions are audited to `auto-tune/<name>/commands.jsonl` (one line per
command, with the event seq it answers). That file is the **protocol record**;
the **session window** is the **reasoning record**. Both belong together — a
command in the audit with no matching reasoning in the window is a process
failure.

### c. Snapshot

```bash
PYTHONPATH=src $AT_PYTHON -m participants.auto_tune snapshot --session <name>
```

### d. Observe → Update State

Update state: **Current Phase** → `observing`.

1. Read `auto-tune/<name>/harness.log` — focus on errors, warnings, and the final outcome
2. Read `auto-tune/<name>/events.jsonl` — scan for key event types (escalation, rationalise, ratify_request)
3. Compare snapshots if a previous run exists — diff the harness.log

**Then immediately write your observations to `session-state.md`:**

- Add a new run entry (latest run at top, full template)
- Collapse the previous latest run to a one-liner
- Update **Next Action**, **Current Phase**, and **Patterns & Notes**

**Do this update before deciding what to do next.** The state file must always reflect reality.

### e. Decide

Read `session-state.md` (your fresh update) and decide:

- If the goal is met → go to step 4 (Document)
- If there are crashes → fix (see Rule 1), update state, then go to f
- If there's an improvement to try → snapshot, edit code, commit, update state (**Files Modified**), then go to f
- If 3 runs pass with no meaningful improvement → surface this to the user and ask whether to continue

### f. Reset for next run

Update state: **Current Phase** → `resetting`.

```bash
PYTHONPATH=src $AT_PYTHON -m participants.auto_tune stop-supervisor --session <name>
PYTHONPATH=src $AT_PYTHON -m participants.auto_tune stop-harness --session <name>
PYTHONPATH=src $AT_PYTHON -m participants.auto_tune reset --session <name>
```

Then return to step a.

## 4. Document (once, at end)

Every auto-tune session must consolidate existing documentation:

1. **Spec** (`specs/<existing>.md`) — behavioural rules, test matrix, definitions derived from what you observed. Reference the auto-tune session directory as evidence.

2. **Plan** (`plans/impl/<existing>.md`) — implementation tasks, test mapping, design decisions. Include:
   - A link to the auto-tune session directory as evidence
   - Test mapping with status (✅ for passing tests, ☐ for not yet written)

3. **Tests** (`tests/test_<feature>.py`) — tests covering every spec criterion. Use `caplog` for log assertions, `BusCapture` for bus message capture. Follow patterns from existing test files.

4. **Commit** — commit everything on the auto-tune branch with a descriptive message. DO NOT MERGE INTO MAIN.

5. **Teardown** (optional) — if the session is complete and the user doesn't want to keep the worktree:

```bash
cd <main-repo-root>
PYTHONPATH=src .venv/bin/python -m participants.auto_tune teardown --session <name>
```

Update state: **Current Phase** → `complete`.

### Documentation Rules

- The spec must be derivable from what you observed during auto-tune runs, not invented
- The plan must reference the actual code changes made during the session
- Tests must cover spec criteria (TL-1 through TL-N style IDs)
