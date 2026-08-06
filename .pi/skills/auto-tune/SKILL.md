---
name: auto-tune
description: Drives an auto-tune session to tune Kalvin's rationalisation behaviour (the significance model) using repeated training runs, observation, code edits, and documentation updates. Use when the user says "/auto-tune" or asks to auto-tune, tune, or iterate on the codebase using training runs. Establishes a goal, runs training sessions, observes results, edits the significance-model code and owning spec together, and re-runs to confirm.
---

# Auto-Tune

Tune **Kalvin's significance model** — how `expand()`, `significance.py`, the
rationaliser, and the cascade produce significance and proposals from a kline
and a candidate pool — by running a curriculum, reading what Kalvin actually
did, changing the model so it does better, and re-running.

A curriculum exercises the reactor/cogitator/rationaliser under controlled
conditions. You observe how Kalvin rationalises, then change the system so it
rationalises better. The thing being tuned is the significance model: the code
**and** the owning spec that defines its intended semantics, evolved together.
When a run shows Kalvin scoring something S4 that the curriculum intends to
produce an S2/S3 proposal, that is the signal that the model's semantics are
the thing to change — state the intended semantics, edit the code and the spec
together, re-run to confirm. This is the core activity; a session whose goal is
a model-semantics change is the norm, not an exception.

## The arbiter

Every run produces a verdict: `auto-tune summary --session <name>`. **Read it
before reading code.** It aggregates the run into:

- **`outcome`** — one of `completed`, `deadlocked`, `supervisor-stalled`,
  `stalled`, `crashed`, `incomplete`. This is the termination signal.
- **`significance`** — the S1–S4 histogram over the run's rationalise events,
  plus ground/frame counts. This is what Kalvin actually did.
- **`entries_total` / `entries_satisfied` / `entries_deadlocked`** — whether
  the run's entries resolved.
- **`diagnosis`** — a one-line pointer at the file/concept to inspect first.

The summary is produced from file state only, so it works after any run — live
or deadlocked — without touching the harness. It replaces a hand-maintained
state file: run-state is machine-produced, not something you re-derive or
re-transcribe each loop.

| Outcome              | Meaning                                                                   | What you do                                                                                                                                                                                                                                                            |
| -------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `completed`          | Terminal completion fired                                                 | Read the significance profile → judge whether the goal was actually met                                                                                                                                                                                                |
| `deadlocked`         | Run ended with submitted-but-unsatisfied entries, no completion           | The satisfaction model has no path for those entries — diagnose the gap                                                                                                                                                                                                |
| `supervisor-stalled` | A `ratify_request` sat unanswered (you didn't enact a decision)           | You abdicated the supervisor role — see §Supervisor decisions below                                                                                                                                                                                                    |
| `stalled`            | Connected but frozen: unsatisfied work, no events for the stall threshold | **Stop driving.** The trainer's satisfaction accounting has deadlocked — `step` will only re-poll a dead stream. Stop the run, read `training.harness.log` + the last lesson's compiled entries, diagnose why submitted work isn't rationalised, fix the model, re-run |
| `crashed`            | Error in the stream                                                       | Reproduce, fix, re-run before interpreting anything else                                                                                                                                                                                                               |
| `incomplete`         | Run still in progress and still moving                                    | Keep driving it                                                                                                                                                                                                                                                        |

## Where things live

| Artefact        | Path                                                                          | Role                                                                                                                                                                            |
| --------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Significance    | `src/kalvin/expand.py`                                                        | `expand()` — composes significance from query + candidate                                                                                                                       |
| Significance    | `src/kalvin/significance.py`                                                  | Band layout (S1–S4), `BandLayout.classify`                                                                                                                                      |
| Rationaliser    | `src/kalvin/rationaliser.py`                                                  | Candidate retrieval + slow-path cogitation fan-out                                                                                                                              |
| Reactor         | `src/training/trainer/reactor.py`                                             | Mechanical S2/S3: auto-countersign, recurrence, decision escalation                                                                                                             |
| Trainer         | `src/training/trainer/trainer.py`                                             | Lesson loop, satisfaction accounting, decision gate                                                                                                                             |
| Harness adapter | `src/training/harness/adapter.py`                                             | Rationaliser ↔ bus wiring                                                                                                                                                       |
| Curricula       | `curricula/*.md` (+ `.json`)                                                  | Authored curricula. **Read the chosen `.md` in full before step 1** — it states the Objective, Goal, Approach, and per-lesson intended routing that defines your done-criteria. |
| Auto-tune CLI   | `src/training/auto_tune/`                                                     | Session lifecycle, `summary` aggregation                                                                                                                                        |
| Run artefacts   | `auto-tune/<name>/{events.jsonl,run-summary.json,training.harness.log,runs/}` | Per-run evidence (run-summary is the arbiter)                                                                                                                                   |

Spec and docs (read for what the code _means_; do not re-derive in comments):

- `specs/model.md`, `specs/rationaliser.md` — the significance model's intended semantics
- `specs/auto-tune.md` — the harness/CLI contract (subcommands, run-summary object, rules)
- `specs/supervisor-decision.md` — the decision gate and what each decision means
- `CONTEXT.md` — domain glossary

## Workflow

1. **Establish the goal — read the curriculum first.** Read the chosen
   `curricula/<name>.md` in full **before** deciding anything else. Its
   Objective, Goal, and Approach already state the intended rationalisation
   behaviour; lift your goal and done-criteria from there (an observable
   outcome in the summary — e.g. "lessons 3–5 produce S3 proposals with zero
   deadlocks") rather than authoring them from the filename. Note the per-lesson
   intended routing — that is the ground truth you'll judge the significance
   histogram against.

2. **Run → read the summary → act.** Drive a run (see
   [lifecycle.md](references/lifecycle.md)); when it ends, read `summary`. The
   `outcome` tells you what to do next:
   - `completed` and the significance profile matches the curriculum's
     intended routing → Document. Judge the histogram against the per-lesson
     expectations you noted in step 1, not against a remembered number — watch
     in particular for everything collapsing to S1 fast-path (the curriculum
     exercised nothing). To confirm _which_ klines landed (the histogram only
     gives the S-level distribution), match the goal against the event's
     `query.values`/`proposal.values` — never the `for_display` label, which
     is a decode and may mislead on a packed signature.
   - `crashed` → reproduce with a minimal test, fix, verify, re-run.
   - `deadlocked` → the diagnosis pointer names the gap; state the intended
     semantics, edit the model code **and** the owning spec together, re-run.
     This is the expected, primary path — a deadlock is a finding, not a stall.
   - `supervisor-stalled` → you stopped supervising (see below); resume properly.
   - `stalled` → the run is frozen mid-stream (see §Stalled runs). Stop
     driving, stop the processes, diagnose the satisfaction deadlock from
     `training.harness.log` and the compiled entries, fix the model, re-run.
   - A real improvement to try → snapshot, edit, commit, re-run.
   - Genuine forks → go with your best shot. If it doesn't work out try the next fork.
     **Keep going**.

3. **Commit each meaningful change** on the `auto-tune/<name>` branch. Never
   merge into main. Reference the session name in the commit message. When you
   change model behaviour, update the owning spec in the same change.

4. **Document at the end.** Finalise the owning spec (the intended semantics,
   grounded in observed evidence), the plan, and tests covering the spec
   criteria. Reference the session directory as evidence.

### Snapshot before each change

```bash
PYTHONPATH=src $AT_PYTHON -m training.auto_tune snapshot --session <name>
```

Snapshots give you before/after comparisons across runs.

## Stalled runs

A run can freeze without ending: the supervisor stays connected, the stream
simply stops, and `step` blocks its full timeout on every call because no event
is coming. This is a trainer-side satisfaction deadlock — submitted entries
that never get rationalised, so the lesson can never complete (a catch-22:
completion needs the events, the events never arrive). The arbiter surfaces
this as `outcome: stalled` (connected + unsatisfied work + idle past the
threshold) so you do **not** mistake it for a busy run and churn `step`.

When you see `stalled`:

1. **Stop driving.** Further `step`/`continue` calls re-poll a dead stream.
2. **Snapshot** the frozen state for before/after evidence.
3. **Read `training.harness.log`** — the last entries show what the trainer
   emitted and where it went silent (e.g. a flood of auto-countersigns then
   nothing, meaning some entries never produced a rationalise event).
4. **Recompile the stuck lesson** (`compile_source` on its kscript) and
   compare the compiled entries against the events that _did_ fire — the
   entries absent from the stream are the ones the rationaliser dropped.
5. **Diagnose and fix the model** (the rationaliser / significance / reactor),
   update the owning spec, `reset`, re-run. A stall is a finding about the
   model, same as a deadlock.

## Supervisor decisions (Pi-in-the-Loop)

You are the supervisor participant. The CLI supervisor process is a protocol
relay — it does not think. A `ratify_request` event is a decision the model
cannot make mechanically (auto-countersign didn't match; recurrence didn't
apply): it surfaces a proposal and asks _you_ whether to ratify, scaffold, or
skip. Each carries the proposal, the misfit diagnosis, and the curriculum
context — read those, decide, and emit one command. Decide against the
curriculum's stated intent (e.g. `s3-auto-countersign` asserts "zero LLM,
zero ratify" as its whole point — ratifying there corrupts the measurement),
not against the histogram alone. Routine observation events
(`rationalise`, `progress`, `ground`) need no action; advance them.

The arbiter makes supervision observable: `supervisor-stalled` means a
`ratify_request` sat unanswered, and `ratify_requests` in the summary counts
how many decisions arose. If the count is high and `proposals_ratified` is
near zero, proposals surfaced but you enacted none — the run had no closure
path through you. Read the events, enact decisions deliberately, and the
summary on the next run will reflect it.

See [commands.md](references/commands.md) for the command vocabulary.

## Stop / ask

- **Stop** when `outcome: completed` **and** the significance profile satisfies
  the goal — then Document.
- **Ask the user** only on a genuine fork the arbiter surfaces but cannot
  resolve: the intended semantics are ambiguous and you need a decision about
  _what_ Kalvin should do. Do not ask about _whether_ you're allowed to change
  the model — that is auto-tune's core activity.
- If 3 runs pass with no movement in the summary **and** no identifiable
  semantics gap, surface this and ask whether to continue. Do not use this
  branch to escape a gap you _have_ identified.

## Entry points

**New session.** `init` a worktree (see [lifecycle.md](references/lifecycle.md)
for setup commands), then **read the curriculum `.md` in full** and establish
the goal from it (Workflow step 1). Create a one-line session record (goal +
curriculum + done-criteria) at `auto-tune/<name>/session-state.md`.

**Resume session.** `cd .worktrees/auto-tune/<name>/`, read
`session-state.md` for the goal, **re-read the curriculum `.md`** to recover
the per-lesson intended routing, run `summary` for the current run's verdict.
Continue from the outcome.

**Context handoff.** If context is large, update `session-state.md` (goal +
latest run's summary verdict + next action) and run `/auto-tune-handoff` or
start fresh with "resume auto-tune <name>". The `auto-tune-handoff` extension
monitors context usage and will steer you to do this at the ceiling.
