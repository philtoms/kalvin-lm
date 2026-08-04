# Auto-tune prompt set for kb

A tailored set of kb agent prompts (triage / executor / reviewer / merge) that
decomposes an **auto-tune** session into a chain of small, dependent tasks split
across separate kb agents. Derived from `.pi/skills/auto-tune/SKILL.md`, but
**re-projects** the skill onto kb's native task model rather than reproducing it
verbatim.

This set **does not replace** the repo's default cascade prompts in
`.kb/prompts/` — it lives alongside them so you can opt in per auto-tune
session. The default prompts remain active for ordinary cascade work.

## Two deliberate departures from SKILL.md

The skill was written for a single pi session doing the whole loop in one
context window. Under kb, two of its assumptions no longer hold:

### 1. kb tasks ARE the state — no `session-state.md`

SKILL.md invents `auto-tune/<name>/session-state.md` as a hand-rolled blackboard
for surviving context resets. **Under kb that blackboard is redundant** — the
task system is built for exactly this and survives agent/process boundaries
natively. Every session-state field maps onto a task-system feature:

| session-state.md field        | kb task equivalent                                      |
|-------------------------------|---------------------------------------------------------|
| Goal / Done Criteria          | task `description` + PROMPT.md Mission                  |
| Run Log (per-run observation) | `task_log()` entries (timestamped, append-only, shared) |
| Patterns & Notes              | `task_log()` entries                                     |
| Files Modified                | `task_log()` + the git diff                             |
| Current Phase                 | step `status` + `currentStep`                           |
| Next Action                   | the `depends` chain + the next task's Mission           |
| Session identity (name, etc.) | task `description` / PROMPT.md (static, per-task)       |

So this prompt set **never creates or maintains `session-state.md`**.
Observations, diagnosis, and diffs go to `task_log()`. Resuming = read the last
task's log + spawn the next task with a `depends` edge. This removes a redundant
shared mutable file and the second-writer sync problem it created.

### 2. Trailing cascade documentation is deferred, not baked in

SKILL §4 makes a DOC step (finalise spec Test Matrix, plan test-mapping, write
tests, session-dir evidence links) mandatory at the end of every session. Under
kb that is **trailing consolidation** that can be applied later, once the user
is satisfied a change is worth keeping. So this set does **not** auto-emit a DOC
task.

A crucial boundary, though: the **owning-spec edit that defines a model-semantics
change is NOT documentation** — it is the "intended semantics" half of the change
and stays coupled to the code edit inside EDIT-N. Only the *trailing* DOC work
(plan test-mapping table, full Test Matrix sweep, test files) is deferrable, and
it spawns as an ordinary cascade task when the user asks.

## Why a separate set at all

Auto-tune is a strict, serial loop. The git worktree and the running harness/
supervisor processes are **shared mutable state**. kb's strength here is its
`depends` chain: by splitting the loop into INIT → RUN-N → EDIT-N tasks, each
phase runs in its own agent with a tight, phase-specific contract, while the
dependency edges guarantee only one agent touches the shared state at a time.

The default kb prompts are tuned for cascade code tasks (specs/plans/code) and
have no notion of: the run loop, the pi-in-the-loop supervisor discipline, or
the code+spec-together rule for significance-model changes. This set encodes
those.

## The task decomposition (what triage emits)

```
INIT ──▶ RUN-1 ──▶ EDIT-1 ──▶ RUN-2 ──▶ EDIT-2 ──▶ …   (DOC only on request)
```

| Phase   | Depends on        | Owns                                                          | Must not do                          |
|---------|-------------------|---------------------------------------------------------------|--------------------------------------|
| INIT    | —                 | `init`, baseline run, snapshot, log baseline via task_log()   | edit src/ or specs/                  |
| RUN-N   | prev RUN or EDIT  | drive one run, snapshot, log observation via task_log()       | edit src/ or specs/                  |
| EDIT-N  | RUN-N             | diagnose, state intended semantics, edit code **+** owning spec, commit | split the code/spec semantics change |
| DOC     | last EDIT         | finalise spec/plan/tests (ordinary cascade task)              | be auto-emitted; merge to main       |

## Activation

kb resolves each agent prompt in this order (first existing source wins):

1. `$KB_AGENT_PROMPT_{TRIAGE|EXECUTOR|REVIEWER|MERGE}` — file path or raw text
2. `${cwd}/.kb/prompts/{agent}.md` — the active project set
3. `~/.pi/agent/kb-prompts/{agent}.md` — global set
4. kb's builtin

### Option A — env vars (recommended; non-destructive, per shell)

From the repo root:

```bash
for a in triage executor reviewer merge; do
  export KB_AGENT_PROMPT_$(echo $a | tr a-z A-Z)="$(pwd)/.kb/prompts-auto-tune/$a.md"
done
```

### Option B — copy/symlink over `.kb/prompts/`

```bash
for a in triage executor reviewer merge; do
  ln -sf ../prompts-auto-tune/$a.md .kb/prompts/$a.md
done
```

Restore the originals from git when done.

## Files

- `triage.md` — decomposes an auto-tune goal into the chained task structure.
- `executor.md` — phase-aware execution: worktree discipline, task-log-as-state,
  pi-in-the-loop supervisor reasoning, code+spec-together edits.
- `reviewer.md` — checks phase contracts and code↔spec coherence; audits the
  decision-event reasoning trail; rejects revived `session-state.md`.
- `merge.md` — keeps significance-model code/spec changes paired through a merge.

## Source of truth

The auto-tune loop's *behaviour* is defined in `.pi/skills/auto-tune/SKILL.md`.
These prompts are a *projection* of that skill onto kb's task model, with two
deliberate departures noted above. If the loop's behaviour changes, update the
skill first, then this set. (The skill's `session-state.md` mechanism remains
correct for non-kb, single-session use; this set simply doesn't use it.)
