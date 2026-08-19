---
name: dialogue-dev
description: Investigates and progresses the dialogue sub-project src/dialogue/ — the authored-script ↔ real-actor ↔ rules triad and the coverage/displacement loop that brings them into agreement. Use when the user says "/dialogue-dev" or asks to work on, debug, advance, tune, or understand the dialogue training system, the rationalising trainee, the synthesizing trainer, the runner, or dialogue scripts.
---

# Dialogue Dev

Guide for investigating and progressing the dialogue sub-project:
`src/dialogue/`.

## Conceptual model

`src/dialogue/` is the sub-project. **Read the code for what it means; do not
re-derive or reassert it in code comments or commits.** `CONTEXT.md`'s
Dialogue subsystem section maps the modules (actors, runner, rationaliser,
supervisor, decoder). This skill is navigation and discipline only.

## The two signals

Every run produces two numbers. Read both before reading code.

- **Displacement** (`uncovered`) — agreement: coverage rows never emitted.
  Zero is the target. Zero at a tiny event count is a stall, not a success.
- **Escalation load** (`supervisor escalations: N asks, M emitted`, only with
  `--rationalise-trainer` / `--rationalise-both`) — depth: how often the
  rationalising trainer had nothing to say and asked the supervisor. Lower is
  the trajectory; it is the real work, not a bug to "fix" by forcing the
  supervisor to speak.

## Where everything lives

| Artefact     | Path                                                | Role                                                                                                                   |
| ------------ | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Actors       | `src/dialogue/actors.py`                            | `ScriptTrainer`/`ScriptTrainee` (table), `SynthesizingTrainer`, `RationalisingTrainee` (K), `RationalisingTrainer` (T) |
| Runner       | `src/dialogue/runner.py`                            | Bus subscriber + driver; opens/closes a run; coverage/divergence                                                       |
| Rationaliser | `src/dialogue/rationalise.py`                       | Pure shared engine: `(state, incoming) -> (batch, observations)`                                                       |
| Supervisor   | `src/dialogue/synthesize.py`                        | `synthesize` — answers from compiled source when cogitation has nothing                                                |
| Decoder      | `src/dialogue/decoder.py`                           | Script → `list[DecodedTurn]`; resolver, not gatekeeper                                                                 |
| Driver       | `dev/dialogue/dialogue_run.py`                      | End-to-end CLI; renders trace, displacement, escalation load                                                           |
| Probe        | `dev/dialogue/probe_rationalise.py`                 | Drives the pure engine turn-by-turn; edit in place per question                                                        |
| Scripts      | `scripts/dialogue-mhall.json`, `dialogue-wdmh.json` | Authored dialogues (mhall is canonical)                                                                                |
| Smoke test   | `tests/test_dialogue_smoke.py`                      | Basic-operation acceptance (DDT-1..3)                                                                                  |

## Commands

```bash
# End-to-end (reach for this first)
PYTHONPATH=src .venv/bin/python dev/dialogue/dialogue_run.py                            # both table actors
PYTHONPATH=src .venv/bin/python dev/dialogue/dialogue_run.py --rationalise              # real trainee, table trainer
PYTHONPATH=src .venv/bin/python dev/dialogue/dialogue_run.py --rationalise-trainer      # real trainer, table trainee
PYTHONPATH=src .venv/bin/python dev/dialogue/dialogue_run.py --synthesize               # synthesizing trainer, table trainee
PYTHONPATH=src .venv/bin/python dev/dialogue/dialogue_run.py --rationalise-both         # both rationalising (read escalation load)
PYTHONPATH=src .venv/bin/python dev/dialogue/dialogue_run.py --rationalise-both -v      # + groundings + supervisor-supplied turns
PYTHONPATH=src .venv/bin/python dev/dialogue/dialogue_run.py --divergence               # fail (exit 1) on first divergence

# Turn-by-turn engine probe (edit in place)
PYTHONPATH=src .venv/bin/python dev/dialogue/probe_rationalise.py

# Smoke test (DDT-3: canonical MHALL with table actors must stay zero displacement)
PYTHONPATH=src .venv/bin/python -m pytest tests/test_dialogue_smoke.py -q
```

Flags are orthogonal and combinable, except `--rationalise-both` (shorthand
for both rationalising actors; exclusive with the others). Default mode
accepts divergences (exit 0) and reports them in the trace; `--divergence`
fails fast. A script with `priors` runs them as a sequence of runs before
the target's own run.

## Workflow

1. **Reproduce before reading code.** Run the dialogue; read both signals.
2. **Diagnose** — which artefact is wrong (script / code / rules), and is the
   turn earned, escalated, or scripted? Read the code, not its comments, for
   what these mean.
3. **Edit** — smallest honest change. Keep code and comments minimal: describe
   _what_ a block does, not a theory of why the design is right. Don't mirror
   the spec's prose into the code.
4. **Verify** — re-run; snapshot both signals before and after.

## Guardrails (do not violate)

- The runner opens a run (delivers the first row to the opposite role) and
  closes it (close observed / coverage exhausted / mutual PASS). Actors never
  open.
- Every `accept` owes `burst >= 1`; nothing substantive publishes a PASS; two
  consecutive PASSes (one per role) is terminal.
- DDT-3 (canonical MHALL, table actors) must stay zero displacement — the
  core-loop guard.
- Doc maintenance follows `AGENTS.md` (locate → assess → update source and
  CONTEXT → report). The owning layer for this sub-project is `src/dialogue/`
  (mapped in CONTEXT.md); do not duplicate its content into code comments,
  commits, or other docs.

## Escalate to the user when

- A turn can't be earned without a genuine design fork in the engine.
- A change would alter a dialogue contract in `src/dialogue/`.
- DDT-3 regresses.
- Escalation load stalls across three runs.
