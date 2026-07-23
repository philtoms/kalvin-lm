---
name: dialogue-dev
description: Investigates and progresses the dialogue sub-project src/training/dialogue/ — the authored-script ↔ real-actor ↔ rules triad and the coverage/displacement loop that brings them into agreement. Use when the user says "/dialogue-dev" or asks to work on, debug, advance, tune, or understand the dialogue training system, the rationalising trainee, the synthesizing trainer, the runner, or dialogue scripts.
---

# Dialogue Dev

Guide for investigating and progressing the dialogue sub-project:
`src/training/dialogue/`.

## Direction

The sub-project moves a **script ↔ code ↔ rules** triad toward agreement *and*
toward depth.

- **Agreement** — the actors produce the authored exchange (zero displacement).
- **Depth** — the actors produce it from their own understanding, not from an
  oracle handed to them.

Every turn an actor emits is **earned** (the `Rationaliser` engine derives it
from its own state), **escalated** (the `synthesize` supervisor supplies it
because the engine had nothing to say), or **scripted** (the table cursor).
Move turns down that ranking. Displacement measures agreement and is already
zero on the canonical run; **escalation load measures depth and is where the
work is.**

The engine is role-neutral and shared by both actors. It earns the trainee's
bands (S3/S4, the S2 similar-fit) but not yet the trainer's (S2 canon-reply,
S1 ratification, S2 opening) — those are the mirror paths to grow. Today's
trainer escalates ~all its substantive turns; that is the deficiency, not the
design.

The script is **not a golden master.** A turn the code or rules cannot
honestly produce is a defect in one of the three, not a target to
reverse-engineer. The same applies to the supervisor: don't "fix" an
unearned turn by making `synthesize` say it — that preserves shallow
agreement. Earn it in the engine, or accept the escalation as honest.

## Where everything lives

| Artefact | Path | Role |
| --- | --- | --- |
| Actors | `src/training/dialogue/actors.py` | `ScriptTrainer`/`ScriptTrainee` (scripted), `SynthesizingTrainer` (escalated), `RationalisingTrainee` (earned, K), `RationalisingTrainer` (earned routing / escalated speech, T) |
| Runner | `src/training/dialogue/runner.py` | Bus subscriber + driver, coverage/divergence/PASS |
| Rationaliser | `src/training/dialogue/rationalise.py` | Pure shared engine: `(state, incoming) -> (batch, observations)` |
| Supervisor | `src/training/dialogue/synthesize.py` | `synthesize` — the oracle the engine shrinks |
| Decoder | `src/training/dialogue/decoder.py` | Script → `list[DecodedTurn]` (resolver, not gatekeeper) |
| Driver | `dev/dialogue/dialogue_run.py` | End-to-end CLI; renders trace, divergence, escalation load |
| Probe | `dev/dialogue/probe_rationalise.py` | Drives the pure engine turn-by-turn; edit in place per question |
| Scripts | `scripts/dialogue-mhall.json`, `dialogue-wdmh.json` | Authored dialogues (mhall is canonical) |
| Smoke test | `tests/test_dialogue_smoke.py` | Basic-operation acceptance (DDT-1..3) |
| Contracts | `specs/dialogue-driven-training.md`, `specs/dialogue-cogitation.md` | Working sketches — replace, don't augment |
| Build records | `plans/implement-*.md` | What was built + rationale (not active contract) |

## Workflow

1. **Reproduce before reading code.** Run the dialogue, read both signals.
2. **Read both signals** — displacement (agreement) and escalation load (depth).
   A zero-displacement run at `14 asks, 14 emitted` is a stall, not a success.
3. **Diagnose** — pick channel (dialogue / grounding), artefact (script / code
   / rules), and tier (earned / escalated / scripted).
4. **Decide** — which artefact is wrong, and is the turn earned or honestly
   oracle-dependent?
5. **Edit** — smallest honest change; snapshot both signals before and after.
6. **Document** — single owning layer only; ask first per the coding-activity rule.

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
fails fast.

## Invariants

- The engine is **stateless about its own emissions** — dedup lives in the actor.
- The engine is **oracle-free** — it never reads the compiled source or table.
  A path that needs them is not yet earned; it belongs behind the supervisor.
- The decoder is a **resolver, not a gatekeeper** — it builds the declared
  kline and does not check signature/nodes consistency.
- The runner is **tier-blind** — it polices coverage and grounding only; depth
  is the actors' bookkeeping (e.g. `supervisor_escalations()`).
- Every `accept` owes `burst >= 1`; nothing substantive publishes a PASS;
  two consecutive PASSes (one per role) is terminal.
- The script is **de-positional** — the first row carries no opening semantics.

## Escalate to the user when

- A turn can't be earned without a genuine design fork in the engine.
- A change would alter a contract in `specs/`.
- DDT-3 (canonical MHALL, table actors) regresses — the core-loop guard.
- Escalation load stalls across three runs — reconsider the mirror's shape,
  or whether the remaining escalations are honest.
