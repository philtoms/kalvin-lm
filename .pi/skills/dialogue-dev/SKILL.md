---
name: dialogue-dev
description: Drives the dialogue harness (src/dialogue/harness.py + engine.py) — the synchronous, non-judging compile→feed→present loop — to tune Kalvin's rationalising engine against .ks scripts, and to author scripts that test engine theory or bring out new behaviour. Use when the user says "/dialogue-dev".
---

# Dialogue Harness

Tune **Kalvin's rationalising engine** by running a script
through the dialogue harness, reading what the engine actually did from the
trace + the grounded/work_list tails, changing the engine so it does
better, and re-running. The engine is the target of the work; the
The script is the lever. The harness is a faithful, non-judging presenter.

## The three signals

Every run ends with a trace and two lists. Read all before reading code.

- **trace** - What the engine did (asks, proposals, groundings)
- **grounded** — what the engine knows (identities, canons, relationships).
- **work_list (pending at end of run)** — what the engine was still working on
  when turns ran out. Distinguish _genuine residue_ (signatures the
  script never makes groundable — e.g. an unbound `L`) from _stalled
  klines_ (something that should have grounded but the engine had no
  path). The latter is the work.

## Run

```bash
PYTHONPATH=src .venv/bin/python -m dialogue.harness data/scripts/mhall.ks             # the canonical kscript
PYTHONPATH=src .venv/bin/python -m dialogue.harness data/scripts/mhall.ks -e          # structural supervisor
PYTHONPATH=src .venv/bin/python -m dialogue.harness data/scripts/wdmh.ks -p data/dialogue/mhall.json # persistent memory
```

Each lesson's kscript runs through a shared engine (state persists across
lessons), and the trace is presented.

```

The harness compiles each lesson's kscript, feeds each compiled entry to the engine
one at a time, answers S4 identity asks inline , and
presents the trace. One linear pass, no waits, no convergence loop.

## Workflow

1. **Read the script.** and understand what it is saying. Use the annotations -
   they explain what is being taught or asked.
2. **Run the script.** Read grounded + work_list. Form an
   **exploratory expectation** from the annotation + what has grounded —
   something to compare the trace against and explore, not a pass/fail
   spec. Write it down so you can compare later. A near-match (right nodes,
   different signature) is a finding that the engine is on the right track, not a
   failure. Then **report** what you found: did the engine meet the expectation, how
   was it out, what went wrong — feeding the next step.
   Judge the decoded prose honestly: if the engine's proposal is semantic
   gibberish, say so — a near-match is only "on the right track" when
   its decoded prose actually means something.
3. **Diagnose: engine or script?** (See below. Engine first.)
4. **Edit** the suspect (engine code, or — only when confident — the
   `.ks`). Smallest honest change. Comments minimal: describe what, not
   why the design is right.
5. **Verify.** Re-run; compare grounded + work_list before/after.
6. **Commit on the `dialogue` branch.** One change per commit; name what
   shifted.

## Diagnose: engine or script?

**Engine first — always.** This is the target work. When a trace stalls,
a kline vanishes, or grounded/work_list diverge from the trainer
expectation, suspect the engine and dig there until the engine path is
genuinely exhausted.
  See [trace-reading.md](references/trace-reading.md) for the per-step
  vocabulary and a worked reading, and

**Retreat to script-authoring only when** you are confident a `.ks`
change can test a theory or bring out a different result. Script writing
advances the use case for the engine — it is not a workaround for
engine bugs. Typical retreats:

- **Prime-before-test ordering.** A question fed before the prime's
  identities land can't be answered (the parts aren't known yet). Reorder
  the `.ks` so priming precedes questioning, to test whether the engine
  _can_ answer given correct ordering. (mhall's WDMH is the canonical
  example — the question arrives at step 9, the identities at steps 13+.)
- **Shake things up.** Once the engine settles on a script, author a new
  new `.ks` to find the next edge — introduce ambiguity, withhold an
  identity the harness could offer, add a second question, invert
  ordering deliberately.
- **Test a theory.** A minimal `.ks` constructed to exercise one engine
  path (a single `==` ask with its goal; a lone unseen canon; a denotes with no
  reciprocal) is the fastest way to confirm or refute a hypothesis about
  engine behaviour.

When you do author a `.ks`, it lives in `data/scripts/`. Annotation
prose (parenthetical lines) is the trainer rationale — the harness
carries it onto each kline's `KDbg.annotation` and prints it as a section
header. Always compile the `.ks` to ensure there are no errors.
  See [script-reading.md](references/script-reading.md) for what a `.ks` script
  is, how to read it semantically, and how to construct new ones.

## Discipline (do not violate)

- **The harness never judges.** No verdict, no band-matching, no
  success/failure marker. Judgement is the trainer's (you, outside the
  loop). If you are adding evaluation logic to the harness, stop.
- **Engine first.** Suspect the engine before the script; edit `.ks`
  only to test a theory or shake things up, never to work around an
  engine bug.
- **Check the source before asserting behaviour as fact.** Inferences
  stated as established rules cause real bugs.
- **The engine speaks in semantic predicates** (`is_identity`,
  `is_unknown`, `is_canon`, `is_relationship`), never raw `kline.nodes`.
- keep all ad-hoc investigative scripts in dev/dialogue. Do not delete after use.

## Where things live

| Artefact        | Path                          | Role                                                     |
| --------------- | ----------------------------- | -------------------------------------------------------- |
| Engine          | `src/dialogue/engine.py`      | The fork under tune                                      |
| Harness         | `src/dialogue/harness.py`     | The non-judging compile→feed→present loop + CLI          |
| Compiler        | `src/ks/`                     | KScript → KValue; carries annotation/scope/labels        |
| Curricula       | `data/scripts/*.ks`           | `mhall.ks` is canonical                                  |

## Stop / ask

- **Stop** when grounded + work_list match the trainer expectation and
  the suite passes.
- **Ask the user** only on a genuine engine-semantics fork (e.g.
  "should a `==` goal require compositional operands, or just two
  grounded values?"). Do not ask permission to treat engine behaviour as
  suspect — that is the core activity.
```
