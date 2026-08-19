---
name: lean-harness
description: Drives the lean dialogue harness (src/dialogue/harness.py + engine.py) — the synchronous, non-judging compile→feed→present loop — to tune Kalvin's rationalising engine against a .ks curriculum, and to author curricula that test engine theory or bring out new behaviour. Use when the user says "/lean-harness".
---

# Lean Harness

Tune **Kalvin's rationalising engine** by running a curriculum
through the lean harness, reading what the engine actually did from the
trace + the grounded/work_list tails, changing the engine so it does
better, and re-running. The engine is the target of the work; the
curriculum is the lever. The harness is a faithful, non-judging presenter.

## Before you start — read, do not infer

The discipline of this skill: read before you reason. Each artefact below
exists because a session learned the hard way that guessing costs more
than reading. Load them every session.

1. **`CONTEXT.md`** — the domain glossary. Every term (kline, rationalise,
   significance bands, grounding, frame, work-list, terminal, canon,
   misfit, relationship) has a precise meaning; several are counter-
   intuitive. Do not infer them.
2. **`docs/behaviour-notes.md`** — read **§Rules Uncovered** for settled behaviour,
   and **§Active state of K** for the current frontier (not yet rules).
   Re-read every session; both change. ⚠️ marks a rule under active
   suspicion — the boundary between the two sections.
3. **`src/dialogue/engine.py`** — the engine under tune (a fork of
   `dialogue/rationalise.py`, which stays as the verbatim reference).
4. **`src/dialogue/harness.py`** — the loop. Read to confirm the
   non-judging contract before changing anything.

See [trace-reading.md](references/trace-reading.md) for the per-step
vocabulary and a worked reading.

## The privileged working mode

**Do not assume existing code is correct.** This contradicts "source is
the truth document," but here the source is exactly what we are improving.
Engine behaviour that drops or deadlocks (the fast route dropping unseen
canons; relationships that couldn't ground) was the bug, not the
contract. When a trace shows something vanish or stall, treat the engine
as the suspect first.

## The two signals

Every run ends with two lists. Read both before reading code.

- **grounded** — what K knows (identities, canons, relationships).
- **work_list (pending at end of run)** — what K was still working on
  when turns ran out. Distinguish _genuine residue_ (signatures the
  curriculum never makes groundable — e.g. an unbound `L`) from _stalled
  klines_ (something that should have grounded but the engine had no
  path). The latter is the work.

## Run

```bash
PYTHONPATH=src .venv/bin/python -m dialogue.harness curricula/wdmh-underfit.md  # the canonical curriculum
PYTHONPATH=src .venv/bin/python -m dialogue.harness curricula/wdmh-underfit.md -v # + hex signatures alongside labels
PYTHONPATH=src .venv/bin/python -m dialogue.harness data/scripts/mhall.ks       # or any raw .ks script
```

The harness dispatches on suffix: `.ks` runs the raw script; markdown is
parsed as a curriculum (Objective/Approach/Goal/Lessons),
each lesson's kscript runs through a shared engine (state persists across
lessons), and the trace is presented.
```

The harness compiles each lesson's kscript, feeds each compiled entry to the engine
one at a time, answers S4 identity asks inline from the curriculum, and
presents the trace. One linear pass, no waits, no convergence loop.

## Workflow

1. **Read** CONTEXT.md + behaviour-notes (Rules and Active state of K).
   Note any ⚠️ open suspicions and the current frontier — that's where to
   pick up.
2. **Run the curriculum.** Read grounded + work_list. Form an
   **exploratory expectation** from the annotation + what has grounded —
   something to compare the trace against and explore, not a pass/fail
   spec. Write it down so you can compare later. A near-match (right nodes,
   different signature) is a finding that K is on the right track, not a
   failure. Then **report** what you found: did K meet the expectation, how
   was it out, what went wrong — feeding the next step. See trace-reading
   §Worked example 2.
3. **Diagnose: engine or curriculum?** (See below. Engine first.)
4. **Edit** the suspect (engine code, or — only when confident — the
   `.ks`). Smallest honest change. Comments minimal: describe what, not
   why the design is right.
5. **Update behaviour-notes in the same change** — a new Rule, a moved
   Active-state item, or a ⚠️ promoted to active investigation. Keep it lean.
6. **Verify.** Re-run; compare grounded + work_list before/after.
7. **Commit on the `dialogue` branch.** One change per commit; name what
   shifted.

## Diagnose: engine or curriculum?

**Engine first — always.** This is the target work. When a trace stalls,
a kline vanishes, or grounded/work_list diverge from the trainer
expectation, suspect the engine and dig there until the engine path is
genuinely exhausted.

**Retreat to curriculum-authoring only when** you are confident a `.ks`
change can test a theory or bring out a different result. Curriculum
writing advances the use case for the engine — it is not a workaround for
engine bugs. Typical retreats:

- **Prime-before-test ordering.** A question fed before the prime's
  identities land can't be answered (the parts aren't known yet). Reorder
  the `.ks` so priming precedes questioning, to test whether the engine
  _can_ answer given correct ordering. (mhall's WDMH is the canonical
  example — the question arrives at step 9, the identities at steps 13+.)
- **Shake things up.** Once the engine settles on a curriculum, author a
  new `.ks` to find the next edge — introduce ambiguity, withhold an
  identity the harness could offer, add a second question, invert
  ordering deliberately.
- **Test a theory.** A minimal `.ks` constructed to exercise one engine
  path (a single countersign; a lone unseen canon; a denotes with no
  reciprocal) is the fastest way to confirm or refute a hypothesis about
  engine behaviour.

When you do author a `.ks`, it lives in `data/scripts/`. Annotation
prose (parenthetical lines) is the trainer rationale — the harness
carries it onto each kline's `KDbg.annotation` and prints it as a section
header.

## Discipline (do not violate)

- **The harness never judges.** No verdict, no band-matching, no
  success/failure marker. Judgement is the trainer's (you, outside the
  loop). If you are adding evaluation logic to the harness, stop.
- **Engine first.** Suspect the engine before the curriculum; edit `.ks`
  only to test a theory or shake things up, never to work around an
  engine bug.
- **Check the source before asserting behaviour as fact.** Inferences
  stated as established rules cause real bugs (on record: the
  `_is_groundable` lift; the invented "fast route is for terminals" rule).
- **The engine speaks in semantic predicates** (`is_identity`,
  `is_unknown`, `is_canon`, `is_relationship`), never raw `kline.nodes`.
- **Update CONTEXT.md** when a glossary term shifts. Source and glossary
  move together.
- **Keep behaviour-notes lean** One line per rule. No more. Keep active
  state focused. Don't record decisions made, history changed etc - its
  not that kind of document.

## Where things live

| Artefact        | Path                          | Role                                                     |
| --------------- | ----------------------------- | -------------------------------------------------------- |
| Engine          | `src/dialogue/engine.py`      | The fork under tune                                      |
| Reference       | `src/dialogue/rationalise.py` | Verbatim baseline; engine.py diverges from here          |
| Harness         | `src/dialogue/harness.py`     | The non-judging compile→feed→present loop + CLI          |
| Compiler        | `src/ks/`                     | KScript → KValue; carries annotation/scope/labels        |
| Curricula       | `data/scripts/*.ks`           | `mhall.ks` is canonical                                  |
| Glossary        | `CONTEXT.md`                  | Domain terms                                             |
| Behaviour notes | `docs/behaviour-notes.md`     | Rules + Active state of K + Process (read every session) |

## Stop / ask

- **Stop** when grounded + work_list match the trainer expectation and
  the suite passes — then update behaviour-notes (move an Active-state
  item to Rules, or delete it) and CONTEXT if a term shifted.
- **Ask the user** only on a genuine engine-semantics fork (e.g.
  "should a countersign require compositional operands, or just two
  grounded values?"). Do not ask permission to treat engine behaviour as
  suspect — that is the core activity.
