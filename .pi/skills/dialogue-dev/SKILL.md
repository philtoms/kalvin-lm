---
name: dialogue-dev
description: Investigates and progresses the dialogue sub-project src/training/dialogue/ — the authored-script ↔ real-actor ↔ rules triad and the coverage/displacement loop that brings them into agreement. Use when the user says "/dialogue-dev" or asks to work on, debug, advance, tune, or understand the dialogue training system, the rationalising trainee, the synthesizing trainer, the runner, or dialogue scripts.
---

# Dialogue Dev

> The dialogue sub-project is where a Trainer (T) and a Trainee (K) are brought
> into agreement with an author through a **script ↔ code ↔ rules** triad. This
> skill is the concentrated guide for investigating it and making progress in
> it. Read it when you start dialogue work; refer back to the sections you need.

## The one principle that governs every change

**The script is not a golden master.** It is one of three coupled artefacts —
the **script** (`scripts/dialogue-*.json`), the **code** (`src/training/dialogue/`),
and the **rules** (`specs/dialogue-driven-training.md`, `specs/dialogue-cogitation.md`)
— and dialogue work exists to move all three toward agreement with the author.

A script turn that the code or the rules cannot honestly produce is a defect in
one of the three, not a target to reverse-engineer. Treating the script as an
oracle the code must blindly reproduce has, in the past, encoded the author's
slips into the shape of the code.

**When a turn does not fit, locate the disagreement, decide which artefact is
wrong, and update them together.** This is the loop every dialogue change runs
through.

## Mental model (read first)

A **lesson** is an authored dialogue between T and K, turn by turn, captured as
a deterministic **dialogue script** (JSON). A **runner** decodes the script,
then drives two **actors** over the harness `MessageBus`, tracking how much of
the authored exchange the actors actually traverse.

- **Both sides emit on their own schedule and count.** The runner is a
  **coverage-tracking subscriber**, not a turn-taker.
- The signal that matters is the **displacement** — coverage rows never
  emitted. Zero displacement means the actors traversed all of the authored
  exchange.
- The **default actors** (`ScriptTrainer`, `ScriptTrainee`) are script-reading
  scaffolding — content-blind, cursor-advancing. They cannot diverge from the
  script. They exist so the runner's coverage machinery has something to drive
  before the real actors arrive.
- The **real actors** are drop-in: `SynthesizingTrainer` (derives each turn
  from the compiled source) and `RationalisingTrainee` (a stateful, rationalising
  trainee). Either or both can be substituted via flags on the driver.

### Two verification channels

| Channel       | What                         | How       | Where policed                        |
| ------------- | ---------------------------- | --------- | ------------------------------------ |
| **Dialogue**  | What actors say (emissions)  | Black-box | Coverage budget + close, by emission |
| **Grounding** | What K knows (S1 groundings) | White-box | `expected_groundings` subset check   |

Grounding is K's internal bookkeeping; it does **not** emit into the dialogue.
A rationalising trainee exposes its S1 groundings via `drain_observations`;
the runner drains them after each K turn and checks every asserted grounding
was observed at least once (a subset check — model B). Extra K groundings are
not policed.

## Where everything lives

| Artefact             | Path                                                                                                                     | Role                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| Package root         | `src/training/dialogue/__init__.py`                                                                                      | Public re-exports                                                                  |
| Decoder              | `src/training/dialogue/decoder.py`                                                                                       | Script → flat `list[DecodedTurn]` (config-time, a resolver)                        |
| Actors               | `src/training/dialogue/actors.py`                                                                                        | `ScriptTrainer`/`ScriptTrainee`, `SynthesizingTrainer`, `RationalisingTrainee`     |
| Runner               | `src/training/dialogue/runner.py`                                                                                        | Bus subscriber + driver, coverage/divergence/PASS, `RunResult`                     |
| Rationaliser         | `src/training/dialogue/rationalise.py`                                                                                   | Pure engine: `(state, incoming) -> (batch, observations)`                          |
| Synthesize           | `src/training/dialogue/synthesize.py`                                                                                    | `SynthesizingTrainer`'s R1/R2/R3 derivation                                        |
| Driver (end-to-end)  | `dev/dialogue/dialogue_run.py`                                                                                           | CLI; renders trace, divergence, grounding                                          |
| Probe (turn-by-turn) | `dev/dialogue/probe_rationalise.py`                                                                                      | Drives the pure `Rationaliser` directly; batch + observations + work-list per turn |
| Canonical script     | `scripts/dialogue-mhall.json`                                                                                            | "Mary had a little lamb" — the reference dialogue                                  |
| Second script        | `scripts/dialogue-wdmh.json`                                                                                             | Second example                                                                     |
| Frozen fixture       | `tests/_fixtures/` (`mhall_script`)                                                                                      | The script frozen for the smoke test                                               |
| Smoke test           | `tests/test_dialogue_smoke.py`                                                                                           | Basic-operation acceptance tests (DDT-1..3)                                        |
| Contract (runner)    | `specs/dialogue-driven-training.md`                                                                                      | Authoritative current contract — **a working sketch**                              |
| Contract (cog)       | `specs/dialogue-cogitation.md`                                                                                           | Cogitation rules — **the most speculative part**                                   |
| Build records        | `plans/implement-dialogue-driven-training.md`, `implement-rationalising-trainee.md`, `implement-synthesizing-trainer.md` | What was built + rationale (not active contract)                                   |

The contracts are explicitly **working sketches, not frozen**. Replace them
wholesale as discovery reshapes the design; do not augment them with
implementation choices. Behavioural granularity belongs in code, and tests are
added when a behaviour is _newly discovered_, not to defend yesterday's.

## The investigation workflow

Start here on any dialogue task. Do not skip to code edits.

### 1. Reproduce — run the canonical script first

The first action is always to run the dialogue and read its output, not to
read code. Two diagnostic surfaces, chosen by granularity:

**End-to-end (`dev/dialogue/dialogue_run.py`)** — the whole run through the bus,
runner, and actors. Reach for this first; it's the single richest surface.

```bash
PYTHONPATH=src .venv/bin/python dev/dialogue/dialogue_run.py                          # default (MHALL), both table actors
PYTHONPATH=src .venv/bin/python dev/dialogue/dialogue_run.py scripts/dialogue-wdmh.json
PYTHONPATH=src .venv/bin/python dev/dialogue/dialogue_run.py --rationalise            # real trainee, table trainer (deterministic oracle)
PYTHONPATH=src .venv/bin/python dev/dialogue/dialogue_run.py --synthesize             # real trainer, table trainee
PYTHONPATH=src .venv/bin/python dev/dialogue/dialogue_run.py --synthesize --rationalise  # both real actors
PYTHONPATH=src .venv/bin/python dev/dialogue/dialogue_run.py --rationalise -v         # also list K's grounded klines
PYTHONPATH=src .venv/bin/python dev/dialogue/dialogue_run.py --divergence             # fail (exit 1) on the first immediate divergence
```

Flags are **orthogonal**. By default divergences are **accepted**: the run
completes (exit 0) and any unmatched emissions/groundings are listed in the
trace. Pass `--divergence` to fail (exit 1) on the first immediate divergence
instead — useful when you want the run to stop and surface a divergence report
the moment an actor strays. The canonical workflow below uses the default (accept):

1. **Both table actors** — should always pass with zero displacement. If it
   doesn't, the script or decoder is broken before any actor work matters.
2. **`--rationalise` (real trainee vs. table trainer)** — the rationalising
   trainee against a deterministic oracle. This is where trainee work shows up.
3. **`--synthesize --rationalise` (both real)** — the full real-actor run.

Exit code is `1` only on an immediate divergence **under `--divergence`**;
`0` otherwise. With the default (accept), a clean exit does **not** mean zero
displacement — read the counts and the unmatched-emissions/groundings sections.

**Turn-by-turn (`dev/dialogue/probe_rationalise.py`)** — drives the pure `Rationaliser`
engine directly, with no actor, no bus, no runner. It hand-feeds T queries and
prints K's batch, observations, work-list depth (with each entry), and —
crucially — both the **declared** and **structural** significance of each
incoming query. Reach for this when an end-to-end run misbehaves and you need
to see what the engine is doing on a specific turn in isolation.

```bash
PYTHONPATH=src .venv/bin/python dev/dialogue/probe_rationalise.py
```

It is meant to be **edited in place** for the investigation at hand: change
the compiled `source`, swap the hand-fed queries (`_find(entries, op=..., label=...)`),
or add steps to walk further into the dialogue. Keep it a probe — a scratch
harness you reshape per question, not a growing test suite. The smoke test
(`tests/test_dialogue_smoke.py`) is where confirmed behaviour gets pinned. Don't
run this **until** new behaviour has been confirmed.

### 2. Read the output

The driver prints, in order:

- **Exchange (arrival order)** — every emission rendered in scripted form
  (`role op signature:[nodes] band`), with the verbatim script JSON row shown
  beneath each matched emission. A PASS renders as `<role> PASS`.
- **`--- displacement (uncovered) ---`** — authored coverage rows never
  emitted. **This is the primary signal.** Each shows the row that was expected
  and never produced.
- **Counts**: `events received`, `uncovered (displacement)`,
  `uncovered groundings`.
- **With `-v` and `--rationalise`**: `K grounded klines` — K's post-run model,
  grouped by owning signature, rendered in scripted form.

On an immediate divergence (`--divergence`, i.e. `on_divergence="fail"`), stderr gets a
**divergence report** instead: the emitted kline that diverged, the verdict
(`exhausted` — every authored copy already consumed; or `unmatched` — matches
no row), the last healthy coverage match, and the still-uncovered same-role
rows. A `GroundingDivergence` report mirrors this for the grounding channel
(`missing` — an asserted grounding never observed). When divergences are
accepted (the default), the run completes and the same information surfaces as
**unmatched emissions / unmatched groundings** sections in the trace instead.

### 3. Diagnose — pick the channel, pick the artefact

Map the symptom to the channel and the artefact:

| Symptom                                    | Channel   | Likely artefact to inspect first                                                                |
| ------------------------------------------ | --------- | ----------------------------------------------------------------------------------------------- |
| Non-zero displacement (rows never emitted) | Dialogue  | Real actor's emission logic; or the script asserts something the rules don't produce            |
| `Divergence: unmatched`                    | Dialogue  | Actor emitted content not in the script; or script is missing a row the actor honestly produces |
| `Divergence: exhausted`                    | Dialogue  | Actor re-emitted a content beyond its authored multiplicity; or the script under-counts it      |
| `GroundingDivergence: missing`             | Grounding | Rationaliser never grounds an asserted kline; or the script over-asserts                        |
| Table actors themselves diverge            | —         | Script or decoder is malformed — fix before anything else                                       |
| `DecodeError`                              | —         | Script JSON, label resolution, or op vocabulary                                                 |

Then read the relevant code with the symptom in hand. The module map above
tells you where to look. Key invariants to keep in mind while reading:

- **The decoder is a resolver, not a gatekeeper.** It builds the kline the turn
  _declares_ (declared signature verbatim, nodes resolved to canonical
  signatures) and does **not** check that signature and nodes are consistent —
  an author may declare a deliberate misfit. Compound-word signatures get
  `COMPOUND_TOKEN` prepended at decode (compound catch-up).
- **The runner is de-positional.** The first row carries no opening semantics;
  anticipation and interjection are permitted and unflagged. Coverage is
  order-agnostic on middle entries (the natural fit for real actors).
- **Coverage is consumed first; the close terminates only once its own coverage
  copies are spent.** A close that recurs as coverage closes on its final
  occurrence, not its first.
- **Every `accept` publishes at least one proposal (`burst >= 1`).** An actor
  with nothing substantive publishes a PASS (`{PASS: []}` at S1). Two
  consecutive PASSes (one from each role) is terminal (mutual PASS).
- **Emission deduplication lives in the actor, not the engine.** The
  `Rationaliser` is stateless about its own emissions and may re-derive a
  proposal on successive turns; the `RationalisingTrainee` actor drops any
  proposal it has already published, so K never repeats itself.

### 4. Decide — which artefact is wrong?

Apply the one principle. For each disagreement, ask:

- **Is the script asserting something the rules honestly produce?** If not, the
  script is wrong (an author's slip) — fix the script row.
- **Is the code failing to produce what the rules honestly describe?** If so,
  the code is wrong — fix the actor / engine / runner.
- **Are the rules themselves underspecified or wrong?** If the disagreement
  exposes a gap or an error in the contract, fix the spec (replace, don't
  augment).

Often more than one moves together. The goal is agreement, not blame.

### 5. Edit — smallest honest change, then re-run

- Make the smallest change that moves the artefacts toward agreement.
- **Snapshot the run output before and after** so you have a displacement
  comparison.
- Re-run the driver (the relevant flag combination) and confirm the
  displacement shrinks (or the divergence clears) without introducing new ones.
- Run the smoke test: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_dialogue_smoke.py -q`.
  The canonical MHALL run with table actors must stay at zero displacement
  (DDT-3) — that is the core-loop wiring guard.

### 6. Document — keep the triad in sync

Per the project's coding-activity rule, after the code change ask the user
before updating docs. When updating, edit the **single owning layer** only:

- **Behavioural rules / contract** → `specs/dialogue-driven-training.md`
  (runner, actors, decode) or `specs/dialogue-cogitation.md` (cogitation).
  Replace wholesale as discovery reshapes; do not augment with implementation
  choices.
- **Implementation rationale** → the relevant `plans/implement-*.md` (build
  record, not active contract).
- **New behaviour discovered** → add a test criterion to the spec's test matrix
  (DDT-N style) and a test in `tests/test_dialogue_smoke.py`. Do not pin
  yesterday's implementation choices.

Do not duplicate content across layers; cross-reference instead. If the docs
already accurately describe the new state, make no change.

## Working on each component

### The script (`scripts/dialogue-*.json`)

Shape: `{source, priors?, run?, turns[], events?}`. Each turn is
`{role, op, signature, nodes, significance, notes?, close?}`.

- `source` is the single source of truth for kline structure — a KScript source
  string or a path to a `.ks` file.
- `op` ∈ `{COUNTERSIGNS, CANONIZES, CONNOTES, DENOTES, IDENTITY}`. Anything
  else is a decode error.
- `significance` ∈ `{S1, S2, S3, S4}` (band lookup).
- An **annotation-only turn** (`notes`, no `op`) is human commentary — dropped
  at decode. Use them to narrate the dialogue's intent for human readers.
- `close: true` marks the run's terminal content (role-agnostic). At most one
  survives in a composed script (priors collapse earlier closes to ordinary
  coverage rows). When no turn is `close`, the last row is.
- `events` (optional) holds expected K S1 groundings — **targeted assertions,
  not an exhaustive manifest**. The runner verifies each was observed at least
  once (subset check). Add an event only when you want to pin a specific K
  grounding.
- `priors` (optional) is a list of other script-file paths whose turns run
  before this script's own, in order.

### The decoder (`decoder.py`)

`decode(script)` runs once at configuration time and returns a flat ordered
`list[DecodedTurn]`. It resolves every symbolic label against `source`,
attaches significance by band lookup, drops annotation-only turns, and does
compound catch-up. It does not gatekeep structure.

When editing: preserve the resolver-not-gatekeeper property. A new op or a new
resolution rule belongs here only if it is a genuine resolution concern
(symbol → kline), not a validity check.

### The actors (`actors.py`)

The `Actor` base publishes a burst via its injected `EventSink`; `accept`
collects `next_events` and publishes them as one burst, emitting a PASS when
nothing is produced. Two defaults read the table; two reals derive turns.

- **`ScriptTrainer` / `ScriptTrainee`** — `_TableActor` subclasses, content-blind,
  cursor-advancing. Cannot diverge. Exist to exercise the runner.
- **`SynthesizingTrainer`** — derives each turn from the compiled source via
  `synthesize` (R1 opening at S2, R2 reply to an identity by precedence:
  canon → CONNOTES → ratify-identity; R3 echo an exact compiled match), and
  falls back to the decoded table for its driving moves when K PASSes (a
  close, the next script's opening).
- **`RationalisingTrainee`** — wraps the pure `Rationaliser` engine, owns the
  `RationaliserState`, deduplicates its own emissions, and exposes S1
  groundings via `drain_observations`.

When substituting a real actor: it must be drop-in via the factory
`(sink) -> Actor`. Real-actor state lives on the actor; the runner holds only
coverage bookkeeping.

### The runner (`runner.py`)

A wildcard subscriber over `MessageBus` plus a thin driver. Owns the bus,
builds the bus-wired sinks, constructs the actors, seeds the trainer, runs
`bus.run()` on a dedicated thread until a terminal condition, and reports
`RunResult`. Coverage is a per-key `Counter` (multiplicity = authored copies);
the close is a single content key. PASS is intercepted before content matching.

When editing: the runner holds **coverage bookkeeping only** — whose-turn,
cursors, and pacing live in the actors; the relay lives in the bus. Don't push
actor concerns into the runner.

### The rationaliser (`rationalise.py`)

The pure engine. Each `rationalise(state, incoming)` call routes every query
(S4 pops the identity ask; everything else appends to the work-list, with an
S2 misfit additionally unpacking its unrecognised nodes/signature as identity
placeholders), then cogitates LIFO: promotable/groundable entries are grounded
at S1 (observed, not emitted) and dropped; a countersignable entry takes the
S3 path; a multi-node misfit takes the S2 similar-fit path. Returns
`(batch, observations)`.

When editing: keep the engine **stateless about its own emissions** —
deduplication is the actor's job, so the engine's state stays a pure model of
what K has grounded. This is the single most important invariant here.

## Common pitfalls

- **Treating the script as an oracle.** Don't reverse-engineer the code to
  reproduce a script row the rules don't honestly produce. See §The one
  principle.
- **Reading code before running a probe.** The driver's trace/displacement
  report and `dev/dialogue/probe_rationalise.py`'s turn-by-turn view are the fastest
  paths to a diagnosis. Always reproduce first; drop into code only with a
  symptom in hand.
- **Pinning implementation choices as tests.** Tests are added when a behaviour
  is _newly discovered_, not to defend the current mechanism. The smoke test
  intentionally covers basic operation only.
- **Augmenting the spec instead of replacing it.** The contracts are working
  sketches. Replace wholesale as discovery reshapes the design.
- **Over-engineering or over-documenting.** This is a simplified kalvin system -
  Lets keep it that way.
- **Confusing the two channels.** Dialogue divergence and grounding divergence
  are separate. Read the right report.
- **Forgetting `burst >= 1`.** An actor that returns nothing from
  `next_events` still owes a proposal — the base emits a PASS. A run that
  stalls on mutual PASS is a real-actor symptom, not a runner bug.
- **Expecting positional semantics.** The script is de-positional. The first
  row is not special; anticipation and interjection are allowed.

## When to escalate to the user

- A disagreement cannot be resolved without a design decision (which artefact
  is wrong is genuinely ambiguous).
- A change would alter the public contract in `specs/dialogue-driven-training.md`
  or `specs/dialogue-cogitation.md`.
- The smoke test (DDT-3, canonical MHALL with table actors) regresses — that is
  the core-loop guard and its breakage is a signal, not something to paper over.
- Three runs pass with no meaningful change in displacement — surface the stall
  rather than looping.
