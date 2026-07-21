# Dialogue-Driven Training — Specification

> **Working sketch, not a frozen contract.** This spec describes the current
> shape of the dialogue sub-project as it stands today. It is intentionally
> light: replace it wholesale as fresh discoveries reshape the design, rather
> than augmenting it. Behavioural granularity belongs in the code (and in tests
> added when a behaviour is _newly discovered_, not to defend yesterday's).

## Overview

A lesson is an authored **dialogue** between a Trainer (T) and a Trainee (K),
turn by turn, captured as a deterministic **dialogue script**. A **runner**
decodes the script, then drives two **actors** over the harness `MessageBus`,
tracking how much of the authored exchange the actors actually traverse.

Both sides emit on their own schedule and count; the runner is a
**coverage-tracking subscriber** that watches for a terminal condition and
reports the **displacement** — coverage rows never emitted. The default actors
are script-reading scaffolding; either side is meant to be replaced by a real
trainer or trainee, and the runner is expected to evolve when real actors
arrive.

## Purpose of dialogue work

The script is **not** a golden master. It is one of three coupled artefacts —
the **script**, the **code**, and the **rules** — that dialogue work exists to
bring into agreement with the author. The running purpose of every dialogue
change is to move all three toward that agreement; a script turn that the code
or the rules cannot honestly produce is a defect in one of the three, not a
target to reverse-engineer. Treating the script as an oracle the code must
blindly reproduce has, in the past, encoded mistakes (an author's slip becomes
the shape of the code). When a turn does not fit, locate the disagreement,
decide which artefact is wrong, and update them together.

## Dependencies

- `@CONTEXT.md` — Structural State, Canon, KValue, Role, Significance.
- `@specs/kscript.md`, `@specs/kline.md`, `@specs/kvalue.md`, `@specs/agent.md`.
- `@specs/harness-server.md` — the `MessageBus` the run is driven over.

## The Dialogue Script

A JSON object:

```
DialogueScript:
  source:  str          # KScript source, or a path to a .ks file
  priors:  list[str]    # optional: other script files whose turns run first
  run:     RunConfig    # optional: run modifiers (e.g. on_divergence)
  turns:   list[Turn]   # the ordered T/K exchange
  events:  list[Turn]   # optional: expected K groundings (white-box assertions)
```

```
Turn:
  actor:        "T" | "K"
  op:           str   # COUNTERSIGNS | CANONIZES | CONNOTES | DENOTES | IDENTITY
  signature:    str   # symbolic label, resolved by the decoder
  nodes:        list[str]
  significance: "S1" | "S2" | "S3" | "S4"
  notes:        str   # human commentary; ignored by the decoder
  close:        bool  # optional: marks this turn as the run's close
```

`source` is the single source of truth for kline structure. `turns` is the
exchange. An **annotation-only turn** (notes, no `op`) is human commentary and
is dropped at decode. A `close: true` turn marks the run's terminal content
(role-agnostic; may sit on T or K). When no turn is `close`, the last row is.
`events` (same row shape as `turns`, no `close` semantics) holds expected K
S1 groundings the runner verifies white-box (see §Grounding verification) —
targeted assertions, not an exhaustive manifest.

## Decode

```
decode(script) -> list[DecodedTurn]
```

A single-stage, configuration-time function. Per turn it resolves the kline
from `source`, attaches significance by band lookup, and passes through
`actor`/`op`. The decoder is a **resolver**: it builds the kline the turn
declares (declared signature verbatim, nodes resolved to canonical signatures)
and does not check that signature and nodes are consistent — an author may
declare a deliberate misfit. Annotation-only turns are dropped; every label
must resolve to a compiled entry (else a decode error).

**Compound catch-up.** A CANONIZES turn whose signature names a compound-word
(a label with a compiled compound identity) is decoded with `COMPOUND_TOKEN`
(@nlp_tokenizer spec) prepended to its nodes. The author writes the subwords
(`Mary => M ary`); the decoder adds the system marker so the kline matches
the compound identity the compiler produces. Without this, the declared
subwords would form a misfit against the compound's CT-encoded signature.

```
DecodedTurn:
  actor:  "T" | "K"
  op:     str
  value:  KValue    # KLine(signature, nodes) + significance
```

## The Runner

```
run(decoded, trainer_factory, trainee_factory, *, expected_groundings=(), on_divergence) -> Runner
```

The run is driven over the harness `MessageBus`:

- The **bus is the sink and the relay.** Actors publish a **burst** of events
  to an `EventSink` injected at construction; the runner's bus-wired sink
  bridges each burst to one `Message` addressed to the other role.
- The **runner is a coverage-tracking wildcard subscriber.** It updates its
  covered set on each emission and calls `bus.stop()` on a terminal condition.
- A thin **driver** seeds the trainer (an `accept` with an empty burst) and
  runs `bus.run()` on a dedicated thread.

An actor holds an `EventSink` and publishes a burst via `on_burst`. `accept`
is fire-and-forget: it receives the incoming burst (empty = "you open") and
publishes one-or-many replies. Every `accept` publishes **at least one**
proposal (`burst >= 1`): an actor with nothing substantive publishes a **PASS**
— a sentinel `{PASS: []}` at S1. The runner intercepts a PASS before matching
(neither coverage, nor divergence, nor a close); two consecutive PASSes from
the two roles is a terminal condition (mutual PASS).

The dialogue (what actors say) and K's grounding (what K knows) are verified
on **separate channels**: the dialogue is **black-box** (coverage by emission);
K's S1 groundings are **white-box** (a rationalising trainee exposes them via
`drain_observations`, and the runner checks them against `expected_groundings`).
Grounding is K's internal bookkeeping and does not emit into the dialogue.

### Actor contract

```
EventSink:
  on_burst(events: list[RationaliseEvent]) -> None

Actor:
  role: str
  accept(incoming: list[RationaliseEvent]) -> None   # publishes a burst via its sink

RationalisingTrainee (adds):
  drain_observations() -> list[KValue]   # K's S1 grounding events since last call
```

The runner builds the bus-wired sink and constructs actors via **factories**
`(sink) -> Actor`, so any actor is drop-in. The two defaults
(`ScriptTrainer`, `ScriptTrainee`) read the decoded script and are content-blind —
they advance their own cursor in script order and never realign to incoming
content. `ScriptTrainee` exposes no `drain_observations`; grounding assertions
apply only to a trainee that does (a rationalising trainee).

### Matching & termination

Each emission is matched against a **coverage budget** (a content key's
multiplicity in the coverage rows) and the close, by `(role, kline,
significance)` equality. Coverage is consumed first; the close terminates only
once its own coverage copies are spent (so a close that recurs as coverage
closes on its final occurrence, not its first):

- **In the budget with copies remaining** → consume one copy. Every budget
  spent → terminate (coverage exhaustion).
- **Equals the close (budget exhausted for its key)** → terminate. A unique
  close has no coverage copies, so terminates on first emission.
- **In the script but budget exhausted, and not the close** → immediate
  divergence (reason `"exhausted"`).
- **Present nowhere** → immediate divergence (reason `"unmatched"`).

Either divergence stops the run at once, regardless of policy;
`on_divergence` governs report-only (`"fail"` raises `Divergence`; `"accept"`
appends to `RunResult.unmatched`). The close may be emitted by any agent at any
time; the script is **de-positional** (the first row carries no opening
semantics, and anticipation/interjection are permitted and unflagged).

### Grounding verification (white-box)

After each K turn the runner drains the trainee's observations (if it exposes
`drain_observations`) and, at run end, checks every asserted grounding in
`expected_groundings` was observed at least once — a **subset** check (model B):
asserted groundings K never performs are a `GroundingDivergence` (reason
`"missing"`); extra K groundings not asserted are not policed. Grounding key
is `(signature, nodes, significance)` (role is always K). Grounding assertions
are ignored for a non-observable trainee (the table actors).

### Types

```
Divergence(Exception):
    role, emitted, unconsumed, reason, last_coverage_event

GroundingDivergence(Exception):
    grounded, unconsumed, reason ("missing"), last_coverage_event

RunResult:
    events:                list[RationaliseEvent]   # arrival-ordered
    unmatched:             list[RationaliseEvent]   # divergences (accept-mode)
    uncovered:             list[DecodedTurn]        # DISPLACEMENT: rows never emitted
    last_coverage_event:   RationaliseEvent | None
    unmatched_groundings:  list[KValue]             # grounding divergences (accept-mode)
    uncovered_groundings:  list[DecodedTurn]        # asserted groundings never observed
```

The signal that matters is the **displacement** (`uncovered`): coverage rows
never emitted. A script is orchestrated to cover the whole exchange; zero
displacement means the actors traversed all of it.

## Test Matrix

Tests live in `tests/test_dialogue_smoke.py` and cover **basic operation**
only. Add criteria here as fresh behaviours are discovered — do not enumerate
today's implementation choices as contract.

| ID    | Criterion                                                                                                                                  |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| DDT-1 | `decode(script)` returns a flat ordered `list[DecodedTurn]`, one per structural turn, significance attached by band lookup.                |
| DDT-2 | A malformed script (missing `source`/`turns`) is a decode error.                                                                           |
| DDT-3 | The canonical MHALL dialogue runs end-to-end through the runner with the default actors and covers the whole exchange (zero displacement). |

## Out of Scope

- Measuring, detecting, or signalling learning or grounding.

## Canonical Example

The reference dialogue is "Mary had a little lamb" (`scripts/dialogue-mhall.json`,
frozen for tests in `tests/_fixtures`). It is a **reference**, not a golden
master: its turns are edited in step with the code and the rules (see
§Purpose of dialogue work). A single depth-first cascade: the trainer opens
with the primary at S2; the trainee requests each unknown operand at S4; the
trainer supplies it; the trainee proposes role bindings at S3; the trainer
ratifies each at S1; the trainee closes with the primary's S1 countersign.
Where one side's operands outnumber the other's, the residual is synthesised
into a left-operand signature and connoted at S3 like any other pairing.
