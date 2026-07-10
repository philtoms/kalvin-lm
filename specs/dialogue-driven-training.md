# Dialogue-Driven Training — Specification

## Overview

A lesson is an authored **dialogue** between a Trainer (T) and a Trainee (K),
turn by turn. The dialogue is a deterministic, table-driven artifact. A
**dialogue runner** loads and decodes the table, then drives the two actors.

Both sides emit on their own schedule, in any order and any count; the runner is
a **sink** that receives emissions, tracks them against a coverage set, and
watches for a terminal condition. Order is not enforced; anticipation (an actor
emitting ahead of the table's causal order) is permitted and unflagged.

The runner loads and feeds the actors, closes
mechanically (the close content is emitted, the coverage set is exhausted, or
both actors pass in turn), and tracks coverage and immediate divergence. The
**displacement** — coverage rows never emitted — is the signal of how much of
the authored exchange the actors traversed.

Both sides of the loop are coded so that either can be replaced by a real trainer
or a real trainee. The default actors supplied here are table-reading
scaffolding. The runner does not defend
against replacement — it will evolve when real actors arrive.

The runner is **bus-driven**: it drives the exchange over the harness
`MessageBus`. The actors are sink-driven — each holds an `EventSink` and
publishes its turns via `accept` (fire-and-forget, one-or-many per incoming —
see §Actor contract and the PASS proposal).

## Dependencies

- `@CONTEXT.md` — Structural State, Canon, KValue, Role, Significance.
- `@specs/kscript.md` — compiled-entry `op` field (the structural states).
- `@specs/kline.md` — `is_canon`, KLine equality.
- `@specs/kvalue.md` — KValue (KLine + significance).
- `@specs/agent.md` — `RationaliseEvent` (the event type actors publish).
- `@specs/harness-server.md` — the `MessageBus` the run is driven over.

## Definitions

### Dialogue Table

The source artifact for a lesson. A JSON object:

```
DialogueTable:
  script:  str          # KScript source, or a path to a .ks file — the
                         # authority for kline structure
  priors:  list[str]    # optional: paths to other DialogueTable files whose
                         # turns run before this table's own, in list order.
                         # Resolved recursively; each prior's turns are
                         # inserted (in list order) ahead of this table's.
                         # A prior resolves its own `script`; only its turns
                         # are carried in.
  run:     RunConfig    # optional: run modifiers. The runner is format-agnostic
                         # and consumes DecodedTurns, not the raw table; the
                         # loader resolves the section into run inputs.
  turns:   list[Turn]   # ordered, the exact T/K exchange
```

```
RunConfig:
  on_divergence: "fail" | "accept"   # default "fail"
```

**All run modifiers live in the `run` section** (so future run knobs extend it
without touching the rest of the table). Unknown keys inside `run` are a decode
error. The single run modifier today is `on_divergence`.

```
Turn:
  actor:        "T" | "K"
  op:           str   # a Structural State: COUNTERSIGNED | CANONIZED |
                      #                CONNOTED | UNDERSIGNED | IDENTITY
  signature:    str   # symbolic label, resolved by the decoder
  nodes:        list[str]   # symbolic labels, resolved by the decoder
  significance: "S1" | "S2" | "S3" | "S4"
  notes:        str   # human commentary; ignored by the decoder
  close:        bool  # optional: `true` marks this turn as the close. The
                      # runner reads it as the run's terminal content (§Table
                      # Structure) — any agent emitting it at any time ends the
                      # run. When no turn is marked `close`, the last row is the
                      # close. Role-agnostic: it may sit on either a trainer (T)
                      # or trainee (K) row.
```

`script` is the single source of truth for kline structure (canonical
signatures, atom values, subword composition). `turns` is the exact exchange.
`op` is a Structural State (`@CONTEXT.md`).

`script` may be given **inline** (the KScript source text) or as a **path** to a
`.ks` file. The two are disambiguated by path-likeness, not existence: a value
is treated as a path when it contains a path separator (`/` or `\`) or ends in
`.ks`, and is then resolved against the file system; any other value is used
verbatim as inline source. A path-like value that is missing or unreadable is a
load error (no silent inline fallback, so a typo or wrong working directory is
not masked).

`priors` (optional) names other `DialogueTable` files whose turns precede this
table's own. Each prior is loaded and resolved against **its own** `script`;
only its `turns` are carried in, inserted in list order (priors[0] first), so a
multi-file lesson reads as one continuous exchange. Priors resolve recursively
(a prior may name its own priors); the merged `turns` are what `decode` and the
runner consume. A missing or malformed prior file is a load error.

An **annotation-only turn** carries `notes` but no `op`; it is human commentary
and is not part of the exchange.

The `turns` are an ordered list that documents **cause and effect** (a
ratification follows its proposal, an identity supply follows its request). That
order is authorial/human-facing; the runner does not enforce it (§Anticipation).

### Decoded Turn

A turn resolved to an exchangeable structure, produced once at configuration
time by the **decoder**:

```
DecodedTurn:
  actor:  "T" | "K"
  op:     str       # passed through from the Turn (a Structural State)
  value:  KValue    # KLine(signature, nodes) + significance, resolved to uint64
```

`actor`, `op`, and `significance` are three independent axes, all carried
alongside the KLine and never folded into it.

### Decoder

A single-stage, configuration-time function:

```
decode(table) -> list[DecodedTurn]
```

Per turn:

1. **Resolve the kline from `script`** (script is authority). The decoder is a
   **resolver**: it builds the kline the turn declares — the
   declared `signature` verbatim, the `nodes` resolved to their canonical
   signatures. An author may declare a signature that differs from the canon
   its nodes form (a deliberate misfit — see `@specs/dialogue-cogitation.md`).
   - **CANONIZED** — resolve each node label to its canonical signature
     (canon-preferred, atom fallback) and build `KLine(signature, nodes)` with
     the declared signature verbatim.
   - **IDENTITY** — resolve the atom by label.
   - **Constructed relation** (`COUNTERSIGNED` / `CONNOTED` / `UNDERSIGNED`) —
     resolve each node label to its canonical signature and rebuild the relation
     KLine.
2. **Attach significance** by band lookup (`"S1"→SIG_S1`, …) from the turn.
3. **Pass through** `actor` and `op`; **drop** annotation-only turns and ignore
   `notes` on the rest.

Every signature and node label must resolve to a compiled entry (a label not
found in the script is a decode error). The relationship between a signature
and its nodes is the author's to declare.

The decode path also validates the close uniqueness (§Table Structure): the
close content appears exactly once in the table. A close whose content also
appears as a coverage row is malformed at decode time (an emission of that
content would be ambiguous: coverage or close?).

## Table Structure

A decoded table is **de-positional**. It is a **coverage set** (every turn)
plus one **close** — the ``close:true`` turn if any, else the last row — which
any agent may emit at any time to end the run. The authored order documents
cause and effect; the runner does not enforce it.

### Invariants

- The close content is **unique**: its ``(role, kline, significance)`` key
  appears exactly once. A close that duplicates a coverage row would make an
  emission of that content ambiguous (coverage or close?) — malformed at decode
  time.

The runner seeds the trainer mechanically (an ``accept`` with ``message=None``);
the close is position-free. A close can happen anywhere, on any agent, at any time.

## The Runner

```
run(decoded, trainer_factory, trainee_factory, *, on_divergence) -> Runner
```

The run is driven by the harness **`MessageBus`**
(`src/training/harness/bus.py`). The bus is the **sink and the relay**; the
runner is a **coverage-tracking wildcard subscriber** plus a thin driver that
seeds the trainer and runs the bus until a terminal condition. The runner
depends on `training.harness` — it is a training application and belongs next to
the harness.

- **Sink = the bus (via a bus-wired `EventSink`).** Actors publish events to an
  `EventSink` injected at construction; the runner's bus-wired sink bridges
  each `on_event` to a `Message` addressed to the other role. The bus's `send`
  is thread-safe — actors publish from any thread (including a cogitation
  thread), which is the true non-blocking behaviour the run regime requires.
  No `asyncio`; no second concurrency model. The actor does not know about the
  bus; it publishes to its sink, and the sink routes.
- **Relay = the bus's role dispatch.** Each actor subscribes to its own role;
  the bus-wired sink addresses each published event to the **other** role; the
  bus delivers. The runner does not relay — the bus does.
- **Coverage = a wildcard subscriber.** The runner subscribes to `*` (every
  message), updates its covered set on each emission, and calls `bus.stop()` on
  a terminal condition. A PASS emission (§Actor contract) is intercepted
  before matching: neither coverage, nor divergence, nor a close.
- **Driver.** A thin `run()` seeds the trainer by addressing a `Message` to the
  trainer's role (`message=None`), then runs `bus.run()` on a dedicated thread
  until a terminal condition: the close content is emitted, the coverage set is
  exhausted, or both actors pass in turn. Every `accept` yields at least one
  proposal (`burst >= 1`), so the bus never blocks indefinitely. The
  displacement (uncovered coverage rows) is what matters.

The dialogue metaphor: messy and real. **No synchronised alternation** — an
actor may emit one-or-many replies per incoming message (at least one — a PASS
when it has nothing substantive), and the relay does not track turns.
**Anticipation** (emitting ahead of authored causal order) and
**interjection** (emitting unsolicited) are first-class and unflagged. No
participant polices order; agents must rationalise and cogitate to make sense
of the stream — the point of Kalvin.

### Actor contract

An actor is a dialogue participant. It holds an **`EventSink`** (injected at
construction) and publishes events to it via `on_event`. The runner builds a
bus-wired sink per actor (bridging `on_event` to a bus `Message` addressed to
the other role) and constructs each actor with its sink, so any actor is
drop-in.

```
EventSink:
  on_event(event: RationaliseEvent) -> None

Actor:
  role: str
  accept(event: RationaliseEvent | None) -> None   # publishes via its sink
```

`accept` is **fire-and-forget**: it receives an incoming event (or `None` for
the opening seed) and returns immediately; the actor decides when and how many
events to publish via its sink — possibly later (from a cogitation thread),
possibly many times (a priming burst, a scaffolding sequence). The actor does
not know about the bus; it publishes to its sink, and the sink routes.

`accept` always publishes **at least one** proposal (`burst >= 1`): the actors
in a dialogue drive each other's next events, so an actor never replies with
nothing. When an actor's cogitation yields nothing substantive (the
rationaliser has no workable move; a table actor has exhausted its rows), it
publishes a **PASS** — a sentinel proposal `{PASS: []}` at S1 (`PASS_SIGNATURE`,
a reserved signature; the meaning lives in the signature, S1 only sorts it
highest). The runner intercepts a PASS **before** content matching: it is
neither coverage, nor divergence, nor a close; it routes to the other role,
and **two consecutive PASSes from the two roles** (each side passing in turn)
is a terminal condition (mutual PASS — the actors have nothing more to say).
Two PASSes from the same role are not terminal — that side is waiting while the
other still has content. The `Actor` base class enforces `burst >= 1` by
emitting a PASS when an actor's `next_events` yields nothing, so the contract
holds for every actor.

The runner constructs actors via **factories** ``(sink) -> Actor``: only the
runner owns the bus, so only it can build the bus-wired sink, so it builds the
actors too. This makes any actor — including `SynthesizingTrainer` and
`RationalisingTrainee` — drop-in.

Both default actors (`TableTrainer`, `TableTrainee`) read the decoded table,
filter to their own `actor`, and yield those rows in order with their role on
each event. They emit by index; `incoming` only supplies the emitted event's
`query`.

### Permitted state

The runner holds **coverage bookkeeping** only (as a bus subscriber): the
table's **fixed set of distinct coverage contents**, a **covered subset** that
grows monotonically as emissions match, the close content key, a `closed` flag,
and the **role of the most recent PASS** (for mutual-PASS termination —
§Actor contract). That is its entire state: whose turn it is, per-actor
cursors, pacing, retry counts, and idle deadlines all live in the actors. The
relay lives in the bus; the runner observes and records.

## Matching

The runner is a **content matcher**. Each emission observed by the
wildcard subscriber is first checked against termination: once `closed` is set,
the run is over and **every subsequent emission is dropped** (not recorded, not
matched, not divergence). This matters because the bus dispatches role handlers
*before* wildcards: a role handler may react to a terminal emission and enqueue
another *before* the wildcard marks the run closed. Such a trailing emission is
noise, not divergence. Before termination, each emission is checked for PASS
(§Actor contract): a PASS is intercepted **before** matching — it is neither
coverage, nor divergence, nor a close. Any other emission is matched against the
table's **distinct coverage contents** and the close by content equality
`(role, kline, significance)`:

- **Equals the close** — terminate (the close may be emitted by any agent at
  any time).
- **Present in the distinct coverage contents** — mark that content **covered**.
  Duplicate table rows collapsed to this one distinct content at construction;
  coverage is **idempotent** — re-emitting already-covered content is *not*
  divergence. When the coverage set is exhausted, the run terminates (entry
  exhaustion).
- **Present nowhere in the table** (neither close nor any coverage content) —
  **immediate divergence**. Under `on_divergence="fail"` the runner raises
  `Divergence`. Under `on_divergence="accept"` the emission is recorded in
  `RunResult.unmatched` and the run continues.

A real trainer is expected to synthesise its training responses, and a real
trainee its responses; both announce their role and are matched against the
authored table. Under the table-reading `TableTrainer` / `TableTrainee` a
divergence cannot occur — they read the same table the runner matches against.
The match exists for the real actors: it is the mechanism that surfaces an
emission the authored exchange did not authorise.

### Anticipation and interjection

The table's order documents cause and effect but is not enforced within the
middle. An actor emitting content that, in the authored causal order, depends
on a not-yet-emitted cause (e.g. a trainer ratifying before the request
arrives) is a normal match against whatever same-role distinct content its
content equals. **Interjection** — emitting unsolicited, or emitting again
after the other side has gone quiet — is the same: a normal emission, routed
by the bus to the other role. Both anticipation and interjection are permitted
behaviour — the dialogue is messy and real, and agents must rationalise and
cogitate to make sense of it.

A table is de-positional (§Table Structure):
the close may be emitted by any agent at any time, and the first row carries no
opening semantics. Anticipation and interjection apply to the whole table; nothing is
non-anticipatable. Ordering is not enforced by the relay.

## Termination

The driver runs `bus.run()` until a terminal condition (the subscriber calls
`bus.stop()`):

- **Close emitted** — any agent emits the close content (the ``close:true``
  turn's content, or the last row's), anywhere in the stream.
- **Coverage exhausted** — every distinct coverage content has been covered.
- **Mutual PASS** — the two roles each publish a PASS in turn (a PASS from one
  role followed by a PASS from the other). Two PASSes from the same role are
  not terminal — that side is waiting while the other still has content.

Because
`accept` is fire-and-forget and replies are **one-or-many** (`burst >= 1`,
§Actor contract), an actor never goes silent — it always publishes at least one
proposal, emitting a PASS when it has nothing substantive — so there is **no
idle timeout**. The signal that matters is the **displacement** (§Types): the
coverage rows never emitted, reported in `RunResult.uncovered`. A script is
orchestrated to cover the whole exchange; the displacement measures how far
the realized dialogue fell short of that.

## Types

```
Divergence(Exception):
    role:           str                  # the role of the divergent emission
    emitted:        KValue               # the unmatched emission's proposal
    unconsumed:     tuple[DecodedTurn]   # unconsumed same-role rows at the moment of divergence

RunResult:
    events:    list[RationaliseEvent]    # every received emission, in ARRIVAL order
    unmatched: list[RationaliseEvent]    # immediate divergences: emissions matching nothing (accept-mode only)
    uncovered: list[DecodedTurn]         # DISPLACEMENT: distinct coverage rows never emitted
```

`events` is **arrival-ordered** — every received emission in the order
the bus delivered it. `unmatched` is populated only under
`on_divergence="accept"` (immediate divergences recorded as they occur).
`uncovered` is the **displacement** — the distinct coverage rows never emitted,
measuring how far the realized dialogue fell short of the authored
whole-exchange coverage. A script is orchestrated to cover the whole exchange;
zero displacement means the actors traversed all of it.

## What Training Is

Training is a deterministic mechanism that ensures the correct next response to
a trainee event is available. It measures learning only indirectly, through
experience: Kalvin learns by doing, and the displacement (§Types) records how
far a realized dialogue fell short of the authored exchange. Grounding means
Kalvin understands; it is an internal state, surfaced only as the
significance on each emission. The runner's concern is the exchange and its
coverage — the actors bring their own learning and memory.

## Out of Scope

- How a real trainer produces its turns, or a real trainee its responses. Both
  arrive with their own cogitation and (for the trainee) memory; the runner's
  `Actor` interface is the contract they satisfy.
- Supervisor escalation on a request the trainer cannot resolve — belongs to the
  real trainer.
- Reject-and-re-prompt (an actor revising a non-matching turn). Requires a
  rejection channel back to the actor, a contract change beyond `accept`.
  Deferred.
- Per-turn ordering constraints (gating a turn on the prior consumption of
  another). The whole-table middle floats; dependencies are enforced only
  indirectly, via content-match failure.
- An actor replying on a *different* bus than the run's own. The run owns its
  bus; actors reply to the sink they were handed.
- Multi-actor-per-role. Dialogue mode assumes one actor per role.
- Multi-cascade *tables*. The runner routes multi-script boundaries via the
  `close` marker and the synthesizing trainer opens each script's own primary
  in turn (via ``primaries_from_source``), so a multi-script table runs
  script-to-script. What is not yet specified is authoring a *misfit* second
  script (e.g. a question script whose atoms are not all first-class compiled
  klines) — the S2-misfit pedagogy that motivates a developmental second
  script. The close is the ``close:true`` turn or the last row (§Table
  Structure); a multi-script table still has a single close.
- Measuring, detecting, or signalling learning or grounding.

## Canonical Example

The reference dialogue is the "Mary had a little lamb" exchange
(`scripts/dialogue-mhall.json`): a single depth-first cascade. The trainer
opens with the primary `{MHALL:[SVO]}` at S2; the trainee requests each unknown
operand at S4; the trainer supplies it; the trainee proposes the role bindings
(Mary↔subject, had↔verb, ALL↔object) at S3; the trainer ratifies each at S1; the
trainee closes with the primary's S1 countersign — emitting the reciprocal
pair (`{MHALL:[SVO]}` and `{SVO:[MHALL]}` at S1), since a COUNTERSIGNED state
is bidirectional (`@CONTEXT.md`, Structural State). The runner drives the two
table-reading actors over the harness message bus; it tracks coverage (zero
displacement when the whole exchange is traversed).

## Test Matrix

| ID     | Criterion                                                                                                                                                                                                                                                                                            | Origin ref             |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| DDT-1  | A dialogue table has `script` + ordered `turns`; each turn has actor, op (a Structural State), signature, nodes, significance, notes                                                                                                                                                                 | §Dialogue Table        |
| DDT-2  | Each turn's `op` is a Structural State (`@CONTEXT.md`)                                                                                                                                                                                                                                               | §Dialogue Table        |
| DDT-3  | The decoder pre-decodes every turn into a flat ordered `list[DecodedTurn]` at configuration time                                                                                                                                                                                                     | §Decoder               |
| DDT-4  | Decoding resolves the kline from `script`, attaches significance by lookup, passes through actor/op, drops annotation-only turns and ignores notes                                                                                                                                                   | §Decoder               |
| DDT-5  | A CANONIZED turn resolves each node label to its canonical signature and builds `KLine(signature, nodes)` with the declared signature verbatim (an author may declare a deliberate misfit)                                                                                                                | §Decoder               |
| DDT-6  | A dialogue table may carry an optional `run` section holding run modifiers; all run modifiers live in it; the loader resolves them into `run` inputs; unknown `run` keys are a decode error; the runner consumes `DecodedTurn`s, not the raw table                                                    | §Dialogue Table        |
| DDT-7  | The `turns` are an ordered list documenting cause and effect; that order is not enforced by the runner                                                                                                                                                                                             | §Dialogue Table        |
| DDT-8  | A decoded table is de-positional: a coverage set (every turn) plus one close (the `close:true` turn, or the last row) any agent may emit at any time                                                                                                                                                    | §Table Structure       |
| DDT-9  | The close content is unique — its `(role, kline, significance)` appears exactly once; a close duplicating a coverage row is malformed at decode time                                                                                                                                                 | §Table Structure       |
| DDT-10 | The run is driven by the harness `MessageBus`: the bus is the sink and relay; the runner is a coverage-tracking wildcard subscriber plus a thin driver that seeds the trainer and runs `bus.run()` until a terminal condition. The runner depends on `training.harness`                             | §The Runner            |
| DDT-11 | The runner holds coverage bookkeeping only (fixed distinct coverage set, growing covered subset, close key, closed flag, last-PASS role); whose-turn, per-actor cursors, turn tracking, pacing, idle deadline, and complete/covered judgment all live in the actors. The relay lives in the bus                                        | §Permitted state       |
| DDT-12 | An emission observed by the wildcard subscriber matches the distinct coverage contents / close by `(role, kline, significance)` content equality                                                                                                                                                     | §Matching              |
| DDT-13 | A content present in the distinct coverage set marks it covered (idempotent); duplicate table rows collapse to one distinct content; re-emitting covered content is not divergence                                                                                                                   | §Matching              |
| DDT-14 | Zero matches (and not the close) is immediate divergence: `on_divergence="fail"` raises `Divergence(role, emitted, unconsumed)`; `"accept"` appends to `RunResult.unmatched` and continues                                                                                                          | §Matching              |
| DDT-15 | The run terminates on the close content being emitted (any agent, any time), the coverage set being exhausted, or a mutual PASS; once closed, every subsequent emission is dropped                                                                                                                      | §Termination           |
| DDT-16 | Anticipation and interjection are permitted and unflagged across the whole table: an emission matching a same-role distinct content is a normal match regardless of authored causal order or whether it was solicited; there are no positional pins                                                  | §Anticipation          |
| DDT-18 | The runner terminates mechanically (close / exhaustion / mutual PASS; no idle timeout — every `accept` yields ≥1 proposal) and reports the **displacement** (`uncovered`) — coverage rows never emitted                                                                                                | §Termination, §Types   |
| DDT-19 | `Divergence` carries `(role, emitted, unconsumed)` and has no cursor (coverage is content-keyed)                                                                                                                                                                                                     | §Types                 |
| DDT-20 | `RunResult` has arrival-ordered `events`, plus `unmatched` (immediate divergences, accept-mode) and `uncovered` (displacement: coverage rows never emitted)                                                                                                                                           | §Types                 |
| DDT-21 | An actor holds an `EventSink` (injected at construction) and publishes via `on_event`; `accept(event)` receives incoming and the actor publishes one-or-many replies via its sink (`event=None` = "you open"); `burst >= 1` — when cogitation yields nothing the actor publishes a PASS (`{PASS: []}` at S1). The runner builds the bus-wired sink and constructs actors via factories `(sink) -> Actor`; any actor is drop-in | §Actor contract |
| DDT-22 | There is no synchronised alternation: an actor may reply one-or-many times per `accept` (at least one — a PASS when it has nothing substantive); each reply is routed by the bus to the other role. A PASS is intercepted before matching (not coverage, not divergence, not close); two consecutive PASSes from the two roles is a terminal condition (mutual PASS) | §The Runner, §Actor contract |
| DDT-23 | The trainer and trainee are symmetric readers of the same decoded table, differing only by actor; the default actors emit by index (`incoming` only supplies the emitted event's `query`)                                                                                                              | §Actor contract        |
| DDT-24 | The runner routes on the actor's self-declared `event.role`: the actor announces itself and the bus addresses its emissions to the other role — the shape a real, possibly asynchronous actor will use                                                                                              | §Matching              |
| DDT-25 | Training measures learning only indirectly, through the displacement; grounding is an internal state surfaced as significance on each emission. The runner's concern is the exchange and its coverage                                                                                               | §What Training Is      |
