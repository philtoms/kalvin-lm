# Dialogue-Driven Training — Specification

## Overview

A lesson is an authored **dialogue** between a Trainer (T) and a Trainee (K),
turn by turn. The dialogue is a deterministic, table-driven artifact. A
**dialogue runner** loads and decodes the table, then drives the two actors
until the table is exhausted.

After the trainer delivers the opening entry to the trainee, both sides emit on
their own schedule, in any order and any count; the runner is a **sink** that
receives emissions, validates them against a coverage set, and watches for the
closing entry. Order within the middle is not enforced; anticipation (an actor
emitting ahead of the table's causal order) is permitted and unflagged.

Both sides of the loop are coded so that either can be replaced by a real trainer
or a real trainee. The default actors supplied here are table-reading doubles;
they are not the design's point, only its scaffolding. The runner does not defend
against replacement — it will evolve when real actors arrive.

The runner is **bus-driven**: it drives the exchange over the harness
`MessageBus`. The actors are sink-driven — each holds an `EventSink` and
publishes its turns via `accept` (fire-and-forget, zero-or-many per incoming).

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
  close:        bool  # optional: `true` marks this turn as a script close.
                      # The runner reads it as the script boundary and routes
                      # the next turn as an open. Absent (or false) unless
                      # this turn is a close. Role-agnostic: it may sit on
                      # either a trainer (T) or trainee (K) row — closing is
                      # a runner concern, not a role constraint.
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
order is authorial/human-facing; the runner does not enforce it within the
middle zone (§Anticipation).

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
   **resolver, not a gatekeeper**: it builds the kline the turn declares — the
   declared `signature` verbatim, the `nodes` resolved to their canonical
   signatures — and never checks that the signature "matches" the nodes. An
   author may declare a signature that differs from the canon its nodes form
   (a deliberate misfit — see `scripts/dialogue-rationalisation-behaviours.md`).
   - **CANONIZED** — resolve each node label to its canonical signature
     (canon-preferred, atom fallback) and build `KLine(signature, nodes)` with
     the declared signature verbatim. No canon retrieval by node-list, no
     signature-consistency check.
   - **IDENTITY** — resolve the atom by label.
   - **Constructed relation** (`COUNTERSIGNED` / `CONNOTED` / `UNDERSIGNED`) —
     resolve each node label to its canonical signature and rebuild the relation
     KLine.
2. **Attach significance** by band lookup (`"S1"→SIG_S1`, …) from the turn.
3. **Pass through** `actor` and `op`; **drop** annotation-only turns and ignore
   `notes` on the rest.

Every signature and node label must resolve to a compiled entry (a label not
found in the script is a decode error). What the decoder does *not* do is
relate the signature to the nodes: that relationship is the author's to
declare, not the decoder's to police.

The decode path also validates the run-time **zones** (§Zones): the opening is
a trainer row; the closing is content-distinct from the opening and from every
middle row. A table violating any of these is malformed at decode time.

## Zones

A decoded table has three zones, two of them positionally pinned:

- **Opening** — `decoded[0]`. Consumed **first**, positionally. Trainer-emitted.
- **Middle** — `decoded[1:-1]`. The **coverage set**. Not positionally
  constrained; matched by content; order not enforced; duplicates collapse.
- **Closing** — `decoded[-1]`. Consumed **last**, positionally. The final
  expectation; completion requires it.

### Invariants

- The opening is a trainer (`T`) row.
- The closing is **content-distinct** from the opening **and** from every
  middle row: its `(role, kline, significance)` appears exactly once in the
  table. A closing that duplicates the opening would make coverage degenerate
  (the opening satisfies it); a closing that duplicates a middle row would make
  the positional "consumed last" semantics ambiguous (which occurrence is the
  closing?). A table violating either is malformed at decode time.

## The Runner

```
run(decoded, trainer_factory, trainee_factory, *, on_divergence, idle_timeout) -> Runner
```

The run is driven by the harness **`MessageBus`**
(`src/training/harness/bus.py`). The bus is the **sink and the relay**; the
runner is a **coverage-tracking wildcard subscriber** plus a thin driver that
seeds the opening and runs the bus until the closing is seen. The runner
depends on `training.harness` — it is a training application and belongs next
to the harness.

- **Sink = the bus (via a bus-wired `EventSink`).** Actors publish events to an
  `EventSink` injected at construction; the runner's bus-wired sink bridges
  each `on_event` to a `Message` addressed to the other role. The bus's `send`
  is thread-safe — actors publish from any thread (including a cogitation
  thread), which is the true non-blocking behaviour the run regime requires.
  No `asyncio`; no second concurrency model. The actor does not know about the
  bus; it publishes to its sink, and the sink routes.
- **Relay = the bus's role dispatch.** Each actor subscribes to its own role;
  the bus-wired sink addresses each published event to the **other** role; the
  bus delivers. The runner does not relay — the bus does. The runner never
  reroutes.
- **Coverage = a wildcard subscriber.** The runner subscribes to `*` (every
  message), updates its distinct-middle / covered / closing-seen bookkeeping on
  each emission, and calls `bus.stop()` when `closing_seen`.
- **Driver.** A thin `run()` seeds the opening by addressing a `Message` to the
  trainer's role (`message=None` signals "you open"), then runs `bus.run()` on
  a dedicated thread until `closing_seen` or the idle timeout fires.

The dialogue metaphor: messy and real. **No synchronised alternation** — an
actor may emit zero-or-many replies per incoming message, and the relay does
not track turns. **Anticipation** (emitting ahead of authored causal order) and
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
the opening seed) and returns immediately; the actor decides when and whether
to publish events via its sink — possibly later (from a cogitation thread),
possibly many times (a priming burst, a scaffolding sequence), possibly never.
The actor does not know about the bus; it merely publishes to its sink, and the
sink routes.

The runner constructs actors via **factories** ``(sink) -> Actor``: only the
runner owns the bus, so only it can build the bus-wired sink, so it builds the
actors too. This makes any actor — including `SynthesizingTrainer` and
`RationalisingTrainee` — drop-in.

Both default actors (`TableTrainer`, `TableTrainee`) read the decoded table,
filter to their own `actor`, and yield those rows in order with their role on
each event. They never inspect the incoming event to decide what to emit.

### Permitted state

The runner holds **coverage bookkeeping** only (as a bus subscriber): the
table's **fixed set of distinct middle contents**, a **covered subset** that
grows monotonically as emissions match, a `closing-seen` flag, and the idle
deadline. It holds **no** actor-coupling state — no notion of whose turn it is,
no per-actor cursors, no pacing, no retry counts. The relay lives in the bus;
the runner only observes and records.

## Matching

The runner is a **content matcher**. Each emission observed by the wildcard
subscriber is matched against the table's **distinct middle contents** and the
closing by content equality `(role, kline, significance)`:

- **Equals the closing** — mark `closing-seen`.
- **Present in the distinct middle contents** — mark that content **covered**.
  Duplicate table rows collapsed to this one distinct content at construction;
  coverage is **idempotent** — re-emitting already-covered content is *not*
  divergence (it leaves the content covered).
- **Present nowhere in the table** (neither closing nor any middle content) —
  **divergence**. Under `on_divergence="fail"` the runner raises `Divergence`.
  Under `on_divergence="accept"` the emission is recorded in
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
by the bus to the other role. Both anticipation and interjection are **not**
divergence, **not** flagged, and **not** recorded specially. They are permitted
behaviour — the dialogue is messy and real, and agents must rationalise and
cogitate to make sense of it.

The positional pins (opening first, closing last) are the **only** ordering
constraints; they are enforced by decode-time validation and call order, not
by the relay. Anticipation and interjection apply to the middle zone only; the
opening and closing are not anticipatable.

## Completion

```
complete = closing-seen
```

Completion is the **closing entry alone** — the only really important goal. The
closing is consumed only by an emission matching its content.

Coverage is a measure of **efficiency**, not a matching count, and is **not a
terminal condition**: duplicate table rows collapse to one distinct content,
and `covered` reports whether every distinct middle content has been seen at
least once. A run is complete the moment the closing arrives, regardless of how
much of the middle was seen. Extreme anticipation (closing-first, zero middle
coverage) is technically complete, though rare in practice; `covered` makes the
inefficiency visible. The coverage fraction becomes a meaningful signal when a
training strategy thins the middle before start (e.g. randomly removing
entries), making completion closing-driven and coverage the measure of how much
of the authored exchange the actors actually traversed.

### Termination and stall

The driver runs `bus.run()` until `closing_seen` (the subscriber calls
`bus.stop()`). Because `accept` is fire-and-forget and replies are zero-or-many,
there is no "actor finished" signal; the run ends only on the closing. If the
actors go silent before the closing — a **stall** — the **idle timeout** ends
the run: a silence-bounded deadline (the bus's `queue.get(timeout=...)`); if no
emission arrives within it and the closing has not been seen, the run stops
incomplete (`RunResult.complete = False`). The idle timeout bounds quiescence,
not work — a long but active cogitation that keeps emitting does not trip it;
only true silence does. A timeout is a *property of the result* (the run did
not finish), not a divergence (nothing went wrong); it is surfaced in
`RunResult` (non-fatal), not raised.

## Types

```
Divergence(Exception):
    role:           str                  # the role of the divergent emission
    emitted:        KValue               # the unmatched emission's proposal
    unconsumed:     tuple[DecodedTurn]   # unconsumed same-role rows at the moment of divergence

RunResult:
    events:    list[RationaliseEvent]    # every received emission, in ARRIVAL order
    complete:  bool                      # closing-seen (the only terminal goal)
    covered:   bool                      # every distinct middle content emitted
    unmatched: list[RationaliseEvent]    # emissions matching nothing (accept-mode only)
    uncovered: list[DecodedTurn]         # distinct middle rows never consumed (incomplete runs)
```

`RunResult.events` is **arrival-ordered** — every received emission in the order
the bus delivered it. `unmatched` is populated only under
`on_divergence="accept"`; `uncovered` lists the distinct middle rows never seen
(an efficiency diagnostic for incomplete runs).

## What Training Is

Training is a deterministic mechanism that ensures the correct next response to
a trainee event is available. It does not measure learning — learning is not
measurable, and Kalvin learns through experience. Grounding means Kalvin
understands; it is not an event, not broadcast, and not part of this loop. The
runner carries no notion of "learned" and emits no grounding signal.

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
  script. The runner still treats the last row as the single close.
- Measuring, detecting, or signalling learning or grounding.

## Canonical Example

The reference dialogue is the "Mary had a little lamb" exchange
(`scripts/dialogue-mhall.json`): a single depth-first cascade. The trainer opens
with the primary `{MHALL:[SVO]}` at S2; the trainee requests each unknown
operand at S4; the trainer supplies it; the trainee proposes the role bindings
(Mary↔subject, had↔verb, ALL↔object) at S3; the trainer ratifies each at S1; the
trainee closes with the primary's S1 countersign. The runner drives the two
table-reading actors over the harness message bus to completion.

## Test Matrix

| ID     | Criterion                                                                                                                                                                                                                                                                                            | Origin ref             |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| DDT-1  | A dialogue table has `script` + ordered `turns`; each turn has actor, op (a Structural State), signature, nodes, significance, notes                                                                                                                                                                 | §Dialogue Table        |
| DDT-2  | Each turn's `op` is a Structural State (`@CONTEXT.md`)                                                                                                                                                                                                                                               | §Dialogue Table        |
| DDT-3  | The decoder pre-decodes every turn into a flat ordered `list[DecodedTurn]` at configuration time                                                                                                                                                                                                     | §Decoder               |
| DDT-4  | Decoding resolves the kline from `script`, attaches significance by lookup, passes through actor/op, drops annotation-only turns and ignores notes                                                                                                                                                   | §Decoder               |
| DDT-5  | A CANONIZED turn resolves each node label to its canonical signature and builds `KLine(signature, nodes)` with the declared signature verbatim — no canon retrieval by node-list, no signature-consistency check (an author may declare a deliberate misfit)                                          | §Decoder               |
| DDT-6  | A dialogue table may carry an optional `run` section holding run modifiers; all run modifiers live in it; the loader resolves them into `run` inputs; unknown `run` keys are a decode error; the runner consumes `DecodedTurn`s, not the raw table                                                    | §Dialogue Table        |
| DDT-7  | The `turns` are an ordered list documenting cause and effect; that order is not enforced within the middle by the runner                                                                                                                                                                             | §Dialogue Table        |
| DDT-8  | A decoded table has three zones: opening (`decoded[0]`, first, positional, trainer), middle (`decoded[1:-1]`, coverage set), closing (`decoded[-1]`, last, positional)                                                                                                                              | §Zones                 |
| DDT-9  | The opening is a trainer row; the closing is content-distinct from the opening **and** from every middle row; a table violating either is malformed at decode time                                                                                                                                  | §Zones, §Invariants    |
| DDT-10 | The run is driven by the harness `MessageBus`: the bus is the sink and relay; the runner is a coverage-tracking wildcard subscriber plus a thin driver that seeds the opening and runs `bus.run()` until the closing. The runner depends on `training.harness`                                       | §The Runner            |
| DDT-11 | The runner holds coverage bookkeeping only (fixed distinct middle content set, growing covered subset, closing-seen flag, idle deadline); no per-actor cursors, turn tracking, or pacing. The relay lives in the bus                                                                                  | §Permitted state       |
| DDT-12 | An emission observed by the wildcard subscriber matches the distinct middle contents / closing by `(role, kline, significance)` content equality                                                                                                                                                     | §Matching              |
| DDT-13 | A content present in the distinct middle marks it covered (idempotent); duplicate table rows collapse to one distinct content; re-emitting covered content is not divergence                                                                                                                         | §Matching              |
| DDT-14 | Zero matches (and not the closing) is divergence: `on_divergence="fail"` raises `Divergence(role, emitted, unconsumed)`; `"accept"` appends to `RunResult.unmatched` and continues                                                                                                                   | §Matching              |
| DDT-15 | An emission equal to the closing's content marks `closing-seen` and consumes the closing                                                                                                                                                                                                             | §Matching              |
| DDT-16 | Anticipation and interjection within the middle are permitted and unflagged: an emission matching a same-role distinct content is a normal match regardless of authored causal order or whether it was solicited                                                                                     | §Anticipation          |
| DDT-17 | The opening and closing are the only positional constraints (enforced by decode-time validation and call order, not by the relay); anticipation/interjection apply to the middle only; the opening is not anticipatable                                                                              | §Anticipation          |
| DDT-18 | `complete = closing-seen`; an idle timeout ends a stalled run (silence, no closing) as incomplete (`complete = False`, non-fatal); coverage is a separate efficiency diagnostic, not a terminal condition (extreme anticipation — closing-first, zero middle coverage — is technically complete)        | §Completion            |
| DDT-19 | `Divergence` carries `(role, emitted, unconsumed)` and has no cursor (coverage is content-keyed)                                                                                                                                                                                                     | §Types                 |
| DDT-20 | `RunResult` has arrival-ordered `events`, plus `complete`, `covered`, `unmatched` (accept-mode), `uncovered` (incomplete runs)                                                                                                                                                                       | §Types                 |
| DDT-21 | An actor holds an `EventSink` (injected at construction) and publishes via `on_event`; `accept(event)` receives incoming and the actor publishes zero-or-many replies via its sink (`event=None` = "you open"); the runner builds the bus-wired sink and constructs actors via factories `(sink) -> Actor`; any actor is drop-in | §Actor contract |
| DDT-22 | There is no synchronised alternation: an actor may reply zero-or-many times per `accept`; each reply is routed by the bus to the other role; the runner never reroutes                                                                                                                                | §The Runner            |
| DDT-23 | The trainer and trainee are symmetric readers of the same decoded table, differing only by actor; the default actors never inspect the incoming event to decide what to emit                                                                                                                         | §Actor contract        |
| DDT-24 | The runner routes on the actor's self-declared `event.role`: the actor announces itself and the bus addresses its emissions to the other role — the shape a real, possibly asynchronous actor will use                                                                                              | §Matching              |
| DDT-25 | The runner carries no notion of learned/grounding and emits no grounding signal                                                                                                                                                                                                                      | §What Training Is      |
