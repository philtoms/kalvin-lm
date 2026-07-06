# Dialogue-Driven Training — Specification

## Overview

A lesson is an authored **dialogue** between a Trainer (T) and a Trainee (K),
turn by turn. The dialogue is a deterministic, table-driven artifact. A
**dialogue runner** loads and decodes the table, then alternates the two actors
until the table is exhausted. Each actor emits a `RationaliseEvent` per turn; the
runner hands each actor the other's last event and collects its response.

Both sides of the loop are coded so that either can be replaced by a real trainer
or a real trainee. The default actors supplied here are table-reading doubles;
they are not the design's point, only its scaffolding. The runner does not defend
against replacement — it will evolve when real actors arrive.

The runner is **bus-agnostic**: it is a plain Python loop over the two actors and
knows nothing about the harness message bus. Bus integration belongs to the real
actors, not to the runner.

## Dependencies

- `@CONTEXT.md` — Structural State, Canon, KValue, Significance.
- `@specs/kscript.md` — compiled-entry `op` field (the structural states).
- `@specs/kline.md` — `is_canon`, KLine equality.
- `@specs/kvalue.md` — KValue (KLine + significance).
- `@specs/agent.md` — `RationaliseEvent(kind, query, proposal)`.

## Definitions

### Dialogue Table

The source artifact for a lesson. A JSON object:

```
DialogueTable:
  script:  str          # KScript source, or a path to a .ks file — the
                         # authority for kline structure
  turns:   list[Turn]   # ordered, the exact T/K exchange
```

```
Turn:
  actor:        "T" | "K"
  op:           str   # a Structural State: COUNTERSIGNED | CANONIZED |
                      #                CONNOTED | UNDERSIGNED | IDENTITY
  signature:    str   # symbolic label, resolved by the decoder
  nodes:        list[str]   # symbolic labels, resolved by the decoder
  significance: "S1" | "S2" | "S3" | "S4"
  notes:        str   # human commentary; ignored by the decoder
  close:        int   # optional (1, 2, …): this turn closes script `close`.
                      # The runner reads it as the script boundary and routes
                      # the next turn as an open. Absent unless this turn is a
                      # close; must be on a trainee (K) row.
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

An **annotation-only turn** carries `notes` but no `op`; it is human commentary
and is not part of the exchange.

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

1. **Resolve the kline from `script`** (script is authority):
   - **CANONIZED** — retrieve the compiled canon whose node-decoded labels match
     the turn's `nodes` list (node-list match). The turn's symbolic node names
     are the retrieval key.
   - **IDENTITY** — resolve the atom by label.
   - **Constructed relation** (`COUNTERSIGNED` / `CONNOTED` / `UNDERSIGNED`) —
     resolve each node label to its canonical signature and rebuild the relation
     KLine.
2. **Attach significance** by band lookup (`"S1"→SIG_S1`, …) from the turn.
3. **Pass through** `actor` and `op`; **drop** annotation-only turns and ignore
   `notes` on the rest.

Subword node names must be the tokenizer's actual decoded subwords; the table is
maintained against real compiler output.

## The Runner

```
run(decoded, trainer, trainee) -> RunResult
```

`decoded` is the single shared `list[DecodedTurn]`. `trainer` and `trainee` are
objects satisfying the Actor interface (§Actor). **The runner owns the table
cursor.** Each step: read the actor of `decoded[cursor]`, ask that actor for its
next row, validate it, advance. The run ends when the cursor passes the end of
the table.

**Script boundaries are table-declared** (§Dialogue Table `close`). When a turn
carries a `close: n` marker, it ends script `n`; the runner routes the *next*
turn with `incoming=None` (an open), so the next actor opens a fresh script
rather than replying to the close. Close-detection is the runner's job (it owns
the cursor and the markers); the trainer does not detect closes.

Dispatch is **greedy** — and greediness is the runner's behaviour, not the
actor's. While consecutive table rows share an actor, the runner asks the same
actor again. For a strictly alternating table that is one row per actor; a
table with `T,T,K` asks the trainer twice, then the trainee once. The actor
itself does not know about alternation or run boundaries; it only yields its
own rows in order.

> The trainer and trainee are structurally symmetric: both are cursor readers
> of the same table, differing only by which `actor` they read. That symmetry
> is what makes either replaceable by a real actor.

### Actor

```
Actor:
  respond(incoming: RationaliseEvent | None) -> RationaliseEvent | None
```

Each actor yields its rows one at a time in order. `respond` returns the event
for its next row, or `None` when the actor is exhausted. The actor holds **no**
cursor it returns to the runner; it does not know the table's actor sequence,
and the runner decides whose turn it is. The event's `proposal` is the actor's
turn; `query` is the `incoming` event's `proposal` (or the actor's own turn on
the opening). **Every event carries the actor's role** (`event.role`) — the
self-declared routing key (the same discriminator the harness bus calls
_role_, `@CONTEXT.md` §Role). The runner uses it to route and validate; the
actor announces itself rather than being identified by table position. **The
runner owns the validation index** into the full decoded table; the actor
returns only its event.

Both default actors (`TableTrainer`, `TableTrainee`) read the decoded table,
filter to their own `actor`, and yield those rows in order with their role on
each event. They never inspect the incoming event to decide what to emit.

## Validation

The runner is a **router**: it validates each response against the row at its
own full-table cursor (`decoded[cursor]`), by the **role the actor declared on
its event** (`event.role`), not by the table's view of who responded. It checks
`event.role == decoded[cursor].role` and the emitted `proposal` (KLine +
significance) equals `decoded[cursor].value`. A role mismatch or a
kline/significance mismatch is an `ActorDivergence` and fails the run, naming
the role, the cursor, and the expected vs. emitted turn.

A real trainer is expected to synthesise its training responses, and a real
trainee its responses; both announce their role and are validated against the
authored table. Under the table-reading `TableTrainer` / `TableTrainee` a
divergence cannot occur — they read the same table the runner validates
against. The checks exist for the real actors: they are the mechanism that
ensures the correct next response to an event is available, by failing when it
is not. Keying on the event's self-declared role (rather than the table key) is
what lets a real, possibly asynchronous actor announce itself: the runner
routes on what the actor said, checking it against the table.

## What Training Is

Training is a deterministic mechanism that ensures the correct next response to
a trainee event is available. It does not measure learning — learning is not
measurable, and Kalvin learns through experience. Grounding means Kalvin
understands; it is not an event, not broadcast, and not part of this loop. The
runner carries no notion of "learned" and emits no grounding signal.

## Out of Scope

- How a real trainer produces its turns, or a real trainee its responses. Both
  arrive with their own cogitation and (for the trainee) memory; the runner's
  Actor interface is the contract they satisfy.
- Bus integration. The real actors bring the harness message bus into play; the
  runner does not.
- Supervisor escalation on a request the trainer cannot resolve — belongs to the
  real trainer.
- Multi-cascade *synthesis*. The runner routes multi-script boundaries via the
  `close` marker (a multi-script table runs script-to-script), but the
  synthesizing trainer still opens only the first compiled primary; opening
  successive scripts' primaries is not yet implemented.
- Measuring, detecting, or signalling learning or grounding.

## Canonical Example

The reference dialogue is the "Mary had a little lamb" exchange
(`scripts/dialogue-mhall.json`): a single depth-first cascade. The trainer opens
with the primary `{MHALL:[SVO]}` at S2; the trainee requests each unknown
operand at S4; the trainer supplies it; the trainee proposes the role bindings
(Mary↔subject, had↔verb, ALL↔object) at S3; the trainer ratifies each at S1; the
trainee closes with the primary's S1 countersign. The runner alternates the two
table-reading actors to exhaustion.

## Test Matrix

| ID     | Criterion                                                                                                                                                                                                                                                                                            | Origin ref             |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| DDT-1  | A dialogue table has `script` + ordered `turns`; each turn has actor, op (a Structural State), signature, nodes, significance, notes                                                                                                                                                                 | §Dialogue Table        |
| DDT-2  | Each turn's `op` is a Structural State (`@CONTEXT.md`)                                                                                                                                                                                                                                               | §Dialogue Table        |
| DDT-3  | The decoder pre-decodes every turn into a flat ordered `list[DecodedTurn]` at configuration time                                                                                                                                                                                                     | §Decoder               |
| DDT-4  | Decoding resolves the kline from `script`, attaches significance by lookup, passes through actor/op, drops annotation-only turns and ignores notes                                                                                                                                                   | §Decoder               |
| DDT-5  | A CANONIZED turn is retrieved by node-list match against compiled canons                                                                                                                                                                                                                             | §Decoder               |
| DDT-6  | The runner owns the table cursor: each step reads whose row is next and asks that actor; the first row is the trainer's                                                                                                                                                                              | §The Runner            |
| DDT-7  | Greediness is the runner's behaviour: while consecutive table rows share an actor, the same actor is asked again (e.g. `T,T,K` asks the trainer twice, then the trainee)                                                                                                                             | §The Runner            |
| DDT-8  | The trainer and trainee are symmetric cursor readers of the same decoded table, differing only by actor                                                                                                                                                                                              | §The Runner            |
| DDT-9  | An actor yields one `RationaliseEvent` per `respond`, returning just the event (no cursor); the event's `proposal` is the row's KValue, `query` is the incoming event's proposal, and `role` is the actor's self-declared routing key                                                                    | §Actor                 |
| DDT-10 | The default actors never inspect the incoming event to decide what to emit                                                                                                                                                                                                                           | §Actor                 |
| DDT-11 | The runner validates by the role the actor declared on its event (not the table key): each emitted `proposal` must equal the decoded row for that role at the cursor; a role with no table rows or a kline/significance mismatch raises `ActorDivergence` naming role, cursor, expected, and emitted | §Validation            |
| DDT-12 | (removed) The runner no longer treats the trainer as unvalidated; both actors are validated (see DDT-11)                                                                                                                                                                                             | §Validation            |
| DDT-13 | The run ends when the cursor passes the end of the table                                                                                                                                                                                                                                             | §The Runner            |
| DDT-14 | The runner is bus-agnostic (no harness message-bus dependency)                                                                                                                                                                                                                                       | §Overview, §The Runner |
| DDT-15 | The runner carries no notion of learned/grounding and emits no grounding signal                                                                                                                                                                                                                      | §What Training Is      |
| DDT-16 | The runner owns the validation index into the full decoded table; an actor yields its next row's event or `None` when exhausted, returning no cursor of its own                                                                                                                                                             | §Actor                 |
| DDT-17 | The runner is a router: an actor announces itself via `event.role`, and the runner routes/validates on that self-declared role — the shape a real, possibly asynchronous actor will use                                                                                                              | §Validation            |
