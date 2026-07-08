# Dialogue-Driven Training — Specification

## Overview

A lesson is an authored **dialogue** between a Trainer (T) and a Trainee (K),
turn by turn. The dialogue is a deterministic, table-driven artifact. A
**dialogue runner** loads and decodes the table, then drives the two actors
until the table is exhausted. Each actor emits a `RationaliseEvent` per turn;
the runner hands each actor the other's last event and collects its response.

Both sides of the loop are coded so that either can be replaced by a real trainer
or a real trainee. The default actors supplied here are table-reading doubles;
they are not the design's point, only its scaffolding. The runner does not defend
against replacement — it will evolve when real actors arrive.

The runner is **bus-driven**: it drives the exchange over the harness
`MessageBus` (spec `@specs/peer-dialogue.md`). The actors are adapter-driven —
each holds an `EventSink` and publishes its turns via `accept`
(fire-and-forget, zero-or-many per incoming), mirroring how `KAgent` publishes
via its adapter.

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
  priors:  list[str]    # optional: paths to other DialogueTable files whose
                         # turns run before this table's own, in list order.
                         # Resolved recursively; each prior's turns are
                         # inserted (in list order) ahead of this table's.
                         # A prior resolves its own `script`; only its turns
                         # are carried in.
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

## The Runner

The run is driven by the harness `MessageBus` (spec
`@specs/peer-dialogue.md` §The Runner): the bus is the sink and the relay; the
runner is a coverage-tracking wildcard subscriber plus a thin driver that seeds
the opening and runs the bus until the closing is seen. The runner validates
emissions by content match against the table's distinct middle contents and the
closing — not by a positional cursor.

**Script boundaries are table-declared** (§Dialogue Table `close`). When a turn
carries a `close: true` marker, it ends a script. Close-detection is the
runner's job (it owns the table and the markers); the trainer does not detect
closes — on the opening seed (`incoming=None`) it opens a fresh script's primary.

### Actor

```
PeerActor:
  role: str
  accept(incoming: RationaliseEvent | None) -> None   # publishes via its sink
```

Each actor holds an `EventSink` (injected at construction) and publishes its
turns to it. `accept` is **fire-and-forget**: it receives an incoming event (or
`None` for the opening seed) and returns immediately; the actor decides when and
whether to publish — possibly many times, possibly never. The event's `proposal`
is the actor's turn; `query` is the `incoming` event's `proposal` (or the actor's
own turn on the opening). **Every event carries the actor's role** (`event.role`)
— the self-declared routing key. The runner routes each published event to the
*other* role via the bus; the actor does not know about the bus.

Both default actors (`TableTrainer`, `TableTrainee`) read the decoded table,
filter to their own `actor`, and yield those rows in order with their role on
each event. They never inspect the incoming event to decide what to emit.

## Validation

The runner is a **content matcher**: each emission is matched against the table's
distinct middle contents and the closing by `(role, kline, significance)` content
equality. An emission matching a middle content marks it covered (idempotent);
an emission matching the closing marks completion; an emission matching nothing
is **divergence** (`PeerDivergence` under `on_divergence="fail"`, recorded in
`PeerRunResult.unmatched` under `"accept"`).

A real trainer is expected to synthesise its training responses, and a real
trainee its responses; both announce their role and are matched against the
authored table. Under the table-reading `TableTrainer` / `TableTrainee` a
divergence cannot occur — they read the same table the runner matches against.
The match exists for the real actors: it is the mechanism that surfaces an
emission the authored exchange did not authorise.

## What Training Is

Training is a deterministic mechanism that ensures the correct next response to
a trainee event is available. It does not measure learning — learning is not
measurable, and Kalvin learns through experience. Grounding means Kalvin
understands; it is not an event, not broadcast, and not part of this loop. The
runner carries no notion of "learned" and emits no grounding signal.

## Out of Scope

- How a real trainer produces its turns, or a real trainee its responses. Both
  arrive with their own cogitation and (for the trainee) memory; the runner's
  `PeerActor` interface is the contract they satisfy.
- Supervisor escalation on a request the trainer cannot resolve — belongs to the
  real trainer.
- Multi-cascade *tables*. The runner routes multi-script boundaries via the
  `close` marker and the synthesizing trainer opens each script's own primary
  in turn (via ``primaries_from_source``), so a multi-script table runs
  script-to-script. What is not yet specified is authoring a *misfit* second
  script (e.g. a question script whose atoms are not all first-class compiled
  klines) — the S2-misfit pedagogy that motivates a developmental second
  script. Multi-script is also out of scope (the runner still treats the last
  row as the single close).
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
| DDT-5  | A CANONIZED turn resolves each node label to its canonical signature and builds `KLine(signature, nodes)` with the declared signature verbatim — no canon retrieval by node-list, no signature-consistency check (an author may declare a deliberate misfit)         | §Decoder               |
| DDT-6  | (removed) The runner no longer owns a table cursor; emissions are matched by content (see `@specs/peer-dialogue.md` PDT-7)                                                                                                                                                                          | §The Runner            |
| DDT-7  | (removed) Greedy cursor dispatch is gone; an actor may reply zero-or-many per incoming (see `@specs/peer-dialogue.md` PDT-18)                                                                                                                                                                       | §The Runner            |
| DDT-8  | The trainer and trainee are symmetric readers of the same decoded table, differing only by actor                                                                                                                                                                                                     | §The Runner            |
| DDT-9  | An actor publishes one `RationaliseEvent` per row via its sink; the event's `proposal` is the row's KValue, `query` is the incoming event's proposal, and `role` is the actor's self-declared routing key                                                                                            | §Actor                 |
| DDT-10 | The default actors never inspect the incoming event to decide what to emit                                                                                                                                                                                                                           | §Actor                 |
| DDT-11 | (removed) Validation is content-match against the table, not cursor-keyed (see `@specs/peer-dialogue.md` PDT-7..PDT-9); a non-matching emission is `PeerDivergence`/`unmatched`, not `ActorDivergence`                                                                                              | §Validation            |
| DDT-12 | (removed) The runner no longer treats the trainer as unvalidated; both actors are matched (see DDT-11)                                                                                                                                                                                               | §Validation            |
| DDT-13 | (removed) Completion is closing-seen, not cursor-exhaustion (see `@specs/peer-dialogue.md` PDT-13)                                                                                                                                                                                                  | §The Runner            |
| DDT-14 | (removed) The runner is bus-driven, not bus-agnostic (see `@specs/peer-dialogue.md` PDT-5)                                                                                                                                                                                                           | §Overview, §The Runner |
| DDT-15 | The runner carries no notion of learned/grounding and emits no grounding signal                                                                                                                                                                                                                      | §What Training Is      |
| DDT-16 | (removed) There is no validation index; an actor publishes its rows via its sink and holds no cursor (see §Actor)                                                                                                                                                                                    | §Actor                 |
| DDT-17 | The runner routes on the actor's self-declared `event.role`: the actor announces itself and the bus addresses its emissions to the other role — the shape a real, possibly asynchronous actor will use                                                                                              | §Validation            |
