# Peer Dialogue — Specification

## Overview

A **peer dialogue** is an alternative run regime for an authored
`DialogueTable` (`@specs/dialogue-driven-training.md`). After the trainer
delivers the opening entry to the trainee, both sides emit on their own
schedule, in any order and any count; the runner is a **sink** that receives
emissions, validates them against a coverage set, and watches for the closing
entry. Order within the middle is not enforced; anticipation (an actor emitting
ahead of the table's causal order) is permitted and unflagged.

The synchronous `run` (`@specs/dialogue-driven-training.md` §The Runner) is
unchanged. This spec adds a sibling regime, `run_peer`, with its own contract
and result types. The two share the `DialogueTable`, the `DecodedTurn`, and the
content-equality notion of a match; they differ in control regime.

## Dependencies

- `@specs/dialogue-driven-training.md` — `DialogueTable`, `DecodedTurn`,
  `RationaliseEvent`, content-equality of a match (kline + significance).
- `@specs/agent.md` — `RationaliseEvent(kind, query, proposal, role)`.
- `@CONTEXT.md` — Role, KValue, Significance.

## The Table in Peer Mode

A `DialogueTable` declares its run regime via an optional **`peer` section**:
a table carrying a `peer` section is in peer mode; a table without one is in
the default ordered (synchronous) regime. There is no top-level `mode` field —
the section's presence *is* the selector. **All peer operations and modifiers
live in the `peer` section** (so future peer knobs extend it without touching
the rest of the table); the runner is format-agnostic and consumes
`DecodedTurn`s, not the raw table. The loader resolves the section into
`run_peer` inputs.

The single peer modifier today is:

```
peer:
  on_divergence: "fail" | "accept"   # default "fail"
```

Unknown keys inside `peer` are a decode error.

The `turns` of a peer-mode table are an ordered list that documents **cause and
effect** (a ratification follows its proposal, an identity supply follows its
request). That order is authorial/human-facing; the runner does not enforce it
within the middle zone (§Anticipation).

### Zones

A peer-mode decoded table has three zones, two of them positionally pinned:

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
  closing?). A peer-mode table violating either is malformed at decode time.

## The Runner

```
run_peer(decoded, trainer, trainee, *, on_divergence, idle_timeout) -> PeerRunner
```

The peer run is driven by the harness **`MessageBus`**
(`src/training/harness/bus.py`). The bus is the **sink and the relay**; the
peer runner is a **coverage-tracking wildcard subscriber** plus a thin driver
that seeds the opening and runs the bus until the closing is seen. The peer
runner depends on `training.harness` — it is a training application and belongs
next to the harness (ADR-0002).

- **Sink = the bus.** Actors reply by `bus.send(Message(role=<other>,
  action="accept", message=<event>))`. `send` is thread-safe — actors reply
  from any thread (including a cogitation thread), which is the true
  non-blocking behaviour the peer regime requires. No `asyncio`; no second
  concurrency model.
- **Relay = the bus's role dispatch.** Each actor subscribes to its own role;
  an actor addresses its replies to the **other** role; the bus delivers. The
  runner does not relay — the bus does. The runner never reroutes.
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

The synchronous `run` and its `Actor`/`respond` contract are untouched.

### Actor contract

`Actor` gains `accept`:

```
Actor:
  accept(event: RationaliseEvent | None, sink: BusSink) -> None   # peer regime
  respond(incoming: RationaliseEvent | None) -> RationaliseEvent | None  # ordered regime
```

`accept` is **fire-and-forget**: it returns immediately; the actor replies
**zero or many** times by calling `sink.send(Message(role=<other>,
action="accept", message=<RationaliseEvent>))`. The actor decides when and
whether to reply, possibly later (from a cogitation thread), possibly many
times (a priming burst, a scaffolding sequence), possibly never. `event=None`
signals "you open." `BusSink` is a narrow handle exposing only `send` (the bus
satisfies it).

### Permitted state

The runner holds **coverage bookkeeping** only (as a bus subscriber): the
table's **fixed set of distinct middle contents**, a **covered subset** that
grows monotonically as emissions match, a `closing-seen` flag, and the idle
deadline. It holds **no** actor-coupling state — no notion of whose turn it is,
no per-actor cursors, no pacing, no retry counts. The relay lives in the bus;
the runner only observes and records.

## Matching

Each emission observed by the wildcard subscriber is matched against the
table's **distinct middle contents** and the closing by content equality
`(role, kline, significance)`:

- **Equals the closing** — mark `closing-seen`.
- **Present in the distinct middle contents** — mark that content **covered**.
  Duplicate table rows collapsed to this one distinct content at construction;
  coverage is **idempotent** — re-emitting already-covered content is *not*
  divergence (it leaves the content covered).
- **Present nowhere in the table** (neither closing nor any middle content) —
  **divergence**. Under `on_divergence="fail"` the runner raises
  `PeerDivergence`. Under `on_divergence="accept"` the emission is recorded in
  `PeerRunResult.unmatched` and the run continues.

### Anticipation and interjection

The table's order documents cause and effect but is not enforced within the
middle. An actor emitting content that, in the authored causal order, depends
on a not-yet-emitted cause (e.g. a trainer ratifying before the request
arrives) is a normal match against whatever same-role distinct content its
content equals. **Interjection** — emitting unsolicited, or emitting again
after the other side has gone quiet — is the same: a normal emission, routed
by the bus to the other role. Both anticipation and interjection are **not**
divergence, **not** flagged, and **not** recorded specially. They are
permitted behaviour — the dialogue is messy and real, and agents must
rationalise and cogitate to make sense of it.

The positional pins (opening first, closing last) are the **only** ordering
constraints; they are enforced by decode-time validation and call order, not
by the relay. Anticipation and interjection apply to the middle zone only;
the opening and closing are not anticipatable.

## Completion

```
complete = closing-seen
```

Completion is the **closing entry alone** — the only really important goal.
The closing is consumed only by an emission matching its content.

Coverage is a measure of **efficiency**, not a matching count, and is **not a
terminal condition**: duplicate table rows collapse to one distinct content,
and `covered` reports whether every distinct middle content has been seen at
least once. A run is complete the moment the closing arrives, regardless of
how much of the middle was seen. Extreme anticipation (closing-first, zero
middle coverage) is technically complete, though rare in practice; `covered`
makes the inefficiency visible. The coverage fraction becomes a meaningful
signal when a training strategy thins the middle before start (e.g. randomly
removing entries), making completion closing-driven and coverage the measure
of how much of the authored exchange the actors actually traversed.

### Termination and stall

The driver runs `bus.run()` until `closing_seen` (the subscriber calls
`bus.stop()`). Because `accept` is fire-and-forget and replies are zero-or-many,
there is no synchronous "actor finished" signal; the run ends only on the
closing. If the actors go silent before the closing — a **stall** — the **idle
timeout** ends the run: an silence-bounded deadline (the bus's
`queue.get(timeout=...)`); if no emission arrives within it and the closing
has not been seen, the run stops incomplete (`PeerRunResult.complete = False`).
The idle timeout bounds quiescence, not work — a long but active cogitation
that keeps emitting does not trip it; only true silence does. A timeout is a
*property of the result* (the run did not finish), not a divergence (nothing
went wrong); it is surfaced in `PeerRunResult` (non-fatal), not raised.

## Types

```
PeerDivergence(Exception):
    role:           str                  # the role of the divergent emission
    emitted:        KValue               # the unmatched emission's proposal
    unconsumed:     tuple[DecodedTurn]   # unconsumed same-role rows at the moment of divergence

PeerRunResult:
    events:    list[RationaliseEvent]    # every received emission, in ARRIVAL order
    complete:  bool                      # closing-seen (the only terminal goal)
    covered:   bool                      # every distinct middle content emitted
    unmatched: list[RationaliseEvent]    # emissions matching nothing (accept-mode only)
    uncovered: list[DecodedTurn]         # distinct middle rows never consumed (incomplete runs)
```

`PeerDivergence` and `PeerRunResult` are **separate** from the synchronous
`ActorDivergence` and `RunResult`. They carry peer-shaped data (an unconsumed
set, arrival-ordered events) that the synchronous types' cursor-shaped fields
cannot express. `PeerRunResult.events` is **arrival-ordered**, not
table-ordered — a deliberate difference from the synchronous `RunResult.events`.

## Out of Scope

- How real actors produce their emissions. `accept` is the contract; the
  actor's internal cogitation, threading, and reply timing are its own.
- Multi-actor-per-role. Peer mode assumes one actor per role.
- Reject-and-re-prompt (an actor revising a non-matching turn). Requires a
  rejection channel back to the actor, a contract change beyond `accept`.
  Deferred.
- Per-turn ordering constraints (gating a turn on the prior consumption of
  another). The whole-table middle floats; dependencies are enforced only
  indirectly, via content-match failure.
- A peer actor replying on a *different* bus than the run's own. The run owns
  its bus; actors reply to the sink they were handed.

## Test Matrix

| ID    | Criterion                                                                                                                                                                                      | Origin ref        |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| PDT-1 | A `DialogueTable` carries an optional `peer` section whose presence selects peer mode (no top-level `mode` field); all peer modifiers live in it; the loader resolves them into `run_peer` inputs; unknown `peer` keys are a decode error; the runner consumes `DecodedTurn`s, not the raw table | §The Table in Peer Mode |
| PDT-2 | The `turns` of a peer-mode table are an ordered list documenting cause and effect; that order is not enforced within the middle by the runner                                                  | §The Table in Peer Mode |
| PDT-3 | A peer-mode decoded table has three zones: opening (`decoded[0]`, first, positional, trainer), middle (`decoded[1:-1]`, coverage set), closing (`decoded[-1]`, last, positional)               | §Zones            |
| PDT-4 | The opening is a trainer row; the closing is content-distinct from the opening **and** from every middle row; a table violating any is malformed at decode time                                                                | §Invariants       |
| PDT-5 | The peer run is driven by the harness `MessageBus`: the bus is the sink and relay; the runner is a coverage-tracking wildcard subscriber plus a thin driver that seeds the opening and runs `bus.run()` until the closing. The runner depends on `training.harness` (ADR-0002)            | §The Runner       |
| PDT-6 | The runner holds coverage bookkeeping only (fixed distinct middle content set, growing covered subset, closing-seen flag, idle deadline); no per-actor cursors, turn tracking, or pacing. The relay lives in the bus                          | §Permitted state  |
| PDT-7 | An emission observed by the wildcard subscriber matches the distinct middle contents / closing by `(role, kline, significance)` content equality                                                                                  | §Matching         |
| PDT-8 | A content present in the distinct middle marks it covered (idempotent); duplicate table rows collapse to one distinct content; re-emitting covered content is not divergence                         | §Matching         |
| PDT-9 | Zero matches (and not the closing) is divergence: `on_divergence="fail"` raises `PeerDivergence(role, emitted, unconsumed)`; `"accept"` appends to `PeerRunResult.unmatched` and continues      | §Matching         |
| PDT-10 | An emission equal to the closing's content marks `closing-seen` and consumes the closing                                                                                                       | §Matching         |
| PDT-11 | Anticipation and interjection within the middle are permitted and unflagged: an emission matching a same-role distinct content is a normal match regardless of authored causal order or whether it was solicited                | §Anticipation and interjection |
| PDT-12 | The opening and closing are the only positional constraints (enforced by decode-time validation and call order, not by the relay); anticipation/interjection apply to the middle only; the opening is not anticipatable              | §Anticipation and interjection |
| PDT-13 | `complete = closing-seen`; an idle timeout ends a stalled run (silence, no closing) as incomplete (`complete = False`, non-fatal); coverage is a separate efficiency diagnostic, not a terminal condition (extreme anticipation — closing-first, zero middle coverage — is technically complete) | §Completion, §Termination and stall |
| PDT-14 | `PeerDivergence` is a separate type carrying `(role, emitted, unconsumed)`, distinct from synchronous `ActorDivergence`                                                                         | §Types            |
| PDT-15 | `PeerRunResult` is a separate type with arrival-ordered `events`, plus `complete`, `covered`, `unmatched` (accept-mode), `uncovered` (incomplete runs); distinct from synchronous `RunResult`   | §Types            |
| PDT-16 | The synchronous `run`, the `Actor`/`respond` contract, `ActorDivergence`, and `RunResult` are unchanged by this spec                                                                                                                                            | §Overview, §Types |
| PDT-17 | `Actor` gains `accept(event: RationaliseEvent | None, sink: BusSink) -> None` for the peer regime: fire-and-forget, zero-or-many replies via `sink.send(Message(role=<other>, ...))`; `event=None` signals "you open". `respond` is unchanged for the ordered regime     | §Actor contract   |
| PDT-18 | There is no synchronised alternation: an actor may reply zero-or-many times per `accept`; each reply is routed by the bus to the other role (the actor addresses replies to the non-emitter); the runner never reroutes          | §The Runner, §Actor contract |
| PDT-19 | Termination is `closing_seen` (the subscriber calls `bus.stop()`); the idle timeout is silence-bounded (the bus's `queue.get(timeout=...)`) and a stall is reported as `complete = False`, not raised                       | §Termination and stall |
