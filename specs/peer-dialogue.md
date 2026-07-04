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

A `DialogueTable` declares its run regime. The synchronous regime is the
default; a peer-mode table declares peer mode and a divergence policy. The
authoring knobs live on the table; the loader resolves them into `run_peer`
inputs. The runner itself is format-agnostic (it consumes `DecodedTurn`s, not
the raw table).

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
- The opening and the closing are **content-distinct** (different
  `(role, kline, significance)`). A peer-mode table whose opening and closing
  are content-equal is malformed.

## The Runner

```
run_peer(decoded, *, on_divergence) -> PeerRunner
```

`run_peer` is a **sink**, not a driver. It does not call into the actors; it
does not decide whose turn it is; it carries no actor-coupling state. After the
caller has delivered the opening to the trainee, the actors emit on their own
and push emissions into the runner. The runner receives, validates against the
coverage set, and watches for the closing.

The synchronous `run` and its `Actor`/`respond` contract are untouched.

### Sink contract

The runner exposes a push entry point:

```
PeerRunner.receive(event: RationaliseEvent) -> None
PeerRunner.complete -> bool        # closing seen AND middle coverage satisfied
PeerRunner.result -> PeerRunResult
```

`receive` is the sole entry point once the run has begun. Emissions are
accepted in arrival order; the runner imposes no ordering on the caller.

### Permitted state

The runner holds **coverage bookkeeping** only: the set of unconsumed distinct
middle contents, and a `closing-seen` flag. It holds **no** actor-coupling
state — no notion of whose turn it is, no per-actor cursors, no pacing, no
retry counts. Coverage bookkeeping is the runner's own accounting as a sink and
does not couple it to any actor.

## Matching

Each received emission is matched against the unconsumed **same-role** middle
rows by content equality `(role, kline, significance)`:

- **One or more matches** — consume the distinct content. Duplicate rows
  collapse: a single emission satisfying content X consumes all unconsumed
  rows with content X in one step.
- **Zero matches, and the emission is not the closing** — **divergence**. Under
  `on_divergence="fail"` the runner raises `PeerDivergence`. Under
  `on_divergence="accept"` the emission is recorded in `PeerRunResult.unmatched`
  and the run continues.
- **The emission equals the closing content** — mark `closing-seen` and consume
  the closing.

### Anticipation

The table's order documents cause and effect but is not enforced within the
middle. An actor emitting content that, in the authored causal order, depends
on a not-yet-emitted cause (e.g. a trainer ratifying before the request
arrives) is a normal match against whatever unconsumed same-role row its
content equals. Anticipation is **not** divergence, **not** flagged, and
**not** recorded specially. It is permitted behaviour.

The positional pins (opening first, closing last) are the **only** ordering
constraints the runner enforces. Anticipation is permitted only within the
middle zone; the opening and closing are not anticipatable.

## Completion

```
complete = closing-seen AND (middle distinct-set exhausted)
```

The middle distinct-set is exhausted when every distinct
`(role, kline, significance)` among the middle rows has been consumed (via
match or collapse). The closing is consumed only by an emission matching its
content.

Coverage is a **measured property**, reported in `PeerRunResult`, not a
terminal condition on its own: a run may continue after the middle is covered
if the closing has not arrived (further middle emissions would then be
divergences). Completion is the conjunction of closing-seen and middle
exhausted.

## Types

```
PeerDivergence(Exception):
    role:           str                  # the role of the divergent emission
    emitted:        KValue               # the unmatched emission's proposal
    unconsumed:     tuple[DecodedTurn]   # unconsumed same-role rows at the moment of divergence

PeerRunResult:
    events:    list[RationaliseEvent]    # every received emission, in ARRIVAL order
    complete:  bool                      # closing seen AND middle distinct-set exhausted
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

- How the opening is delivered to the trainee. That is the **caller's**
  responsibility; the runner is a pure sink from the first emission onward.
- How real actors produce their emissions, or any concurrency/threading model.
  Push/sink is the contract; whether a test pushes synchronously or the
  harness pushes from threads is the caller's concern. The runner imports no
  concurrency primitive.
- Multi-actor-per-role. Peer mode assumes one actor per role against the
  unconsumed same-role set.
- Reject-and-re-prompt (an actor revising a non-matching turn). Requires a
  rejection channel back to the actor, which is a contract change beyond the
  sink defined here. Deferred.
- Per-turn ordering constraints (gating a turn on the prior consumption of
  another). The whole-table middle floats; dependencies are enforced only
  indirectly, via content-match failure.
- Bus integration beyond the sink shape. The runner is bus-*shaped* (a
  role-addressed sink) without being the threaded harness bus.

## Test Matrix

| ID    | Criterion                                                                                                                                                                                      | Origin ref        |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| PDT-1 | A `DialogueTable` declares its run regime; a peer-mode table declares peer mode and a divergence policy, resolved by the loader into `run_peer` inputs; the runner consumes `DecodedTurn`s, not the raw table | §The Table in Peer Mode |
| PDT-2 | The `turns` of a peer-mode table are an ordered list documenting cause and effect; that order is not enforced within the middle by the runner                                                  | §The Table in Peer Mode |
| PDT-3 | A peer-mode decoded table has three zones: opening (`decoded[0]`, first, positional, trainer), middle (`decoded[1:-1]`, coverage set), closing (`decoded[-1]`, last, positional)               | §Zones            |
| PDT-4 | The opening is a trainer row; the opening and closing are content-distinct; a table violating either is malformed at decode time                                                                | §Invariants       |
| PDT-5 | `run_peer` is a sink: it exposes `receive`, does not call into actors, decides no turns, and holds no actor-coupling state                                                                      | §The Runner, §Sink contract, §Permitted state |
| PDT-6 | The runner holds coverage bookkeeping only (unconsumed distinct middle set, closing-seen flag); no per-actor cursors, turn tracking, or pacing                                                  | §Permitted state  |
| PDT-7 | A received emission matches unconsumed same-role middle rows by `(role, kline, significance)` content equality                                                                                  | §Matching         |
| PDT-8 | One or more matches consume the distinct content; duplicate rows collapse in one step (one emission of X consumes all unconsumed X rows)                                                       | §Matching         |
| PDT-9 | Zero matches (and not the closing) is divergence: `on_divergence="fail"` raises `PeerDivergence(role, emitted, unconsumed)`; `"accept"` appends to `PeerRunResult.unmatched` and continues      | §Matching         |
| PDT-10 | An emission equal to the closing's content marks `closing-seen` and consumes the closing                                                                                                       | §Matching         |
| PDT-11 | Anticipation within the middle is permitted and unflagged: an emission matching an unconsumed same-role row by content is a normal match regardless of authored causal order                   | §Anticipation     |
| PDT-12 | The opening and closing are the only positional constraints; anticipation applies to the middle only; the opening is not anticipatable                                                          | §Anticipation     |
| PDT-13 | `complete = closing-seen AND middle distinct-set exhausted`; coverage alone does not terminate the run                                                                                          | §Completion       |
| PDT-14 | `PeerDivergence` is a separate type carrying `(role, emitted, unconsumed)`, distinct from synchronous `ActorDivergence`                                                                         | §Types            |
| PDT-15 | `PeerRunResult` is a separate type with arrival-ordered `events`, plus `complete`, `covered`, `unmatched` (accept-mode), `uncovered` (incomplete runs); distinct from synchronous `RunResult`   | §Types            |
| PDT-16 | The synchronous `run`, the `Actor`/`respond` contract, `ActorDivergence`, and `RunResult` are unchanged by this spec                                                                            | §Overview, §Types |
