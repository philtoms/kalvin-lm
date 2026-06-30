# Table-Driven Stub KAgent — Specification

## Overview

A deterministic stand-in for Kalvin used to develop the trainer's dialogue-driven
loop independently of real-Kalvin cogitation. It implements the same harness
protocol as `KAgent` but, instead of rationalising, **emits its prescribed
K-turns from a shared dialogue table**, advancing its own cursor over the table's
K-rows on each `rationalise` call. It is **one of the two self-cursored actors**
in the training loop (`@specs/dialogue-driven-training.md` §Training Loop); the
trainer-side loop is the other.

The stub is a contract double, not a Kalvin: it has no model, no cogitation, no
expansion, no misfit reshape. The dialogue table scripts it deterministically. It
never matches the submitted kline — it emits its next K-run from its cursor. When
real Kalvin is reconciled to the same table (next grill), the stub is retired.

## Dependencies

- `@specs/dialogue-driven-training.md` — the dialogue table format, the pre-decoded
  `DecodedTurn`, the Training Loop (self-cursored actors, greedy dispatch,
  two-sided Model A validation, dual-exhaustion termination). The stub is the
  K-actor in that loop.
- `@specs/harness-server.md` — `_KAgentLike` protocol (`rationalise`, `countersign`,
  `save`, `codec`), `on_event` callback, the bus `submit`/`countersign`/`rationalise`
  actions.
- `@specs/agent.md` — `RationaliseEvent(kind, query, proposal)` shape; significance
  on the KValue, not the event.
- `@specs/kscript.md` — compiled-entry declared bands (`op` → S1/S2/S3/S4).

## Definitions

### The Dialogue Table (shared)

The stub consumes the **K-rows of the same pre-decoded dialogue table** the
trainer reads T-rows from — one shared `list[DecodedTurn]`, partitioned by
`actor`. There is no separate "response table"; the table is defined in
`@specs/dialogue-driven-training.md` §Dialogue Table / §Decoder. The stub's view
of it is its ordered subsequence of `actor == "K"` turns.

### Event Kinds the Stub Emits

Per the mental note (not yet spec'd globally), event kinds collapse toward
significance. The stub emits every K-turn as a `frame` event whose
`proposal.significance` is the turn's declared band:

| K-turn intent | Event `kind` | `proposal.significance` | Meaning |
|---------------|--------------|------------------------|---------|
| request (gap) | `frame` | S4 | "I'm missing this operand" |
| ground at S4 (atom) | `frame` | S4 | atom understood, in LTM |
| ground at S2 (canon) | `frame` | S2 | canon resolved, in LTM |
| ground at S3 (relation) | `frame` | S3 | relation resolved, in LTM |
| countersign (primary S1) | `frame` | S1 | primary reciprocal detected, ratified |

The stub never emits the legacy `ground` kind; `ground` is reserved for the future
S1-only reading. Until the global event-kind change lands, the trainer's
satisfaction logic keys on `proposal.significance` (the band), not on `kind`. The
`query` voice of each emitted event is the `value` passed to the `rationalise`
call that triggered the emission.

### Self-Cursored, Greedy Emission

The stub holds a cursor over its K-rows. On each `rationalise(value)` call it
**emits its next K-run** — the consecutive K-rows at the cursor, up to a run
boundary or end — as frame events in authored order, then stops. For the canonical
"Mary had a little lamb" table a K-run is a single row (the table is strictly
1:1 T/K alternating); the model handles longer K-runs identically. The stub does
not inspect `value` to decide what to emit; `value` only supplies the `query`
voice. This makes the stub bus-faithful: it owns its turn exactly as real Kalvin
will.

### Single Cascade

The trainer's opening submission (the primary `{MHALL:[SVO]}`) starts the run, and
the stub drives a single cascade to the end of its K-rows. It never returns to an
idle state mid-run expecting a fresh proactive prompt. The run ends on
**dual-exhaustion** (`@specs/dialogue-driven-training.md` §Training Loop): the
trainer-side cursor and the stub cursor both reach the end together under a
well-formed table.

### Atom Reuse

When a Canon whose operands are already grounded is reached in the K-row sequence,
the table emits the Canon's ground at S2 with no preceding requests. Atom reuse is
**table-prescribed**, not inferred — the table omits the request rows for already-
grounded operands (e.g. `C_SVO` grounds with zero new primes because S, V, O
grounded earlier in the cascade). The stub's `grounded` set is observational, not
behavioural: the stub never consults it to decide what to emit.

## Definition (the stub as a KAgent)

```
StubKAgent:
  adapter:   the adapter whose on_event it calls
  kturns:    the stub's ordered K-rows (the K subsequence of the shared table)
  cursor:    index of the next K-row to emit (0 at construction)
  grounded:  set of KLine signatures grounded or ratified so far (observational)
```

Construction:

```
StubKAgent(adapter, kturns)
```

The adapter is constructed first (`KAgentAdapter(bus)`), then the stub is
constructed with it, then `adapter.bind(stub)`. This mirrors real `KAgent` wiring.
`kturns` is the `actor == "K"` subsequence of the pre-decoded dialogue table;
construction asserts the cursor starts at 0.

### `rationalise(value: KValue) -> bool`

1. Record `value` as the current submission (it supplies the `query` voice).
2. If `cursor >= len(kturns)`: return `True` with no events. This is the normal
   end-of-run signal; whether the trainer-side was also exhausted is checked by
   the loop's dual-exhaustion gate, not by the stub.
3. Otherwise advance the cursor through the current K-run (consecutive K-rows up
   to a run boundary or end), and for each K-row emit a `frame` event with
   `query = value`, `proposal = <the K-row's KValue>`, in authored order. Add
   each grounded/countersigned kline's signature to `grounded`.
4. Return `True`.

For the canonical 1:1 table, step 3 emits exactly one K-row per call.

### `countersign(value: KValue) -> bool`

The stub does not self-ratify in response to trainer countersigns in the bootstrap
dialogue (ratification is by *submission*, not countersign, per the
absent-until-ratify decision). `countersign` is a no-op returning `True`. It
exists only to satisfy `_KAgentLike`.

### `save` / `codec`

No-ops returning `None` / a placeholder. The stub has no persistent model.

## Behavioural Rules

1. The stub is constructed with its K-rows and an adapter; it never constructs its
   own adapter or bus.
2. `rationalise` emits only what the cursor's next K-run prescribes — no
   inference, no expansion, no misfit classification, no matching of `value`.
3. Each `rationalise` advances the cursor by one K-run (one row in the canonical
   table); the cursor never moves backward.
4. Event significance is read from `proposal.significance`; `kind` is `frame` for
   all stub emissions (S1-on-countersign rows are `frame` at SIG_S1).
5. The stub drives a single cascade per run, from its opening emission to the end
   of its K-rows. It does not require proactive re-prompts mid-run.
6. Atom reuse is table-prescribed: the K-row sequence omits requests for already-
   grounded operands.
7. The stub does not detect, validate, or recover from divergence. Whether the
   stub emitted *the right* K (matching the table the trainer-side is validating
   against) is checked by the loop's two-sided Model A validation, not by the
   stub.

## Divergences

The stub **cannot diverge** from the table: it emits its own K-rows from its own
cursor, so by construction it emits exactly what its view of the table says. There
is no "no row matches → silent return" failure mode.

The concerns the previous trigger-keyed model called "divergences" are now
trainer-side, caught by the loop's two-sided validation
(`@specs/dialogue-driven-training.md` §Training Loop):

- **The stub emitted a K that the trainer-side did not expect** (stub/table view
  mismatch, or a truncated trainer table) — caught by the K-side of Model A.
- **The trainer submitted a T that produces no further K** (stub cursor exhausted
  before the trainer cursor) — caught by dual-exhaustion: a non-empty trainer
  cursor at stub exhaustion fails loudly.

The stub is dumb by design; it cannot recover, and it does not need to.

## Out of Scope

- Any real rationalisation: model, cogitation, expand, misfit, significance
  computation. The stub has none of these.
- Multi-cascade lessons. The bootstrap table is single-cascade.
- Remote (WebSocket) use. The stub is embedded, like the test `EventBus` adapter.
- Persistence. No save/load.
- Multi-row K-run pacing (which `rationalise` call triggers a multi-row K-run
  emission, and how run boundaries are detected from K-rows alone). Moot for the
  canonical 1:1 table; deferred until a non-1:1 table is authored
  (`@specs/dialogue-driven-training.md` defers the same question).
- Reconciliation of declared vs reported significance on divergence — trainer-side
  (`@specs/dialogue-driven-training.md` §Divergence).
- The global event-kind change (`ground`/`frame` → significance). The stub is
  written to be compatible with either, by keying on significance.

## Test Matrix

| ID  | Criterion | Origin ref |
|-----|-----------|------------|
| ST-1 | `StubKAgent` satisfies `_KAgentLike` (rationalise, countersign, save, codec) | §Definition |
| ST-2 | `rationalise` emits a K-run's request rows as `frame` events at SIG_S4 | §rationalise |
| ST-3 | `rationalise` emits a K-run's ground rows at their structural band | §rationalise |
| ST-4 | `rationalise` emits a K-run's countersign rows at SIG_S1 | §rationalise |
| ST-5 | The stub is self-cursored: each `rationalise` advances the cursor by one K-run; the cursor never moves backward and never inspects `value` | §Self-Cursored, §Behavioural Rules |
| ST-6 | `rationalise` with the cursor at end returns True with no events (normal end-of-run signal) | §rationalise |
| ST-7 | Events are emitted in authored (cursor) order within a K-run | §rationalise |
| ST-8 | `countersign` is a no-op returning True | §countersign |
| ST-9 | The stub drives a single cascade from its opening emission to the end of its K-rows; no proactive re-prompts mid-run | §Single Cascade |
| ST-10 | Atom reuse is table-prescribed (the K-row sequence omits requests for already-grounded operands); `grounded` is observational, never consulted for emission | §Atom Reuse |
| ST-11 | Event significance is carried on `proposal.significance`, not on the event `kind` | §Event Kinds |
| ST-12 | The stub consumes the K-rows of the shared pre-decoded dialogue table (no separate response table; no trigger-matching) | §The Dialogue Table |
| ST-13 | The `query` voice of every emitted event is the `value` passed to the triggering `rationalise` | §Event Kinds, §rationalise |
