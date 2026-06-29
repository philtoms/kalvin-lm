# Table-Driven Stub KAgent — Specification

## Overview

A deterministic stand-in for Kalvin used to develop the trainer's paced-loop and
satisfaction logic independently of real-Kalvin cogitation. It implements the same
harness protocol as `KAgent` but, instead of rationalising, **returns table-
prescribed responses**: on each submission it emits the table's next-pending
proposals (`request` events) and grounds prior submissions at their structural band
(`ground` events).

The stub exists for bootstrapping. It is a contract double, not a Kalvin: it has no
model, no cogitation, no expansion, no misfit reshape. A training dialogue authored
in a response table drives it deterministically. When real Kalvin is reconciled to
the same table (next grill), the stub is retired.

This spec depends on `@specs/harness-server.md` (KAgentAdapter, KAgentLike
protocol, events), `@specs/agent.md` (RationaliseEvent shape), and `@specs/kscript.md`
(compiled entry band tags). It does **not** depend on cogitation, expand, or model
internals — those are deliberately absent.

## Dependencies

- `@specs/harness-server.md` — `_KAgentLike` protocol (`rationalise`, `countersign`,
  `save`, `codec`), `on_event` callback, the bus `submit`/`countersign`/`rationalise`
  actions.
- `@specs/agent.md` — `RationaliseEvent(kind, query, proposal)` shape; significance
  on the KValue, not the event.
- `@specs/kscript.md` — compiled-entry declared bands (`op` → S1/S2/S3/S4).

## Definitions

### The Response Table

A declarative description of the stub's behaviour, authored per-lesson. The table is
the **idealised dialogue** the trainer must reproduce (see the canonical example in
`@specs/trainer-satisfaction.md` §Canonical Example). It is the source of truth for
what the stub emits, and — critically — for what the trainer's satisfaction logic
should expect to receive.

Each row is:

```
Row:
  trigger:    the trainer submission that this row responds to
              (the KValue submitted by the trainer), OR "initial" for the first row.
  requests:   list of KValues the stub proposes back at S4
              ("I'm missing X") — emitted as `frame` events with proposal at SIG_S4.
  grounds:    list of KValues the stub grounds at their structural band
              — emitted as `frame` events at band, OR `ground` events at S1.
  countersigns: list of KValues the stub self-ratifies to S1
              — emitted as `frame` events at SIG_S1 (the primary's completion).
```

The stub matches an inbound submission against rows by KLine equality (signature +
nodes, KV-2). On match it emits the row's `requests`, `grounds`, and `countersigns`
in order, then consumes the row (each row fires once).

### Event Kinds the Stub Emits

Per the mental note (not yet spec'd globally), event kinds collapse toward
significance. The stub emits:

| Stub intent | Event `kind` | `proposal.significance` | Meaning |
|-------------|--------------|------------------------|---------|
| request (gap) | `frame` | S4 | "I'm missing this operand" |
| ground at S4 (atom) | `frame` | S4 | atom understood, in LTM |
| ground at S2 (canon) | `frame` | S2 | canon resolved, in LTM |
| ground at S3 (relation) | `frame` | S3 | relation resolved, in LTM |
| countersign (primary S1) | `frame` | S1 | primary reciprocal detected, ratified |

The stub never emits the legacy `ground` kind in its `frame`-band rows; `ground` is
reserved for the future S1-only reading. Until the global event-kind change lands, the
trainer's satisfaction logic keys on `proposal.significance` (the band), not on the
`kind` field. This keeps the stub forward-compatible.

### Single Cascade

The trainer's first submission (the primary `{MHALL:[SVO]}`) triggers a single cascade.
The stub drives that cascade to completion: it never returns to an idle state mid-lesson
expecting a fresh proactive prompt. Any authored row whose `trigger` is not reached
within the cascade is never emitted and never grounds — it is out of scope for the stub
and a signal that the trainer's prompting diverged from the table.

### Atom Reuse

When the stub is asked to ground a Canon whose operands are already grounded, it emits
the Canon's ground at S2 immediately, without re-requesting the operands. The table
prescribes this (e.g. `C_SVO` grounds with zero new primes because S, V, O grounded
earlier in the cascade).

## Definition (the stub as a KAgent)

```
StubKAgent:
  adapter:        the adapter whose on_event it calls
  table:          the Response Table (rows)
  pending_rows:   rows not yet fired, indexed by trigger KLine
  fired:          set of rows already fired
  grounded:       set of KLine signatures understood and in (logical) LTM
```

Construction:

```
StubKAgent(adapter, table)
```

The adapter is constructed first (`KAgentAdapter(bus)`), then the stub is constructed
with it, then `adapter.bind(stub)`. This mirrors real `KAgent` wiring.

### `rationalise(value: KValue) -> bool`

1. Record `value` as the current submission.
2. Look up `pending_rows` by `value.kline` (KLine equality).
3. If no row matches, return `True` with no events (the stub has nothing prescribed
   for this submission — a table/trainer divergence; see §Divergences).
4. If a row matches, mark it fired and emit, in order:
   - each `requests` entry: a `frame` event with `query = value`,
     `proposal = <request kline at SIG_S4>`.
   - each `grounds` entry: a `frame` event at its band.
   - each `countersigns` entry: a `frame` event at SIG_S1.
5. Return `True`.

### `countersign(value: KValue) -> bool`

The stub does not self-ratify in response to trainer countersigns in the bootstrap
dialogue (the trainer's ratification is by *submission*, not countersign, per the
absent-until-ratify decision). `countersign` is therefore a no-op that returns `True`.
It exists only to satisfy `_KAgentLike`.

### `save` / `codec`

No-ops returning `None` / a placeholder. The stub has no persistent model.

## Behavioural Rules

1. The stub is constructed with a Response Table and an adapter; it never constructs
   its own adapter or bus.
2. `rationalise` emits only what the matching table row prescribes — no inference, no
   expansion, no misfit classification.
3. A row fires at most once; a second submission of the same trigger KLine finds no
   pending row and emits nothing.
4. Event significance is read from `proposal.significance`; the `kind` field is
   `frame` for all stub emissions (the S1-on-countersign rows are `frame` at SIG_S1).
5. The stub drives a single cascade per lesson: it emits the whole chain of
   requests/grounds/countersigns implied by the matching rows, in authored order.
6. Atom reuse is table-prescribed, not inferred: if a Canon's operands are grounded,
   the table's row for that Canon omits `requests` for them.

## Divergences

A divergence is any trainer submission that matches no pending row. Two cases:

- **Table incomplete** — the trainer submitted a kline the table didn't author a
  response for. The stub returns `True` silently; the trainer's satisfaction logic
  sees no grounding event for that submission and must decide (escalate or wait).
- **Trainer out of order** — the trainer submitted a kline whose row exists but whose
  trigger the table placed elsewhere. Same observable: no match, silent return.

Divergence handling is the **trainer's** concern (`@specs/trainer-satisfaction.md`
§Stalls), not the stub's. The stub is dumb by design; it cannot recover.

## Out of Scope

- Any real rationalisation: model, cogitation, expand, misfit, significance
  computation. The stub has none of these.
- Multi-cascade lessons. The bootstrap table is single-cascade.
- Remote (WebSocket) use. The stub is embedded, like the test `EventBus` adapter.
- Persistence. No save/load.
- Reconciliation of declared vs reported on divergence — that is trainer-side.
- The global event-kind change (`ground`/`frame` → significance). The stub is written
  to be compatible with either, by keying on significance.

## Test Matrix

| ID  | Criterion | Origin ref |
|-----|-----------|------------|
| ST-1 | `StubKAgent` satisfies `_KAgentLike` (rationalise, countersign, save, codec) | §Definition |
| ST-2 | `rationalise` emits a row's `requests` as `frame` events at SIG_S4 | §rationalise |
| ST-3 | `rationalise` emits a row's `grounds` at their structural band | §rationalise |
| ST-4 | `rationalise` emits a row's `countersigns` at SIG_S1 | §rationalise |
| ST-5 | A row fires at most once; a repeat trigger emits nothing | §Behavioural Rules |
| ST-6 | A submission matching no pending row returns True with no events | §Divergences |
| ST-7 | Events are emitted in authored order (requests, then grounds, then countersigns) | §rationalise |
| ST-8 | `countersign` is a no-op returning True | §countersign |
| ST-9 | The stub drives a single cascade; it does not require proactive re-prompts mid-lesson | §Single Cascade |
| ST-10 | Atom reuse is table-prescribed (no requests for already-grounded operands) | §Atom Reuse |
| ST-11 | Event significance is carried on `proposal.significance`, not on the event | §Event Kinds |
