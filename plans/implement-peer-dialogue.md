# Implement Peer Dialogue — Plan

## Spec References

- `@specs/peer-dialogue.md` — WHAT this plan implements (all PDT-\* criteria).
- `@specs/dialogue-driven-training.md` — `DialogueTable`, `DecodedTurn`,
  `RationaliseEvent`, `Actor`, the synchronous `run` (unchanged by this work).
- `@docs/adr/0002-peer-runner-is-a-bus-subscriber.md` — why the peer runner is
  a `MessageBus` subscriber (supersedes ADR-0001).

## Implementation Tasks

All new code lives in `src/training/dialogue/`. No change to the synchronous
`run`, `Actor`/`respond`, `TableTrainer`, `TableTrainee`, `ActorDivergence`,
or `RunResult`.

### Phase 1 — Table regime + decode-time validation

**1.1 Dialogue-table peer section.** Extend the dialogue-table loader to read
an optional `peer` section. The section's presence selects peer mode (there
is no top-level `mode` field); all peer modifiers live in it. The single
modifier today is `on_divergence` (default `"fail"`). Unknown keys inside
`peer` are a decode error. The loader resolves the section into a
`PeerConfig` carried on `DialogueTable.peer` (None ⇒ ordered); the runner
never reads the raw table. → PDT-1.

**1.2 Decode-time peer validations.** When the table carries a `peer` section,
the decode path validates: (a) the opening (`decoded[0]`) is a `T` row; (b) the
opening and closing (`decoded[-1]`) are content-distinct — i.e.
`(role, kline, significance)` differ. Violation raises a malformed-table
error before any run begins. → PDT-3, PDT-4.

### Phase 2 — The peer types (unchanged by the bus decision)

**2.1 `PeerDivergence`.** Exception carrying `(role: str, emitted: KValue,
unconsumed: tuple[DecodedTurn])`. Distinct from `ActorDivergence`; no `cursor`
field. → PDT-14.

**2.2 `PeerRunResult`.** Dataclass: `events: list[RationaliseEvent]`
(arrival-ordered), `complete: bool`, `covered: bool`,
`unmatched: list[RationaliseEvent]`, `uncovered: list[DecodedTurn]`. Distinct
from `RunResult`. → PDT-15.

### Phase 3 — The runner as a MessageBus subscriber (ADR-0002)

The peer runner is a coverage-tracking wildcard subscriber over a
`MessageBus`, plus a thin driver. It is **not** a standalone sink — the bus is
the sink and the relay. The existing standalone-sink `PeerRunner` (from earlier
commits on this branch) is replaced.

**3.1 `BusSink` + `accept` on `Actor`.** Define a narrow `BusSink` protocol
(`send(Message) -> None`) that `MessageBus` satisfies. Add
`Actor.accept(event: RationaliseEvent | None, sink: BusSink) -> None` —
fire-and-forget; the actor replies zero-or-many via `sink.send(Message(
role=<other>, action="accept", message=<RationaliseEvent>))`. `event=None`
signals "you open". The ordered `respond` is unchanged. → PDT-17.

**3.2 `PeerRunner` as a bus subscriber.** Construction builds a `MessageBus`,
subscribes the runner (as a wildcard handler) for coverage, subscribes the two
actors to their own roles (their `accept` is the handler). Holds coverage
bookkeeping only: the table's fixed distinct middle content set, a growing
covered subset, a closing reference, a `closing_seen` flag, the idle deadline.
No actor-coupling state. → PDT-5, PDT-6.

**3.3 The wildcard handler (coverage).** On each observed emission:
1. Append to `events` (arrival order). → PDT-15.
2. If the emission equals the closing content — set `closing_seen`, call
   `bus.stop()`. → PDT-10, PDT-13.
3. Else if present in distinct middle contents — add to covered subset
   (idempotent). → PDT-7, PDT-8.
4. Else — divergence: `on_divergence="fail"` raises `PeerDivergence(role,
   emitted, unconsumed=<uncovered same-role>)`; `"accept"` appends to
   `unmatched`. → PDT-9.

Anticipation/interjection require no special path — matching is content-only
and order-agnostic. → PDT-11, PDT-12.

**3.4 The driver.** `run(trainer, trainee)` seeds the opening by `bus.send(
Message(role="T", action="accept", message=None))`, then runs `bus.run()` on a
thread. The trainer's `accept` handler replies (addressed to "K"), the bus
delivers to the trainee's `accept`, which replies to "T", and so on — the bus
relays; the runner only observes. Terminates on `closing_seen` (subscriber
calls `bus.stop()`). Idle timeout = `queue.get(timeout=idle_timeout)`; on
silence-with-no-closing, the run stops with `complete = False`. → PDT-18,
PDT-19.

**3.5 `run_peer` constructor.** `run_peer(decoded, trainer, trainee,
*, on_divergence="fail", idle_timeout=...) -> PeerRunner`. Builds the bus,
wires subscribers, returns the runner; the caller calls `runner.run()`.

### Phase 4 — Wiring + script

**4.1 `__init__` exports.** Export `run_peer`, `PeerRunner`, `PeerDivergence`,
`PeerRunResult` from `training.dialogue`.

**4.2 `scripts/dialogue_run.py` regime dispatch.** The script reads
`table.is_peer` and dispatches — a peer table constructs a `PeerRunner` (with
`TableTrainer`/`TableTrainee` given `accept` implementations) and calls
`runner.run()`; an ordered table drives the synchronous `run`. No CLI flags
for the regime or peer modifiers (table-driven). → PDT-5.

The default `TableTrainer`/`TableTrainee` gain `accept` as a thin adapter over
their existing cursor logic (they emit their next row(s) via the sink). Their
`respond` is unchanged for the ordered regime.

## Test Mapping Table

Tests are flat under `tests/` (matching the project convention), not
`tests/dialogue/`. Phase 1 tests exist (`tests/test_peer_dialogue_decoder.py`);
the sink-contract tests from earlier commits will be reworked against the
bus-subscriber model.

| Spec ID | Test file                                         | Status   |
| ------- | ------------------------------------------------- | -------- |
| PDT-1   | `tests/test_peer_dialogue_decoder.py`             | done     |
| PDT-2   | `tests/test_peer_runner.py`                       | rework   |
| PDT-3   | `tests/test_peer_dialogue_decoder.py`             | done     |
| PDT-4   | `tests/test_peer_dialogue_decoder.py`             | done     |
| PDT-5   | `tests/test_peer_runner.py`                       | rework   |
| PDT-6   | `tests/test_peer_runner.py`                       | rework   |
| PDT-7   | `tests/test_peer_runner.py`                       | rework   |
| PDT-8   | `tests/test_peer_runner.py`                       | rework   |
| PDT-9   | `tests/test_peer_runner.py`                       | rework   |
| PDT-10  | `tests/test_peer_runner.py`                       | rework   |
| PDT-11  | `tests/test_peer_runner.py`                       | rework   |
| PDT-12  | `tests/test_peer_runner.py`                       | rework   |
| PDT-13  | `tests/test_peer_runner.py`                       | rework   |
| PDT-14  | `tests/test_peer_runner.py`                       | rework   |
| PDT-15  | `tests/test_peer_runner.py`                       | rework   |
| PDT-16  | `tests/test_dialogue_runner.py`                   | done     |
| PDT-17  | `tests/test_peer_runner.py`                       | todo     |
| PDT-18  | `tests/test_peer_runner.py`                       | todo     |
| PDT-19  | `tests/test_peer_runner.py`                       | todo     |

## Design Decisions

- **Bus subscriber, not a standalone sink.** See
  `@docs/adr/0002-peer-runner-is-a-bus-subscriber.md` (supersedes ADR-0001).
  The harness `MessageBus` is the sink and relay; the runner is a coverage-
  tracking wildcard subscriber. This delivers true non-blocking `accept` for
  free (actors reply from any thread via the thread-safe bus) without
  introducing `asyncio` or duplicating the bus. The synchronous ordered `run`
  is untouched and stays bus-agnostic.
- **Fire-and-forget `accept`, zero-or-many replies.** The actor's autonomy
  (when/whether/how-many to reply) is what makes the dialogue messy and real,
  and what lets T and K be out of sync. No synchronised alternation.
- **Routing by bus addressing.** Each actor addresses replies to the other
  role; the bus delivers; the runner never reroutes. Interjection is just
  "sending to the other role unsolicited".
- **Coverage is efficiency, not a count.** Duplicate table rows collapse to
  one distinct content; coverage is idempotent (re-emitting covered content is
  not divergence). Completion is closing-seen alone; an idle timeout ends a
  stalled run as incomplete (non-fatal).
- **Separate `PeerDivergence` / `PeerRunResult`.** The peer regime's data
  (covered subset, arrival-ordered events) has no cursor; reusing the
  synchronous types would force sometimes-`None` cursor fields.

## Status

- Spec: `@specs/peer-dialogue.md` — written (PDT-1..PDT-19).
- ADR: `@docs/adr/0002-peer-runner-is-a-bus-subscriber.md` — written
  (supersedes 0001).
- Plan: this document — written.
- Code: Phase 1 (table/decode) done on this branch; Phases 2–4 to be
  reworked against the bus-subscriber model (the existing standalone-sink
  `PeerRunner` is replaced).
