# Implement Peer Dialogue — Plan

## Spec References

- `@specs/peer-dialogue.md` — WHAT this plan implements (all PDT-\* criteria).
- `@specs/dialogue-driven-training.md` — `DialogueTable`, `DecodedTurn`,
  `RationaliseEvent`, `Actor`, the synchronous `run` (unchanged by this work).
- `@docs/adr/0001-peer-runner-is-a-sink.md` — why `run_peer` is a sink.

## Implementation Tasks

All new code lives in `src/training/dialogue/`. No change to the synchronous
`run`, `Actor`/`respond`, `TableTrainer`, `TableTrainee`, `ActorDivergence`,
or `RunResult`.

### Phase 1 — Table regime + decode-time validation

**1.1 Dialogue-table regime fields.** Extend the dialogue-table loader to read
the regime declaration (peer mode) and the divergence policy
(`on_divergence`, default `"fail"`). These are authoring knobs on the table;
the loader resolves them into `run_peer` inputs. The runner never reads the
raw table. → PDT-1.

**1.2 Decode-time peer validations.** When the table declares peer mode, the
decode path validates: (a) the opening (`decoded[0]`) is a `T` row; (b) the
opening and closing (`decoded[-1]`) are content-distinct — i.e.
`(role, kline, significance)` differ. Violation raises a malformed-table
error before any run begins. → PDT-3, PDT-4.

### Phase 2 — The sink types

**2.1 `PeerDivergence`.** A new exception type in `src/training/dialogue/runner.py`
(or a sibling module — see §Open location question), carrying `(role: str,
emitted: KValue, unconsumed: tuple[DecodedTurn])`. Distinct from
`ActorDivergence`; no `cursor` field. → PDT-14.

**2.2 `PeerRunResult`.** A new dataclass:
`events: list[RationaliseEvent]` (arrival-ordered), `complete: bool`,
`covered: bool`, `unmatched: list[RationaliseEvent]`, `uncovered: list[DecodedTurn]`.
Distinct from `RunResult`. → PDT-15.

### Phase 3 — The runner

**3.1 `PeerRunner` sink.** A class (not a function-loop) holding coverage
bookkeeping only:
- the unconsumed distinct middle set, keyed by `(role, kline, significance)`,
  built once at construction from `decoded[1:-1]`;
- a `closing` reference (`decoded[-1]`) and a `closing_seen` flag.

API: `receive(event) -> None`, property `complete`, property `result`.
`receive` is the sole entry point once the run has begun. → PDT-5, PDT-6.

**3.2 Matching in `receive`.** On each event:
1. If the event equals the closing content — set `closing_seen`, consume the
   closing. → PDT-10.
2. Else look up the event's `(role, kline, significance)` in the unconsumed
   same-role middle set:
   - present → remove the distinct content (duplicates collapsed by set
     membership). → PDT-7, PDT-8.
   - absent → divergence. `on_divergence="fail"` raises `PeerDivergence(role,
     emitted=event.proposal, unconsumed=<frozen unconsumed same-role rows>)`;
     `"accept"` appends `event` to `result.unmatched`. → PDT-9.
3. Append `event` to `result.events` in arrival order. → PDT-15.

Anticipation requires no special code path: matching is content-only and
order-agnostic, so an "ahead-of-causal" emission matches whatever unconsumed
same-role row its content equals. → PDT-11, PDT-12.

**3.3 Completion.** `complete` is a property: `closing_seen AND (unconsumed
middle set empty)`. `covered` is `unconsumed middle set empty` (without the
closing conjunct) — a diagnostic, not a terminal condition. On a query to
`result` before completion, `uncovered` is populated from the remaining
unconsumed middle set. → PDT-13, PDT-15.

**3.4 `run_peer` constructor function.** A thin constructor:
`run_peer(decoded, *, on_divergence="fail") -> PeerRunner`. Validates the
opening/closing invariants a second time defensively, builds the coverage
set, returns the `PeerRunner`. The caller is responsible for delivering the
opening to the trainee before pushing emissions; the runner performs no
outbound delivery. → PDT-5 (Out of Scope: opening delivery).

### Phase 4 — Wiring + script

**4.1 `__init__` exports.** Export `run_peer`, `PeerRunner`, `PeerDivergence`,
`PeerRunResult` from `training.dialogue`.

**4.2 `scripts/dialogue_run.py` flag.** Add a `--peer` flag (and
`--on-divergence {fail,accept}`) that, given a peer-mode table, constructs a
`PeerRunner` and feeds it from the default actors' emissions. Because the
default table-reading actors are pull-shaped, the script acts as the
caller-bridge: it seeds the trainee with the opening, then pulls each actor
in turn and pushes emissions into the runner until `complete`. This bridge is
test/script wiring only — it does not belong in the runner.

## Test Mapping Table

| Spec ID | Test file                                         | Status |
| ------- | ------------------------------------------------- | ------ |
| PDT-1   | `tests/dialogue/test_peer_table_regime.py`        | todo   |
| PDT-2   | `tests/dialogue/test_peer_anticipation.py`        | todo   |
| PDT-3   | `tests/dialogue/test_peer_zones.py`               | todo   |
| PDT-4   | `tests/dialogue/test_peer_decode_validations.py`  | todo   |
| PDT-5   | `tests/dialogue/test_peer_sink_contract.py`       | todo   |
| PDT-6   | `tests/dialogue/test_peer_sink_contract.py`       | todo   |
| PDT-7   | `tests/dialogue/test_peer_matching.py`            | todo   |
| PDT-8   | `tests/dialogue/test_peer_duplicate_collapse.py`  | todo   |
| PDT-9   | `tests/dialogue/test_peer_divergence.py`          | todo   |
| PDT-10  | `tests/dialogue/test_peer_closing.py`             | todo   |
| PDT-11  | `tests/dialogue/test_peer_anticipation.py`        | todo   |
| PDT-12  | `tests/dialogue/test_peer_anticipation.py`        | todo   |
| PDT-13  | `tests/dialogue/test_peer_completion.py`          | todo   |
| PDT-14  | `tests/dialogue/test_peer_divergence.py`          | todo   |
| PDT-15  | `tests/dialogue/test_peer_result.py`              | todo   |
| PDT-16  | `tests/dialogue/test_sync_run_unchanged.py`       | todo   |

## Design Decisions

- **Sink, not driver.** See `@docs/adr/0001-peer-runner-is-a-sink.md`. The
  synchronous `run` is left intact; `run_peer` is a sibling regime with its
  own contract and types.
- **Opening delivery is the caller's job.** A pure sink cannot perform the one
  asymmetric priming act. The script/test wiring layer seeds the trainee.
- **Coverage bookkeeping only.** The runner tracks the unconsumed distinct
  middle set and a closing-seen flag. It tracks nothing per-actor.
- **Duplicates collapse by set membership.** The unconsumed middle is held as
  a set keyed by `(role, kline, significance)`; consuming X removes it once,
  satisfying all duplicate X rows. Completion is over distinct contents.
- **Separate `PeerDivergence` / `PeerRunResult`.** The peer regime's data
  (unconsumed set, arrival-ordered events) has no cursor; reusing the
  synchronous types would force sometimes-`None` cursor fields.

## Open location question (resolve before Phase 2)

The synchronous runner lives in `src/training/dialogue/runner.py`. Two homes
for the peer code:
- **(a) Same file** — one `runner.py` holding both regimes, sharing the
  content-equality notion of a match. Maximises colocation of the two
  sibling regimes.
- **(b) `peer_runner.py` sibling** — a separate module. Keeps each regime's
  file focused; mirrors the type separation (`PeerDivergence`/`PeerRunResult`
  are already separate).

Lean **(b)**: the regimes are structurally different (sink vs loop), the types
are already separate, and a dedicated file signals "this is a different
artifact" to a reader. Confirm at implementation time.

## Status

- Spec: `@specs/peer-dialogue.md` — written.
- ADR: `@docs/adr/0001-peer-runner-is-a-sink.md` — written.
- Plan: this document — written.
- Code: not started.
