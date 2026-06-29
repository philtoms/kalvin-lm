# Plan: Table-Driven Stub KAgent

**Parent:** [`plans/implement-kalvin.md`](../implement-kalvin.md)
**Spec:** [`@specs/stub-kagent.md`](../../specs/stub-kagent.md) (primary)
**Estimate:** 1 day
**Depends on:** `KValue` (`@specs/kvalue.md`), `RationaliseEvent` (`@specs/agent.md`),
the harness `_KAgentLike` protocol (`@specs/harness-server.md` §Kalvin Adapter),
the band-representative significance constants (`@specs/model.md`).

---

## 0. Goal

Deliver the deterministic `StubKAgent` specified in `@specs/stub-kagent.md`: a
table-driven contract double for `KAgent` that lets the trainer's paced-loop and
satisfaction logic be developed and tested independently of real Kalvin
cogitation. The stub implements `_KAgentLike` and emits `RationaliseEvent`s from
an authored Response Table instead of rationalising.

This closes the last spec-without-a-plan gap: `@specs/stub-kagent.md` defines
`StubKAgent`, its Response Table, Behavioural Rules, Divergences, and the ST-1..ST-11
test matrix, but no plan or code existed.

## Non-goals (explicitly out of scope per spec §Out of Scope)

- Real rationalisation (model, cogitation, expand, misfit, significance computation).
- Multi-cascade lessons (the bootstrap table is single-cascade).
- Remote (WebSocket) use — the stub is embedded, like the test `EventBus` adapter.
  In particular, **no harness `__main__.py` participant-class wiring**: the stub is
  constructed directly in tests.
- Persistence (save/load). `save`/`codec` are no-ops that satisfy `_KAgentLike`.
- The global event-kind change (`ground`/`frame` → significance). The stub keys on
  `proposal.significance` and emits `kind="frame"` for everything, so it is
  forward-compatible with either event-kind regime.

## Spec references

- `@specs/stub-kagent.md` — the whole spec; test matrix ST-1..ST-11.
- `@specs/harness-server.md` §Kalvin Adapter — the `_KAgentLike` surface
  (`rationalise`, `countersign`, `save`, `codec`) and the two-phase
  `adapter = KAgentAdapter(bus)` → `stub = StubKAgent(adapter, ...)` →
  `adapter.bind(stub)` wiring (mirrors real `KAgent`).
- `@specs/agent.md` §Events — `RationaliseEvent(kind, query, proposal)`; query and
  proposal are `KValue`s, significance rides on the KValue (KE-3).
- `@specs/kvalue.md` §Equality and Hashing (KV-2) — trigger matching is by KLine
  equality (signature + nodes), ignoring significance.
- `@specs/model.md` §Significance — `SIG_S1`/`SIG_S2`/`SIG_S3`/`SIG_S4`.

## Design decisions

These resolve the few points where the spec is descriptive rather than prescriptive.
Each is faithful to the spec's intent and called out here so it can be challenged.

1. **Location: `src/kalvin/stub_kagent.py`.** The stub is a `KAgent` contract double
   ("the stub as a KAgent", §Definition) and a peer of `src/kalvin/agent.py`. It lives
   in core `kalvin`, **not** under `training/`, because it must be importable without
   dragging in the harness and because the dependency direction is `kalvin ← training`.
   The stub therefore defines a **local** `_AdapterCallback` Protocol (`on_event`) and
   never imports `training.harness.adapter` (mirroring how `KAgent` defines its own
   adapter Protocol in `agent.py`).

2. **No `model` attribute.** The spec: "It is a contract double, not a Kalvin: it has
   no model." The adapter's `_handle_submit` guards STM pre-registration with
   `hasattr(self._kagent, "model")`, so a stub with no `model` attribute is correctly
   skipped. We therefore deliberately omit `model`.

3. **`cogitate_drain(timeout) -> True` is provided even though it is not in
   `_KAgentLike`.** The adapter's `drain()` and `_handle_drain` call
   `kagent.cogitate_drain(...)` **unconditionally** (no `hasattr` guard). The stub
   resolves everything synchronously inside `rationalise`, so there is never pending
   work; `cogitate_drain` is a no-op returning `True`. This makes the stub a drop-in
   for any code path that drains the cogitator. (ST-1 covers `_KAgentLike` proper;
   `cogitate_drain` is documented adapter-compat, not a protocol member.)

4. **`"initial"` trigger fires on the first `rationalise()` call.** The spec lists
   `"initial"` as the trigger for the first row but does not spell out the matching
   rule. Interpretation: a row whose trigger is the sentinel string `"initial"` fires
   on the **first** `rationalise(value)` call, **before** kline matching and consuming
   the row. Subsequent calls fall through to kline matching. This realises §Single
   Cascade ("the trainer's first submission … triggers a single cascade") without the
   author having to repeat the primary kline as a concrete trigger. Guidance (documented
   in the module): do **not** also give the first submission a concrete-trigger row —
   one row per submission, and the initial row owns the first call.

5. **One row per submission.** A single `rationalise()` call fires **at most one** row
   (either the initial row on the first call, or the kline-matched row otherwise). It
   never fires two rows in one call. The cascade advances one submission at a time.

6. **`grounded` is observational, not behavioural.** The spec lists `grounded` (set of
   grounded KLine signatures) as stub state and §Atom Reuse says reuse is
   **table-prescribed, not inferred** (ST-10). The stub therefore maintains `grounded`
   (each emitted ground adds the proposal's kline signature) but **never consults it**
   to decide what to emit. It exists for inspection/debugging and to mirror the KAgent
   notion; omitting the consult is what makes ST-10 hold.

7. **Trigger index keyed by `KValue`.** `pending_rows: dict[KValue, ResponseRow]`.
   Because `KValue.__hash__`/`__eq__` are kline-only (KV-2), `pending_rows.pop(value)`
   matches a submitted `KValue` against a row whose trigger is any `KValue` of the same
   kline, regardless of the significance either side carries. This is exactly the
   KV-2 match the spec requires.

## Module layout (target)

```
src/kalvin/stub_kagent.py   # ResponseRow, StubKAgent, _AdapterCallback Protocol
tests/test_stub_kagent.py   # ST-1..ST-11
```

No other files change. The stub is not wired into `training/harness/__main__.py`
(remote use is out of scope).

## Interface

```python
# src/kalvin/stub_kagent.py

class _AdapterCallback(Protocol):
    def on_event(self, event: RationaliseEvent) -> None: ...

@dataclass(frozen=True)
class ResponseRow:
    trigger: KValue | str        # KValue, or the sentinel "initial"
    requests: tuple[KValue, ...]      # each at SIG_S4
    grounds: tuple[KValue, ...]       # each at its structural band (S2/S3/S4)
    countersigns: tuple[KValue, ...]  # each at SIG_S1

class StubKAgent:
    def __init__(self, adapter: _AdapterCallback, rows: Sequence[ResponseRow]): ...
    # _KAgentLike
    def rationalise(self, value: KValue) -> bool: ...
    def countersign(self, value: KValue) -> bool: ...   # no-op -> True
    def save(self, path, format=None) -> None: ...      # no-op
    def codec(self) -> None: ...                        # placeholder -> None
    # adapter-compat (not in _KAgentLike; see decision 3)
    def cogitate_drain(self, timeout: float | None = None) -> bool: ...  # -> True
    # inspection
    @property
    def fired(self) -> frozenset[KValue]: ...
    @property
    def grounded(self) -> frozenset[int]: ...   # signatures
```

## Implementation tasks

### Task 1: `ResponseRow` + `StubKAgent` (`src/kalvin/stub_kagent.py`)

- Module docstring naming the spec, the non-goals, and design decisions 2/4/6.
- `_AdapterCallback` Protocol (`on_event`).
- `ResponseRow` frozen dataclass: `trigger`, `requests`, `grounds`, `countersigns`
  (tuples). Document the band each list is authored at.
- `StubKAgent.__init__(adapter, rows)`:
  - Store `self._adapter`.
  - Validate: at most one row has `trigger == "initial"` (raise `ValueError`
    otherwise). Duplicate concrete triggers are permitted to collapse by KV-2 (last
    wins) — document this; the canonical table has none.
  - Split rows into `self._initial_row` (the `"initial"` row, or `None`) and
    `self._pending_rows: dict[KValue, ResponseRow]` keyed by the trigger `KValue`.
  - `self._fired: list[KValue]` (in fire order, for inspection) and
    `self._grounded: set[int]` (signatures).
  - `self._first_call = True`.
- `_emit(row, query)`: for each `req` in `row.requests` →
  `on_event(RationaliseEvent("frame", query, req))`; then each `g` in `row.grounds` →
  `on_event(RationaliseEvent("frame", query, g))` and add `g.kline.signature` to
  `grounded`; then each `cs` in `row.countersigns` →
  `on_event(RationaliseEvent("frame", query, cs))` and add `cs.kline.signature` to
  `grounded` (a countersign is grounded-by-ratification). This order realises ST-7.
- `rationalise(value)`:
  1. If `self._first_call` and `self._initial_row is not None`: emit it with
     `query=value`, drop it, set `_first_call=False`, return `True`.
  2. Set `self._first_call=False` (even with no initial row).
  3. `row = self._pending_rows.pop(value, None)` (KV-2 structural key). If `None`,
     return `True` with no events (ST-6). Else `_emit(row, value)`, append `value` to
     `_fired`, return `True` (ST-2/3/4/5/7).
- `countersign(value)` → `True` (ST-8).
- `save(path, format=None)` → `None`; `codec()` → `None`.
- `cogitate_drain(timeout=None)` → `True`.
- `fired` / `grounded` read-only properties.

### Task 2: tests (`tests/test_stub_kagent.py`)

A tiny `_RecordingAdapter` (holds a list of received `RationaliseEvent`) stands in for
`KAgentAdapter.on_event` — the stub only needs an `on_event` callback, so this avoids
spinning up a real bus and keeps the tests unit-level. Helpers build `KValue`s at the
band constants directly (`SIG_S1`/`SIG_S2`/`SIG_S3`/`SIG_S4`).

| ID   | Test (function)                                                                                |
|------|------------------------------------------------------------------------------------------------|
| ST-1 | `test_stub_satisfies_kagent_like` — rationalise/countersign/save/codec present & return specs  |
| ST-2 | `test_requests_emitted_as_frame_at_s4`                                                         |
| ST-3 | `test_grounds_emitted_at_structural_band` (one S2, one S3, one S4 ground)                      |
| ST-4 | `test_countersigns_emitted_at_s1`                                                              |
| ST-5 | `test_row_fires_at_most_once`                                                                  |
| ST-6 | `test_no_match_returns_true_silent`                                                            |
| ST-7 | `test_emission_order_requests_then_grounds_then_countersigns`                                  |
| ST-8 | `test_countersign_is_noop_true`                                                                |
| ST-9 | `test_single_cascade_initial_row_then_chain` — initial row + two matched rows fire in sequence |
| ST-10| `test_atom_reuse_table_prescribed` — a canon row with grounded operands emits zero requests   |
| ST-11| `test_significance_carried_on_proposal_not_event` — `event.kind=="frame"`, sig on proposal    |
| —    | `test_initial_row_fires_on_first_call_only` (§Definition)                                      |
| —    | `test_cogitate_drain_is_noop_true` (adapter-compat, decision 3)                                |
| —    | `test_two_initial_rows_rejected` (validation)                                                  |

## Out of scope (restated)

Anything in the spec's §Out of Scope: real rationalisation, multi-cascade lessons,
remote/WebSocket use (no `__main__.py` wiring), persistence, divergence recovery
(that is the trainer's concern, `@specs/trainer-satisfaction.md` §Stalls), and the
global event-kind change.

## Status

Plan complete; implementation pending.
