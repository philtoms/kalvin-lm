# Rationalise Trainer Significance (S4-Drop MVP) Implementation Plan

**Parent:** Kalvin consuming the trainer's declared significance during
rationalisation — the first piece of the two-way significance dialog. MVP
scope: Kalvin honours an S4 disagreement by dropping the query, which lets
it ratify a ratification and drop a recurring proposal.
**Status:** plan complete; implementation pending
**Spec refs:** `@specs/agent.md` §Rationalisation (new significance-gate
phase), `@specs/kvalue.md` §What a KValue is Not (slimmed — consumption moves
to agent.md), `@specs/harness-server.md` §Kalvin Adapter (new `rationalise`
action) + §Trainer (recurrence → declared-S4 behaviour)
**Depends on:** none new. Builds on the existing significance re-derivation
(`derive_significance`) and the band-representative values (`SIG_S4`).

## Problem

Rationalisation ignores the inbound KValue's declared significance. Both
`specs/agent.md §Rationalisation` and `specs/kvalue.md §What a KValue is Not`
mark consumption as deferred. Consequence: when Kalvin's candidate fan-out
emits the same reshaped proposal `P` from two candidates against one
expectation `E`, the trainer has no way to tell Kalvin "this proposal
doesn't work." Kalvin cogitates it indefinitely; only the trainer-side
`_reactive_rounds` budget catches the stall, and Kalvin never learns it is
spinning.

The MVP un-defers the smallest useful piece of consumption: Kalvin compares
its own **derived** significance to the trainer's **declared** significance,
and on a disagreement where the trainer declares S4, Kalvin drops the query
instead of re-cogitating it. The trainer produces that declared-S4 KValue on
**second-sighting recurrence** of a proposal.

This is *not* a "veto" primitive. It is Kalvin gauging the trainer's
confidence in a two-way significance dialog and choosing (MVP-only) to honour
an S4 disagreement by discarding. The vision doc (`docs/kalvin-vision.md`)
already states this WHY — significance as the trust-dialog (line 39), the
**Correct** posture "The proposal is denied. Kalvin has overreached"
(line 71), and "Kalvin rationalises every kline it receives in exactly the
same way" (line 71). The vision is ahead of the specs; this MVP brings the
specs into line rather than extending the vision.

## Spec References

- `@specs/agent.md` §Rationalisation — gains a new pre-ground significance
  gate (derived vs declared, S4-drop) with test-matrix entries.
- `@specs/kvalue.md` §What a KValue is Not — slimmed: rationalisation
  *does* consume the declared significance (see @agent spec); this spec
  owns only the exchange-unit invariant ("present and addressable").
- `@specs/harness-server.md` §Kalvin Adapter — new `rationalise` action;
  §Trainer — recurrence detection and the declared-S4 re-submission.
- `docs/kalvin-vision.md` — **unchanged**. Already states the WHY.

## Implementation Tasks

### Task 1 — `KAgent.rationalise` significance gate

**File:** `src/kalvin/agent.py` (`rationalise`, immediately after the Phase 1
signature `assert`, before the Phase 2 ground check).

**Why before the ground check:** a recurring proposal is already in Frame
(written by `on_expansion`), and `Model.grounded()` excludes only STM — so a
post-ground gate would be inert against its one real target.

**Spec ref:** `@specs/agent.md` §Rationalisation Phase 1b (new gate).

> **Implementation correction (discovered during Task 1):** the plan below
> originally specified `classify(...) == "S4"`. That is dead code:
> `classify()` collapses the S3|S4 boundary — `SIG_S4` (`0`) satisfies
> `0 >= s34(0)` and classifies as **S3**, so `classify()` can never return
> "S4". S4 is a sentinel detected by value (`== SIG_S4`), exactly as
> `normalise_significance` special-cases `raw_sig == 0`. The implemented gate
> is therefore value-based, not band-based:

```python
# NEW: significance-comparison gate (before ground check).
# MVP: when the sender declares S4 and Kalvin derives otherwise, drop.
from kalvin.expand import SIG_S4, derive_significance

derived_sig = derive_significance(kline, self._model, self._signifier)
if value.significance == SIG_S4 and derived_sig != SIG_S4:
    return True            # drop — no STM write, no event

# ... existing Phase 2 ground check onward unchanged ...
```

Three outcomes (value-based):
- `derived == SIG_S4` and `declared == SIG_S4` (agree, e.g. identity
  declared S4) → process normally.
- `declared == SIG_S4` and `derived != SIG_S4` → drop (MVP).
- `declared != SIG_S4` (any S1/S2/S3 disagreement or agreement) → process
  normally (MVP ignores non-S4 disagreements; deferred).

Notes:
- Drop = `return True` with **no** STM write and **no** event. The proposal
  is not evicted from Frame; it is simply not re-cogitated.
- `derive_significance` yields `SIG_S4` only for identity klines, so the drop
  branch is live only against a trainer-constructed declared-S4 over a
  non-identity kline (the reactor's recurrence path, Task 3). It is
  unreachable from the compiler's IDENTITY→S4 mapping (identity klines agree).
- **Test-audit expansion:** the audit covered more than the planned 73
  `test_agent.py` sites — the `SIG_S4` placeholder convention pervaded the
  suite. Hand-built KValues now declare their honest band via a `_kv(kline,
  model)` helper (`derive_significance`); compiled entries are passed through
  with their compiler-assigned significance. Files touched:
  `test_agent.py`, `test_countersign_resolution.py`, `test_encode_text.py`,
  `test_cogitator_drain.py`.

### Task 2 — `"rationalise"` adapter action

**File:** `src/training/harness/adapter.py` (`on_message` dispatch, ~line 204,
and a new `_handle_rationalise`).

**Why a new action:** `submit` recompiles (re-derives significance from
structure); `countersign` builds the reciprocal and forces `SIG_S1`. Neither
can deliver a trainer-constructed KValue with trainer-chosen significance
straight to `rationalise`. A new action is the honest primitive — "deliver
this KValue to rationalisation" — and is significance-agnostic for future
dialog use. (A direct reactor→KAgent handle would break the "participants
never communicate directly" glossary rule.)

**Spec ref:** `@specs/harness-server.md` §Kalvin Adapter.

```python
# dispatch arm
elif msg.action == "rationalise":
    self._handle_rationalise(msg)

def _handle_rationalise(self, msg: Message) -> None:
    """Deliver a trainer-constructed KValue straight to rationalisation.

    Reuses _materialise_kvalue (live KValue / wire dict / legacy KLine).
    No recompile, no reciprocal, no forced significance. The significance
    on the KValue is the sender's declared assessment.
    """
    if self._kagent is None:
        logger.error("No KAgent bound; cannot rationalise")
        return
    kvalue = _materialise_kvalue(msg.message)
    self._kagent.rationalise(kvalue)   # fire-and-forget; events via on_event
```

The action takes a live `KValue` (in-process) or a wire dict
`{"signature","nodes","significance"}`. `_materialise_kvalue` already
enforces `significance` is present on dicts (fail-loud), so a wire client
cannot accidentally send an S4-less payload.

### Task 3 — Reactor recurrence → declared-S4

**File:** `src/training/trainer/reactor.py` (`process_s2_s3`, `load_lesson`,
new `_seen_proposals` set).

**Spec ref:** `@specs/harness-server.md` §Trainer.

Recurrence is **intra-expectation fan-out**: one expectation `E` retrieves
candidates `Cᵢ`, `Cⱼ`; both expand to the same reshaped proposal `P`;
`on_expansion(P)` fires twice → two `"frame"` events → `process_s2_s3`
sees `P` twice. First sighting scaffolds; second sighting drops.

```python
# __init__: alongside _reactive_rounds
self._seen_proposals: set[EntryKey] = set()

# load_lesson: reset per lesson (recurrence is scoped to this lesson's
# expectations; a proposal that recurred in lesson 1 gets a fresh chance
# in lesson 2)
def load_lesson(self, entries):
    self._current_entries = entries
    self._reactive_rounds = 0
    self._seen_proposals = set()

# process_s2_s3
def process_s2_s3(self, event):
    if self._auto_countersign(event.proposal):
        return True
    key = _entry_key(event.proposal)
    if key in self._seen_proposals:                 # recurrence (2nd sighting)
        self._reactive_rounds += 1                  # count toward budget
        self._handle_reactive_budget_guard(event)   # escalate at cliff (see below)
        self._bus.send(Message(
            role=TRAINEE_ROLE,
            action="rationalise",
            message=KValue(event.proposal.kline, SIG_S4),
            sender=self._role,
        ))
        return True                                 # handled; no supervisor
    self._seen_proposals.add(key)                   # first sighting — record
    if self._delegate_reactive:
        return False
    self._handle_reactive(event)                    # scaffold as before
    return False
```

**Budget guard.** `_handle_reactive` currently bundles three concerns: round
increment, budget-exhaustion escalation, and scaffolding. The recurrence
branch needs only the first two (increment + escalation), not scaffolding.
Extract the budget logic so both paths share it:

```python
def _handle_reactive_budget_guard(self, event) -> None:
    """Shared: escalate on budget exhaustion. Assumes the caller already
    incremented _reactive_rounds. Silently drops after the first over-budget
    event (mirrors the existing _handle_reactive semantics)."""
    if self._reactive_rounds > self._max_reactive_rounds:
        return                                      # already escalated; drop
    if self._reactive_rounds >= self._max_reactive_rounds:
        self._escalate("budget_exhaustion")
        return
    # under budget — no escalation here; caller decides whether to scaffold
```

Then `_handle_reactive` becomes: increment, call `_handle_reactive_budget_guard`,
and (if under budget) cogitate/scaffold as before. This preserves the
escalation safety net for pure-recurrence stalls (a lesson whose proposals
all recur) — budget-exhaustion escalation still fires.

**`SIG_S4` import.** Reactor imports `SIG_S4` from `kalvin.expand` (the
band-representative value, `0`). The KValue carries the proposal kline (the
structurally-keyed identity Kalvin already saw) at declared S4.

**Return value.** `True` now means "auto-countersigned **or**
recurrence-dropped." Both read as "don't ask the supervisor about this
proposal" at the trainer's `if not auto_matched:` branch (`trainer.py:408`),
which is correct — a dropped proposal must not trigger a `ratify_request`.

### Task 4 — Spec updates (cascade)

- **`specs/kvalue.md §What a KValue is Not`** — replace the
  "Consumption of an inbound KValue's significance… not specified here"
  bullet with a slimmed invariant: rationalisation consumes the declared
  significance (see @agent spec); this spec owns only the exchange unit —
  it guarantees the declared significance is present and addressable; the
  consumption contract is out of scope here.
- **`specs/agent.md §Rationalisation`** — add the pre-ground significance
  gate (derived vs declared, S4-drop, drop = return True with no STM/event,
  derived==declared any band → process normally, S1/S2/S3 disagreement →
  deferred). Append new test-matrix IDs (AGT-NN) per the cascade "never
  renumber" rule.
- **`specs/harness-server.md §Kalvin Adapter`** — add the `rationalise`
  action to the action list and the §Trainer recurrence → declared-S4
  behaviour (second sighting, counts toward budget, returns "handled").

## Test Mapping

| Spec ID | Test (file) | Status |
| ------- | ----------- | ------ |
| AGT-NN (gate: derived==declared → process) | `tests/test_agent.py` — declared == derived (incl. S4 identity) → normal processing | new |
| AGT-NN (gate: S4-disagreement → drop) | `tests/test_agent.py` — declared S4, derived ≠ S4 → returns True, no STM write, no event | new |
| AGT-NN (gate: S1/S2/S3 disagreement → ignored) | `tests/test_agent.py` — declared S2, derived S3 → normal processing | new |
| (adapter `rationalise` action) | `tests/test_adapter.py` — live KValue + wire dict reach `kagent.rationalise`; missing-significance dict fails loud | new |
| (reactor recurrence) | `tests/test_reactor.py` — 1st sighting scaffolds, adds to `_seen_proposals`; 2nd sighting sends `rationalise` at SIG_S4, returns True | new |
| (reactor budget preserved) | `tests/test_reactor.py` — pure-recurrence at budget cliff still escalates `budget_exhaustion` | new |
| (reactor per-lesson reset) | `tests/test_reactor.py` — `load_lesson` clears `_seen_proposals` | new |

Baseline to re-run: the full suite (reactor/agent/adapter/trainer/cogitation)
must stay green. The gate's `derived==declared` path is the common case, so
existing tests that submit compiled entries (whose declared band always
equals their derived band) should be unaffected — verify, don't assume.

## Design Decisions

| Decision | Outcome | Rationale |
| --- | --- | --- |
| Primitive for "structural significance" | `derive_significance` | only structural significance primitive that isn't itself the rationalisation (expansion) |
| Gate placement | before the Phase 2 ground check | recurring proposals are in Frame; a post-ground gate is inert against the one target |
| Drop semantics | `return True`, no STM, no event | "drop" = don't re-cogitate, leave Frame untouched; not a veto primitive |
| Declared-S4 source | trainer-constructed over a proposal kline | compiler's IDENTITY→S4 always agrees with derived-S4, so it can't reach the drop branch |
| Recurrence trigger | second sighting (intra-expectation fan-out) | first-sighting drop would gate out all reactive scaffolding; cross-round recurrence is covered by the budget |
| Recurrence state | `_seen_proposals: set[EntryKey]`, structural key, per-lesson reset | structural key → only exact re-emission counts; refined `P'` is genuinely new |
| Bus action | new `"rationalise"` action | `submit`/`countersign` mutate the payload's meaning; a new action is significance-agnostic and future-proof |
| Budget interaction | recurrence counts toward `_reactive_rounds`, skips scaffolding | preserves the escalation safety net for pure-recurrence stalls; a recurring proposal is the clearest "Kalvin is stuck" signal |
| `process_s2_s3` return on recurrence | `True` | overload accepted: "auto-countersigned or recurrence-dropped" both mean "don't ask the supervisor"; a tri-state would over-engineer one MVP branch |

## Deferred

- Consuming declared **S1/S2/S3** disagreement (the gate returns True only on
  S4; other disagreements are ignored).
- Cross-round recurrence (a scaffold re-yielding the same `P`) — covered by
  `_reactive_rounds`, not the seen-set.
- A distinct `"veto"`/`"drop"` event kind — the MVP emits no event on drop.

## Status

- [x] Grill — design resolved (see session summary)
- [x] Plan (this document)
- [x] Task 1: `KAgent.rationalise` significance gate
- [x] Task 2: `"rationalise"` adapter action
- [x] Task 3: Reactor recurrence → declared-S4
- [x] Task 4: Spec updates (kvalue, agent, harness-server)
- [x] Tests (see Test Mapping)
