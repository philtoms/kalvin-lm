# S3 Auto-Countersign — Implementation Plan

## Evidence

Auto-tune session: `auto-tune/s3-scaffolding-bypass/` (runs 1–6)

## Changes Made

### 1. `src/trainer/reactor.py` — `process_s2_s3` returns `bool`

- Return type changed from `None` to `bool`
- Returns `True` immediately when `_auto_countersign` succeeds
- Calls `_handle_reactive` and returns `False` when auto-countersign fails

```python
def process_s2_s3(self, event: RationaliseEvent) -> bool:
    if self._auto_countersign(event.proposal):
        return True
    self._handle_reactive(event)
    return False
```

### 2. `src/trainer/trainer.py` — conditional `ratify_request` suppression

- Captures `bool` return from `process_s2_s3` as `auto_matched`
- `ratify_request` only sent when `auto_matched` is `False`
- Event relay remains unconditional

```python
auto_matched = self._reactor.process_s2_s3(event)

if not auto_matched:
    self._bus.send(Message(
        role=SUPERVISOR_ROLE,
        action="ratify_request",
        ...
    ))
```

### 3. `curricula/s3-auto-countersign.md` — test curriculum

Curriculum designed to trigger S3 classification with auto-countersign
matching co-entries. Achieves zero-LLM / zero-supervisor but S3 is not
exercised due to S1 looseness (see CI-1 in spec).

## Test Mapping

| Spec ID | Test | Status |
|---------|------|--------|
| SAC-1 | `TestProcessS2S3ReturnTrue::test_returns_true_on_auto_countersign` | ✅ |
| SAC-2 | `TestProcessS2S3ReturnFalse::test_returns_false_on_no_match` | ✅ |
| SAC-3 | `TestHandleReactiveNotCalledOnAutoCountersign::test_no_escalation_on_auto_countersign` | ✅ |
| SAC-4 | `TestTrainerRatifySuppression::test_ratify_suppressed_on_auto_countersign` | ✅ |
| SAC-5 | `TestTrainerRatifySuppression::test_ratify_sent_when_auto_countersign_fails` | ✅ |
| SAC-6 | `TestEventRelayRegardless::test_relay_on_auto_countersign` + `test_relay_on_no_auto_countersign` | ✅ |

## Design Decisions

1. **Return `bool` instead of enum**: The reactor only needs to communicate
   two states — auto-countersign succeeded or not. A `bool` is the simplest
   type that expresses this. No need for an enum or sentinel.

2. **Guard in trainer, not reactor**: The decision to suppress `ratify_request`
   belongs to the trainer (the component that sends the message). The reactor
   reports what happened; the trainer decides what to do about it.

3. **Event relay unaffected**: All events are relayed to the supervisor
   regardless of auto-countersign outcome. This preserves observability —
   the supervisor sees every event even when no ratification is needed.

## Deferred Work

The session identified S1 classification looseness as a blocking issue
for exercising S3. Tightening S1 requires careful analysis of cascade
effects on existing tests and is deferred to a future session. See
continuation issues CI-1 and CI-2 in the spec.
