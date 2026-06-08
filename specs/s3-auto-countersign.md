# S3 Auto-Countersign — Specification

## Overview

When auto-countersign succeeds for an S2/S3 proposal, the trainer should
not send a `ratify_request` to the supervisor — the proposal is already
ratified. Previously, `ratify_request` was sent unconditionally for every
S2/S3 event, even when auto-countersign had already handled it.

This spec defines two changes: (1) the reactor's `process_s2_s3` returns
whether auto-countersign succeeded, and (2) the trainer conditionally
suppresses `ratify_request` accordingly.

## Evidence

Auto-tune session: `auto-tune/s3-scaffolding-bypass/`
- Runs 1–6 tested curricula targeting S3 auto-countersign scenarios
- Code changes committed: `process_s2_s3` returns `bool`, trainer suppresses `ratify_request`
- Session identified S1 classification looseness as a blocking issue for S3 exercise (documented in session-state.md, deferred to future work)

## Root Cause

`Reactor.process_s2_s3()` previously returned `None`. The trainer had no
way to know whether auto-countersign handled the event, so it sent
`ratify_request` for every S2/S3 event — even when auto-countersign had
already resolved the proposal. This caused unnecessary supervisor
interaction for events that needed no ratification.

## Fixes Applied

### SAC-1: `process_s2_s3` returns `bool`

`Reactor.process_s2_s3()` now returns:
- `True` — auto-countersign succeeded, no supervisor interaction needed
- `False` — auto-countersign failed, reactive handling was invoked (scaffolding or escalation)

The function checks auto-countersign first. If it succeeds, returns `True`
immediately without calling `_handle_reactive`. If auto-countersign fails,
calls `_handle_reactive` and returns `False`.

### SAC-2: Trainer conditionally suppresses `ratify_request`

`Trainer._handle_rationalise()` captures the `bool` return from
`process_s2_s3`. It only sends `ratify_request` to the supervisor when
auto-countersign did **not** succeed (`auto_matched is False`).

## Behavioural Rules

1. `Reactor.process_s2_s3()` must return `True` when auto-countersign
   succeeds and `False` when reactive handling is invoked.
2. When auto-countersign succeeds, `_handle_reactive` must not be called.
3. The trainer must not send `ratify_request` for S2/S3 events where
   auto-countersign succeeded.
4. The trainer must still send `ratify_request` for S2/S3 events where
   auto-countersign failed.
5. Event relay to the supervisor must continue regardless of auto-countersign
   outcome (all events are relayed).

## Test Matrix

| ID   | Criterion | Status |
|------|-----------|--------|
| SAC-1 | `process_s2_s3` returns `True` when auto-countersign succeeds | ✅ |
| SAC-2 | `process_s2_s3` returns `False` when auto-countersign fails | ✅ |
| SAC-3 | `_handle_reactive` not called when auto-countersign succeeds | ✅ |
| SAC-4 | `ratify_request` suppressed when auto-countersign succeeds | ✅ |
| SAC-5 | `ratify_request` sent when auto-countersign fails | ✅ |
| SAC-6 | Event relay sent regardless of auto-countersign outcome | ✅ |

## Continuation Issues

### CI-1: S1 subset check is too permissive

`_route()` (agent.py) classifies as S1 when `match_count == total` — all
query nodes are a subset of the candidate's nodes. For single-node queries
(e.g., `H > A`, nodes=[A]), any compound entry containing A matches S1.
This means connotate and countersign entries are consumed by S1 before S3
can fire, preventing the auto-countersign path from being exercised.

**Evidence:** Session runs 1–6 all showed co-entries consumed by S1. No
curriculum achieved S3 exercise with auto-countersign.

**Possible fix:** Tighten S1 to require signature alignment (bitwise AND),
not just node subset. Must assess cascade effects on existing tests.

### CI-2: STM pre-registration + loose S1 creates preemption cascade

The harness adapter pre-registers ALL lesson entries in STM before
rationalisation. Combined with loose S1 (CI-1), co-entries whose nodes
appear in other co-entries get consumed immediately. Fix for CI-1 would
likely resolve this as well.
