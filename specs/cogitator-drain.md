# Spec: Cogitator Inter-Lesson Drain

## Summary

The Cogitator is a background thread that processes S2/S3 work items asynchronously.
When a lesson triggers slow-path cogitation, work items may still be processing when
the next lesson begins. These late-arriving events are processed with the new lesson's
reactor state, consuming its reactive budget and corrupting entry satisfaction tracking.

## Behavioural Rules

### DRN-1: Drain before lesson submission
Before submitting each lesson, the Trainer MUST drain the Cogitator backlog.
A drain request is sent via the bus to the KAgent adapter. The adapter calls
`Cogitator.drain()` which blocks until the backlog is empty and the current
work item finishes.

### DRN-2: Deferred lesson submission
While a drain is pending, the Trainer MUST NOT submit lesson entries.
Lesson compilation and submission are deferred until the `drained` response
is received from the adapter.

### DRN-3: No-op for empty backlog
When the Cogitator backlog is empty and no work item is processing,
`drain()` MUST return immediately. This ensures negligible overhead
for lessons that don't trigger slow-path cogitation.

### DRN-4: Drain timeout
The drain operation has a configurable timeout (default 30 seconds).
If the drain times out, the adapter responds with a `drained` message
and the Trainer proceeds. This prevents indefinite blocking.

### DRN-5: Thread-safe processing flag
The Cogitator MUST track a `_processing` flag that indicates whether a
work item is currently being processed. This flag is set before processing
begins and cleared after the work item finishes. The `drain()` method
waits for both the backlog to be empty AND the processing flag to be clear.

## Test Matrix

| ID   | Description                                                    | Status |
|------|----------------------------------------------------------------|--------|
| DRN-1| Drain sent before each lesson, even when no S2/S3 expected     | ✅      |
| DRN-2| Lesson entries not submitted until drained response received   | ✅      |
| DRN-3| Empty-backlog drain completes in <10ms                         | ✅      |
| DRN-4| Drain timeout returns False but does not stop the thread       | ✅      |
| DRN-5| Processing flag correctly guards against premature drain return| ✅      |
| DRN-6| Cross-lesson spillover eliminated: lesson N events don't affect lesson N+1 budget | ✅ |

## Evidence

- Auto-tune session: `auto-tune/explore-agent-capability/`
- Run 4 (without drain): Lesson 4 satisfaction 7/11 (63.6%)
- Run 5 (with drain): Lesson 4 satisfaction 11/11 (100%)
