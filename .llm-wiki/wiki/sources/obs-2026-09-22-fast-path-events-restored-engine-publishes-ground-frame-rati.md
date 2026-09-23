---
type: source
title: "Observation: Fast-path events restored: Engine publishes ground/frame, rationaliser stays silent"
tags:
  - engine
  - events
  - trainer
  - unblocking
status: observation
created: 2026-09-22
updated: 2026-09-22
slug: obs-2026-09-22-fast-path-events-restored-engine-publishes-ground-frame-rati
relevance: high
observed_at: 2026-09-22T09:04:35.741Z
source_context: Restoring fast-path events at the Engine boundary
---

# ⭐ Observation: Fast-path events restored: Engine publishes ground/frame, rationaliser stays silent

Fast-path events restored at the Engine boundary (uncommitted): Engine.rationalise now publishes what the fast path did — "ground" S1 for receipts of already-grounded klines (pre-check before feeding), "frame" S1 for fresh fast-path groundings, "frame" S4 for refusals — checked only when nothing queued (slow-path entries get their events via cogitation emissions instead). This un-blocks the trainer: mark_satisfied fires only from S1 events (or engine errors), so fast-path-grounded entries (identities, S1 receipts, countersign ratifications) had been silently never-satisfied and lessons would stall; the supervisor's ratify resolution (countersign → fast path) is now visible again. The Rationaliser stays pure/silent — publishing is orchestration, and the Engine is the orchestrator; the dialogue harness is unaffected (state-snapshot diffing, no events). Verified: 90 tests, mhall byte-identical, live checks — frame-S1 on fresh receipt, ground-S1 on resubmit, frame-S4 on refusal, frame-S1 on countersign.

*Relevance: high*
*Context: Restoring fast-path events at the Engine boundary*
*Tags: engine events trainer unblocking*

---
*Observed: 2026-09-22T09:04:35.741Z*
