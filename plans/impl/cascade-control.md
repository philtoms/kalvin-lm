# Cascade Control — Implementation Plan

## Evidence

Auto-tune session: `auto-tune/explore-agent-capability/` (runs 6–8)

## Changes

### 1. KAgent.rationalise() — candidate cap

**File:** `src/kalvin/agent.py`

- Add `_max_candidates: int = 8` field to `KAgent.__init__`
- After Phase 5 collects `slow_candidates`, if `len(slow_candidates) > self._max_candidates`:
  - Sort by `(level_rank, -overlap)` where `level_rank` is 0 for S2, 1 for S3
  - Truncate to `_max_candidates`
- S1 candidates are processed before the cap (fast-path break)
- S4 routing (no candidates) is unaffected

### 2. Reactor._handle_reactive() — silent drop

**File:** `src/training/trainer/reactor.py`

- Change budget exhaustion check from `>=` to two-stage:
  - `> max_reactive_rounds`: silent drop (return immediately)
  - `== max_reactive_rounds`: escalate (original behaviour)
- First over-budget event escalates; all subsequent events are dropped

## Test Mapping

| Spec ID | Test | Status |
|---------|------|--------|
| CC-1 | `test_rationalise_caps_candidates` | ✅ |
| CC-2 | `test_s2_prioritised_over_s3` | ✅ |
| CC-3 | `test_higher_overlap_prioritised` | ✅ |
| CC-4 | `test_first_budget_exhaustion_escalates` | ✅ (existing) |
| CC-5 | `test_subsequent_events_silently_dropped` | ✅ |
| CC-6 | `test_s1_fast_path_unaffected` | ✅ (existing) |
| CC-7 | `test_s4_routing_unaffected` | ✅ (existing) |
| CC-8 | `test_default_max_candidates` | ✅ |

## Design Decisions

- **Why top-K instead of smarter `where()`:** Changing `signifies()` would affect all model lookups, not just rationalise. The cap is localised to the rationalise pipeline and easy to reason about.
- **Why 8 as default:** Empirically, 8 candidates provide enough coverage for productive scaffolding while preventing cascade. The conflict-drill curriculum (4 lessons, complex entries) achieved 12/11 satisfaction with this cap.
- **Why silent drop instead of stopping the cogitator:** The cogitator may still be processing work items. Stopping it would lose in-flight results. Silent-drop is the simplest approach that prevents spinning without side effects.
