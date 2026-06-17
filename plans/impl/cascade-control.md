# Cascade Control — Implementation Plan

**Status:** Done — implementation and tests complete.
**Spec refs:**
- `@specs/agent.md` §Candidate Cap (AGT-19a–19e)
- `@specs/reactive-delegation.md` §Reactive-Round Budget (Default Mode) (RD-8a, RD-8b)

## Changes

### 1. `src/kalvin/agent.py` — candidate cap

- Added `max_candidates: int = 8` parameter to `KAgent.__init__` (stored as `_max_candidates`).
- After Phase 5 sorts candidates by `(level_rank, -overlap)` (S1, then S2, then S3; node overlap descending within a level), the S2/S3 portion is truncated to `_max_candidates` before WorkItems are submitted to the Cogitator.
- S1 candidates are processed before the cap (the first S1 resolves the entry and returns early); S4 routing (no candidates) is unaffected.

### 2. `src/training/trainer/reactor.py` — silent drop past budget

- `_handle_reactive` uses a two-stage check against `max_reactive_rounds` (default 5):
  - `_reactive_rounds > max_reactive_rounds`: silent drop (return immediately, no escalation/log/bus message).
  - `_reactive_rounds == max_reactive_rounds`: escalate `budget_exhaustion` (one escalation per lesson).
- The reactive-round counter resets at the start of each lesson.

## Test Mapping

| Spec ID | Test file | Test function | Status |
|---------|-----------|---------------|--------|
| AGT-19a | `tests/test_cascade_control.py` | `test_rationalise_caps_candidates`, `test_default_max_candidates` | ✅ |
| AGT-19b | `tests/test_cascade_control.py` | `test_s2_prioritised_over_s3` | ✅ |
| AGT-19c | `tests/test_cascade_control.py` | `test_higher_overlap_prioritised` | ✅ |
| AGT-19d | `tests/test_cascade_control.py` | `test_s1_fast_path_unaffected` | ✅ |
| AGT-19e | `tests/test_cascade_control.py` | `test_s4_routing_unaffected` | ✅ |
| RD-8a | `tests/test_cascade_control.py` | `test_first_budget_exhaustion_escalates` | ✅ |
| RD-8b | `tests/test_cascade_control.py` | `test_subsequent_events_silently_dropped` | ✅ |

## Design Decisions

1. **Localised cap in `rationalise()` rather than a smarter `model.where()`.** Changing `signifies()` would affect every model lookup, not just rationalisation. The cap is local to the rationalise pipeline and easy to reason about.

2. **Silent drop rather than stopping the Cogitator.** The Cogitator may still have in-flight WorkItems; stopping it would lose those results. Silent-drop prevents the Reactor from spinning on the event stream while the Cogitator drains, without side effects.
