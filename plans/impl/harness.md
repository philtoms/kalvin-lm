# Test Harness Implementation Plan

**Parent:** `docs/roadmap.md` Phase B
**Status:** not started
**Spec refs:** `specs/harness.md`

## Spec References

- `@specs/harness.md` — HRN-1 through HRN-18
- `@specs/agent.md` — rationalise API, event bus (AGT-1..AGT-40)
- `@specs/kscript.md` — compilation, CompiledEntry (KS-1..KS-33)

## Implementation Tasks

### Task 1: Agent.countersign (src/kalvin/agent.py)

- **Spec ref:** @specs/harness §Agent.countersign, HRN-9, HRN-16
- **Test mapping:** HRN-16
- **Details:**
  - Add `countersign(kline: KLine) -> bool` method to Agent
  - Generate reciprocal: `make_signature(kline.nodes)` → reciprocal signature, `[kline.signature]` → reciprocal nodes
  - Call `self.rationalise(reciprocal_kline)`
  - Return the result

### Task 2: Tracking state model (ui/kscript/app.py)

- **Spec ref:** @specs/harness §Tracking State, HRN-1, HRN-2, HRN-10, HRN-12
- **Test mapping:** HRN-1, HRN-2, HRN-10, HRN-12
- **Details:**
  - Add `_submitted: set[tuple[int, tuple[int,...]]]` — keyed by (signature, nodes_tuple)
  - Add `_satisfied: set[tuple[int, tuple[int,...]]]`
  - Helper `_entry_key(entry) -> tuple[int, tuple[int,...]]` for hashing
  - Compile returns entries; diff against submitted to find pending
  - Clear action resets both sets
  - `_save_state` / `_restore_state` include submitted/satisfied in JSON

### Task 3: ResponseItem with status and significance (ui/kscript/regions/responses.py)

- **Spec ref:** @specs/harness §Response Status, §Significance Display, HRN-13
- **Test mapping:** HRN-13
- **Details:**
  - Extend `ResponseItem` to hold: status (pass/pending/mismatch), raw significance (hex string), normalised significance (float string)
  - Update display format: `✓ MHALL => SVO  0xFFFF_FFFF_FFFF_FFFF  1.000`
  - Keep existing S1-S4 filter buttons

### Task 4: Event handler rewrite (ui/kscript/app.py)

- **Spec ref:** @specs/harness §Satisfaction, §Event Correlation, HRN-3, HRN-4, HRN-11, HRN-14, HRN-17, HRN-18
- **Test mapping:** HRN-3, HRN-4, HRN-11, HRN-14, HRN-17, HRN-18
- **Details:**
  - Rewrite `_setup_events` callback:
    - On event: correlate to compiled entry by structural match
    - If rationalise was fast-path: auto-satisfy, display with ✓
    - If slow-path proposal: decompile, compute significance display, display with status
    - If compilation error: display as ✗ response
  - Multiple proposals per expectation all displayed
  - Track which expectations have received proposals

### Task 5: Run mode (ui/kscript/app.py)

- **Spec ref:** @specs/harness §Run Mode, HRN-5, HRN-6, HRN-17
- **Test mapping:** HRN-5, HRN-6, HRN-17
- **Details:**
  - Rewrite `action_run_script` / `_auto_compile_tick`:
    - Compile → diff against submitted → submit all pending sequentially
    - Auto-countersign any proposal that structurally matches an expectation
    - Flag mismatches as pending
    - Continue until all pending submitted

### Task 6: Step mode (ui/kscript/app.py)

- **Spec ref:** @specs/harness §Step Mode, HRN-7, HRN-8
- **Test mapping:** HRN-7, HRN-8
- **Details:**
  - Rewrite `action_step_script`:
    - Compile → diff against submitted → submit first pending entry → halt
    - User inspects response, may ratify
  - Ratify button enabled only when response item selected

### Task 7: Ratify action (ui/kscript/regions/toolbar.py, ui/kscript/app.py)

- **Spec ref:** @specs/harness §Step Mode, HRN-8, HRN-9
- **Test mapping:** HRN-8, HRN-9
- **Details:**
  - Add "Ratify" button to ToolbarRegion (always visible, enabled when response selected)
  - Add `ToolbarRegion.Ratify` message
  - App handler: call `agent.countersign(selected_proposal)` with proposal as-is
  - Move entry to satisfied set on success

### Task 8: Progress display (ui/kscript/regions/toolbar.py)

- **Spec ref:** @specs/harness §Progress Display, HRN-15
- **Test mapping:** HRN-15
- **Details:**
  - Extend status text: `◐ HALTED  5/12 | 3 pending`
  - Update on each submission and satisfaction event

### Task 9: Hot-reload state persistence (ui/kscript/app.py)

- **Spec ref:** @specs/harness §State Persistence, HRN-12
- **Test mapping:** HRN-12
- **Details:**
  - Extend `_save_state` JSON with `submitted` and `satisfied` arrays (signature+nodes pairs)
  - Extend `_restore_state` to reconstruct the sets from JSON

## Test Mapping

| Spec ID | Test file | Test function | Status |
|---------|-----------|---------------|--------|
| HRN-1 | test_harness.py | test_recompile_only_new_submitted | ❌ |
| HRN-2 | test_harness.py | test_submitted_monotonic | ❌ |
| HRN-3 | test_harness.py | test_fast_path_auto_satisfied | ❌ |
| HRN-4 | test_harness.py | test_structural_match_expectation | ❌ |
| HRN-5 | test_harness.py | test_run_submits_all_pending | ❌ |
| HRN-6 | test_harness.py | test_run_auto_countersigns | ❌ |
| HRN-7 | test_harness.py | test_step_submits_one_halts | ❌ |
| HRN-8 | test_harness.py | test_ratify_enabled_on_selection | ❌ |
| HRN-9 | test_harness.py | test_ratify_calls_countersign | ❌ |
| HRN-10 | test_harness.py | test_clear_resets_tracking | ❌ |
| HRN-11 | test_harness.py | test_event_correlation_structural | ❌ |
| HRN-12 | test_harness.py | test_state_persistence_hotreload | ❌ |
| HRN-13 | test_harness.py | test_response_display_format | ❌ |
| HRN-14 | test_harness.py | test_compilation_error_display | ❌ |
| HRN-15 | test_harness.py | test_progress_count_display | ❌ |
| HRN-16 | test_agent.py | test_agent_countersign | ❌ |
| HRN-17 | test_harness.py | test_run_mismatch_flagged_pending | ❌ |
| HRN-18 | test_harness.py | test_multiple_proposals_displayed | ❌ |

## Design Decisions

1. **Structural match over object identity** — events may originate from background threads; object identity is fragile. Structural match (signature + nodes) is robust.
2. **Countersign on Agent, not UI** — countersigning is a Kalvin-level concept. The UI calls `agent.countersign()`, keeping the harness thin.
3. **Monotonic tracking** — Kalvin's model only grows; the harness tracking mirrors this. Only Clear resets.
4. **Entry key as tuple** — (signature, tuple(nodes)) is hashable and unique for structural identity.
5. **Normalised significance = significance / D_MAX** — maps to 0.0–1.0 where S4=0.0, S1≈1.0.

## Status

Not started.
