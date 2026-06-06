# Countersign Resolution — Implementation Plan

## Source

Auto-tune session `mhall-svo` (branch `auto-tune/mhall-svo`).
Evidence: `auto-tune/mhall-svo/runs/001` through `runs/005`.

## Changes Made

### 1. KLine sig_level field
- **File:** `src/kalvin/kline.py`
- **Change:** Added `sig_level: str | None = None` field to `KLine.__slots__` and `__init__`.
- **Status:** ✅ Complete
- **Tests:** 344 passing

### 2. CompiledEntry sig_level propagation
- **Files:** `src/kscript/token_encoder.py`
- **Changes:**
  - `CompiledEntry.__init__` accepts `sig_level` parameter
  - `CompiledEntry.encode` passes `sig_level` to constructor
  - `TokenEncoder._encode_one` reads operator → level mapping and passes to constructor
- **Status:** ✅ Complete
- **Tests:** 344 passing

### 3. Ground check excludes STM
- **Files:** `src/kalvin/model.py`, `src/kalvin/agent.py`
- **Changes:**
  - Added `Model.grounded()` → `_TierChain.contains_excluding_first()`
  - `rationalise()` Phase 2 uses `grounded()` instead of `exists()`
- **Status:** ✅ Complete
- **Test:** ☐ `test_grounded_excludes_stm`

### 4. Pre-registration in adapter
- **File:** `src/harness/adapter.py`
- **Change:** `_handle_submit()` calls `model.add_stm(entry)` for all entries before rationalisation loop.
- **Status:** ✅ Complete
- **Test:** ☐ `test_submit_pre_registers_entries_in_stm`

### 5. Self-filter in candidate retrieval
- **File:** `src/kalvin/agent.py`
- **Change:** `rationalise()` Phase 5 filters out the query kline from `where()` results.
- **Status:** ✅ Complete
- **Test:** ☐ `test_rationalise_excludes_self_from_candidates`

### 6. STM registration before countersign check
- **File:** `src/kalvin/agent.py`
- **Change:** `add_stm()` moved before `is_countersigned()` check in `rationalise()`.
- **Status:** ✅ Complete

## Test Mapping

| Spec ID | Test | Status |
|---------|------|--------|
| CR-1 | `test_countersign_pair_both_resolve_s1` | ☐ |
| CR-2 | `test_submit_pre_registers_entries_in_stm` | ☐ |
| CR-3 | `test_grounded_returns_false_for_stm_only` | ☐ |
| CR-4 | `test_grounded_returns_true_for_frame_entry` | ☐ |
| CR-5 | `test_rationalise_excludes_self_from_candidates` | ☐ |
| CR-6 | `test_sig_level_set_on_compiled_entry` | ☐ |
| CR-7 | `test_undersign_no_special_fast_path` | ☐ |
| CR-8 | `test_connotate_goes_through_slow_path` | ☐ |
