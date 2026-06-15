# Sub-Plan: Expand Robustness + Significance Normalization

**Parent:** structural-grounding work, discovered during auto-tune session `s2-expansion`
**Estimate:** Done — implementation and tests complete
**Evidence:** `auto-tune/s2-expansion/` (session directory with runs, logs)

---

## 0. What Changed and Why

During auto-tune runs extending `first-steps` to trigger S2 expansion, two
issues were discovered:

1. **`expand()` crashed** with `ValueError: Node 0x0 does not resolve to any KLine`
   when countersigned klines (M↔H) created resolution cycles and identity
   klines yielded `make_signature([]) = 0`.

2. **Significance normalization showed S2 as 1.0** — the formula `raw/D_MAX`
   loses float64 precision for distances 0–100 relative to D_MAX (2⁶⁴),
   making S1 and S2 indistinguishable in logs and event streams.

---

## 1. Spec References

- **@expand-robustness spec** — edge_hops cycle detection, identity guard, null-safe expand.
  Test matrix: ER-1 through ER-7.
- **@significance-normalization spec** — distance-based normalization formula.
  Test matrix: SN-1 through SN-6.

---

## 2. Implementation Tasks

### Task 1: edge_hops cycle detection (ER-1)

**File:** `src/kalvin/expand.py` — `edge_hops()`

Added `visited: set[int]` to detect when the resolution chain revisits a
signature. Prevents countersigned pairs (e.g., `{M:[H]} ↔ {H:[M]}`) from
oscillating for all MAX_HOP=100 iterations.

### Task 2: Identity kline guard (ER-2)

**File:** `src/kalvin/expand.py` — `edge_hops()`

After computing `sig = make_signature(kline.nodes)`, break immediately if
`sig == 0` (identity kline with empty nodes). Identity klines are dead ends
— there is no signature to follow from `nodes = []`.

### Task 3: Null-safe expansion (ER-6, ER-7)

**File:** `src/kalvin/expand.py` — `expand()`

Replaced `_as_kline(model, match_sig)` (which raises ValueError on unresolvable
signatures) with `model.find(match_sig)` + None guard at three call sites:
- Mismatched query node → mismatched candidate node
- Mismatched candidate node → mismatched query node
- S3 connotation bridging

Unresolvable match signatures are silently skipped.

### Task 4: Significance normalization (SN-1 through SN-6) — SUPERSEDED

> **Superseded by [ADR-0007](../../docs/adr/0007-band-anchored-significance-normalization.md)**
> and Task 6 below. The original `max(0.0, 1.0 - distance / S2_S3_DISTANCE)`
> formula collapsed every S3 value to `0.0`. See DD-1 supersession note.

### Task 5: Auto-tune git sandbox fix

**File:** `src/participants/auto_tune/session.py` — `_git()`

Added `GIT_CONFIG_NOSYSTEM=1` and removed `HOME` from environment to allow
git operations in sandboxed environments where `.gitconfig` is inaccessible.

### Task 6: Band-anchored normalization (ADR-0007)

**Spec:** `@significance-normalization` (rewritten). **ADR:** `0007`.

Replaces the DD-1 formula (clamped S3 to `0.0`) with band-anchored
normalization so S3 regains full granularity.

- **`src/kalvin/expand.py`** — add `normalise_significance(raw_sig) -> float`,
  the single source of truth, implementing the band-anchored formula
  (S1=1.0; S2 linear in `[0.50, 0.99]`; S3 asymptotic
  `0.50 · S3_K / (S3_K + (distance - 100))`; S4=0.0). Add the constants
  `S2_TOP`, `S2_FLOOR`, `S3_K`. Re-export `normalise_significance`.
- **`src/participants/auto_tune/events.py`** — `_build_significance()` calls
  `normalise_significance(raw_sig)` for the `normalised` field (replaces the
  inline `max(0.0, 1.0 - distance / S2_S3_DISTANCE)`).
- **`src/trainer/trainer.py`** — `sig_norm` via `normalise_significance`; log
  line additionally shows raw distance, e.g. `→ 0.17 (d=200)` (keeps the
  debug readability the `s3-distance` session introduced).
- **`tests/test_normalise_significance.py`** (new) — the SN-1..SN-7 matrix for
  the shared helper (see Test Mapping below).
- **`tests/test_auto_tune_events.py`** — update the existing SN assertions to
  the band-anchored ranges (S2 ∈ [0.50, 0.99]; S3 ∈ (0.0, 0.50), non-zero).

Distance semantics, `classify()`, and routing are **unchanged**.

---

## 3. Test Mapping

### Expand Robustness (ER-*)

| Spec ID | Test File | Test Function | Status |
|---------|-----------|---------------|--------|
| ER-1 | `tests/test_expand.py` | `TestEdgeHops::test_edge_hops_chain` | ✅ |
| ER-2 | `tests/test_expand.py` | `TestEdgeHops::test_edge_hops_identity_kline_er2` | ✅ |
| ER-3 | `tests/test_expand.py` | `TestEdgeHops::test_edge_hops_canonical` | ✅ |
| ER-4 | `tests/test_expand.py` | `TestEdgeHops::test_edge_hops_unresolvable` | ✅ |
| ER-5 | `tests/test_expand.py` | `TestEdgeHops::test_edge_hops_chain` | ✅ |
| ER-6 | `tests/test_expand.py` | `TestExpand::test_expand_no_crash_on_unresolvable_match_sig_er6` | ✅ |
| ER-7 | `tests/test_expand.py` | `TestExpand::test_expand_countersign_cycle_no_crash_er7` | ✅ |

### Significance Normalization (SN-*)

> Rows below cover the **DD-1 / Task 4** formula (superseded). The
> **band-anchored** formula (Task 6, ADR-0007) reuses the same test IDs with
> revised criteria — see the Task 6 test-mapping table.

| Spec ID | Test File | Test Function | Status |
|---------|-----------|---------------|--------|
| SN-1 | `tests/test_auto_tune_events.py` | `TestSignificanceObject::test_s2_less_than_s1` | ✅ |
| SN-2 | `tests/test_auto_tune_events.py` | `TestSignificanceObject::test_s2_less_than_s1` | ✅ |
| SN-3 | `tests/test_auto_tune_events.py` | `TestSignificanceObject::test_s1_range_near_one` | ✅ |
| SN-4 | `tests/test_auto_tune_events.py` | `TestSignificanceObject::test_s23_boundary` | ✅ |
| SN-5 | `tests/test_auto_tune_events.py` | `TestSignificanceObject::test_zero_significance` | ✅ |
| SN-6 | `tests/test_auto_tune_events.py` | `TestSignificanceObject::test_midrange_normalisation` | ✅ |

### Band-Anchored Normalization (Task 6, ADR-0007)

Revised criteria for the rewritten `@significance-normalization` spec. These
supersede the SN-* criteria above; test functions move to a dedicated test
module for the shared `normalise_significance` helper.

| Spec ID | Test File | Test Function | Status |
|---------|-----------|---------------|--------|
| SN-1 | `tests/test_normalise_significance.py` | `test_strict_band_ordering_s1_gt_s2_gt_s3_gt_s4` | ✅ |
| SN-2 | `tests/test_normalise_significance.py` | `test_s1_normalises_to_one` | ✅ |
| SN-3 | `tests/test_normalise_significance.py` | `test_s2_range_and_monotonic` | ✅ |
| SN-4 | `tests/test_normalise_significance.py` | `test_s3_asymptotic_never_zero` | ✅ |
| SN-5 | `tests/test_normalise_significance.py` | `test_raw_zero_to_zero` | ✅ |
| SN-6 | `tests/test_normalise_significance.py` | `test_s3_injective_no_collapse` | ✅ |
| SN-7 | `tests/test_normalise_significance.py` | `test_global_monotonic` | ✅ |

---

## 4. Design Decisions

### DD-1: Normalize against S2_S3_DISTANCE, not D_MAX — SUPERSEDED

> **Superseded by [ADR-0007 — Band-anchored significance normalization](../../docs/adr/0007-band-anchored-significance-normalization.md).**
>
> DD-1 allocated the entire `[0.0, 1.0]` range to S1+S2 and clamped every S3
> value to `0.0`. Empirical auto-tune data shows S3 dominates rationals
> (~75%) and spans a 5× distance range (101→519) — all of that ordering was
> erased. ADR-0007 replaces the formula with band-anchored normalization:
> each band owns a fixed sub-range and S3 is asymptotic in `(0.0, 0.50)`, so
> every distinct S3 distance yields a distinct normalized value. See Task 6.

### DD-2: model.find + None guard instead of _as_kline

`_as_kline` raised ValueError on unresolvable nodes, which is wrong for a
graph traversal — dead ends are normal topology, not errors. Using `model.find`
with a None guard silently skips unresolvable branches.

### DD-3: Cycle detection via visited set in edge_hops

The cycle is detected at the edge_hops level (per-chain visited set) rather
than at the expand level (which already has _visited for query-candidate pairs).
These are different cycles: edge_hops follows resolution chains
(sig→nodes_sig→sig), while expand tracks (query.sig, candidate.sig) pairs.

---

## 5. Status

All implementation complete. All existing tests pass (471 sync tests).
Auto-tune session `s2-expansion` completed cleanly without crashes.
