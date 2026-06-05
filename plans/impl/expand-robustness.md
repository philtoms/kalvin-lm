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

### Task 4: Significance normalization (SN-1 through SN-6)

**Files:**
- `src/participants/auto_tune/events.py` — `_build_significance()`
- `src/trainer/trainer.py` — log normalization

Changed formula from `raw / D_MAX` to `max(0.0, 1.0 - distance / S2_S3_DISTANCE)`
where `distance = (~raw) & MASK64`. Both sites use identical formula.

### Task 5: Auto-tune git sandbox fix

**File:** `src/participants/auto_tune/session.py` — `_git()`

Added `GIT_CONFIG_NOSYSTEM=1` and removed `HOME` from environment to allow
git operations in sandboxed environments where `.gitconfig` is inaccessible.

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

| Spec ID | Test File | Test Function | Status |
|---------|-----------|---------------|--------|
| SN-1 | `tests/test_auto_tune_events.py` | `TestSignificanceObject::test_s2_less_than_s1` | ✅ |
| SN-2 | `tests/test_auto_tune_events.py` | `TestSignificanceObject::test_s2_less_than_s1` | ✅ |
| SN-3 | `tests/test_auto_tune_events.py` | `TestSignificanceObject::test_s1_range_near_one` | ✅ |
| SN-4 | `tests/test_auto_tune_events.py` | `TestSignificanceObject::test_s23_boundary` | ✅ |
| SN-5 | `tests/test_auto_tune_events.py` | `TestSignificanceObject::test_zero_significance` | ✅ |
| SN-6 | `tests/test_auto_tune_events.py` | `TestSignificanceObject::test_midrange_normalisation` | ✅ |

---

## 4. Design Decisions

### DD-1: Normalize against S2_S3_DISTANCE, not D_MAX

D_MAX = 2⁶⁴ exceeds float64 mantissa (53 bits). Distances 0–100 are lost when
divided by D_MAX. S2_S3_DISTANCE (100) is the natural scale for the S2 band,
giving meaningful granularity: S1 ≥ 0.99, S2 = 0.98→0.00, S3 = 0.00.

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
