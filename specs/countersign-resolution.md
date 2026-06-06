# Countersign Resolution Specification

## Overview

When a KScript countersign construct (`A == B`) compiles to two directional entries (`{A: B}` and `{B: A}`), both entries must resolve as S1. The original implementation submitted entries sequentially, so the first entry could never find its countersigner because the second hadn't been added to the model yet.

## Dependencies

This spec depends on:

### KLine (@specs/kline)
- Provides `sig_level` field carrying significance level from compilation.

### KScript (@specs/kscript)
- Countersign (`==`) compiles to two reciprocal entries.
- Undersign (`=`) and connotate (`>`) compile to the same single-node kline structure in opposite directions.

### Model (@specs/model)
- `grounded()` checks Frame, LTM, Base (excludes STM).
- `add_stm()` for transient registration.

## Discovery: Undersign IS Connotate Reversed

Undersign (`S = M → {M: S}`) and connotate (`S > M → {S: M}`) produce structurally identical klines — a single-node entry pointing in opposite directions. Kalvin cannot and should not distinguish them at the rationalisation level. The `sig_level` field (`S1` for undersign, `S3` for connotate) is **training-side metadata** used by the trainer/harness for ratification decisions, not by Kalvin's rationaliser.

This means undersign entries go through the same candidate retrieval and graph expansion path as connotate entries — no special fast path.

## Definition

### Pre-registration

Before rationalising any compiled entry, all entries from the batch are added to STM. This ensures countersign pairs can find each other via `is_countersigned()`.

| Property | Value |
|----------|-------|
| Location | `harness/adapter.py` `_handle_submit()` |
| Mechanism | `self._kagent.model.add_stm(entry)` for each entry before rationalisation loop |

### Ground Check (STM exclusion)

The Phase 2 ground check in `rationalise()` excludes STM entries. Only Frame, LTM, and Base are checked for grounding.

| Method | Location | Behaviour |
|--------|----------|-----------|
| `Model.grounded()` | `model.py` | Checks Frame, LTM, Base via `_TierChain.contains_excluding_first()` |
| `_TierChain.contains_excluding_first()` | `model.py` | Skips the first adapter (STM) |

### Self-filter in Candidate Retrieval

Phase 5 candidate retrieval excludes the query kline itself to prevent trivial S1 self-matching.

### Countersign Resolution Path

For `{A: B}` (countersign direction 1):
1. Pre-registered in STM (adapter)
2. Phase 2 ground check: `grounded()` → False (not in Frame/LTM)
3. Phase 3: `add_stm()` → already present
4. Phase 3: `is_countersigned({A: B})` → looks for `{B: [A]}` → found in STM (pre-registered) → S1

### sig_level on KLine

| Field | Type | Default | Source |
|-------|------|---------|--------|
| `sig_level` | `str \| None` | `None` | Set during `CompiledEntry.encode()` via `TokenEncoder` |

Significance levels by operator:

| Operator | sig_level |
|----------|-----------|
| COUNTERSIGN (`==`) | `S1` |
| UNDERSIGN (`=`) | `S1` |
| CANONIZE (`=>`) | `S2` |
| CONNOTATE (`>`) | `S3` |
| UNSIGNED | `S4` |
| IDENTITY | `S1` |

`sig_level` is **not used by the rationaliser**. It is training-side metadata for the trainer's ratification decisions.

## Behavioural Rules

1. Pre-registration adds all compiled entries to STM before any is rationalised.
2. `Model.grounded()` checks Frame, LTM, Base — never STM.
3. `rationalise()` filters self-matching from candidate retrieval.
4. `is_countersigned()` checks across all tiers including STM.
5. Undersign and connotate entries follow the same rationalisation path — no special fast path for either.
6. `sig_level` is set on `CompiledEntry` during compilation but not consumed by `rationalise()`.

## Test Matrix

| ID | Criterion | Origin |
|----|-----------|--------|
| CR-1 | Countersign pair `{M: H}, {H: M}` both resolve S1 via `is_countersigned()` | Session runs 1-4 |
| CR-2 | Pre-registration ensures both countersign directions are in STM before rationalisation | Adapter change |
| CR-3 | `Model.grounded()` returns False for STM-only entries | Ground check fix |
| CR-4 | `Model.grounded()` returns True for Frame/LTM entries | Ground check fix |
| CR-5 | Candidate retrieval excludes self-matching klines | Self-filter |
| CR-6 | `sig_level` is set on CompiledEntry matching operator type | TokenEncoder |
| CR-7 | Undersign `{M: S}` does NOT get a special fast path in rationalise | Discovery |
| CR-8 | Connotate `{A: D}` goes through slow path (candidate retrieval + cogitation) | Session runs |

## Continuation Issues

The following issues were identified during this session and are candidates for the next auto-tune session:

### CI-1: Cogitator processes already-resolved entries

After `{SVO: MHALL}` resolves as S1 via `is_countersigned()`, the cogitator still produces expansion proposals for it (3 rounds of LLM calls). This wastes LLM budget and causes false escalations. The cogitator should skip or early-exit for entries that have already been promoted to LTM.

**Evidence:** `auto-tune/mhall-svo/runs/005/harness.log` — `FRAME OSV > AHLM → S1` followed by 3 rounds of `Cogitate misfit` on the same entry.

### CI-2: Undersign entries without prior scaffolding go through slow path

Undersign entries like `{M: S}` (from `S = M`) go through candidate retrieval as S3 against existing entries like `{M: H}`. Without prior scaffolding (identities pre-grounded), the slow path may not resolve them. This is **correct behaviour** (undersign = connotate reversed), but the LLM cogitator may not reliably bridge the gap.

**Evidence:** `auto-tune/mhall-svo/runs/005/harness.log` — 15/18 satisfied in single-lesson curriculum. The 3 unsatisfied are undersign/connotate entries that the cogitator couldn't bridge.

### CI-3: promote_participating promotes unrelated STM entries

`promote_participating()` promotes ALL STM entries whose signatures appear in the node union of the query and candidate. This means connotate entries like `{L: O}` get promoted to LTM during countersign ratification just because their signature (L) appears in MHALL's nodes. This causes them to ground trivially on re-rationalisation rather than going through proper graph expansion.

**Evidence:** `GROUND L > O → S1 (fast path)` in session runs — L>O is S3 connotate but grounds because it was promoted to LTM as a side effect.

## Session Evidence

| Run | Session | Curriculum | Result | Notes |
|-----|---------|------------|--------|-------|
| 1 | mhall-svo | 5-lesson | 23/23 S1 | sig_level shortcut active |
| 2 | mhall-svo | 5-lesson | 23/23 S1 | Stable, same shortcut |
| 3 | mhall-svo | 5-lesson | 22/23 | sig_level removed, A>D misfit |
| 4 | mhall-svo | 5-lesson | 22/23 | Stable |
| 5 | mhall-svo | 1-lesson (single) | 15/18 | No shortcuts, 3 slow-path entries |

All runs in `auto-tune/mhall-svo/runs/`.
