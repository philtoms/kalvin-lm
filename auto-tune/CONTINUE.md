# Auto-Tune Continuation Issues

Branch: `auto-tune/mhall-svo`
Origin spec: `specs/countersign-resolution.md`

These issues were identified during the mhall-svo auto-tune session and are
ready for the next tuning session. Pick one, invoke auto-tune, and state the
issue as the goal.

---

## CI-1: Cogitator processes already-resolved entries

After a countersign entry resolves as S1 via `is_countersigned()`, the
cogitator still produces expansion proposals for it across 3 rounds of LLM
calls. This wastes LLM budget and causes false `low_confidence` escalations.
The cogitator should skip or early-exit for entries that have already been
promoted to LTM.

**Evidence:** `auto-tune/mhall-svo/runs/005/harness.log`
**Observable:** `FRAME OSV > AHLM → S1` followed by 3 rounds of
`Cogitate misfit` on the same entry.

**Curriculum:** `curricula/mhall-svo-single.md`
**How to invoke:**
> /auto-tune — goal: stop the cogitator from processing entries that already resolved as S1 during the same rationalise batch. Use curricula/mhall-svo-single.md. Open session.

### ✅ Resolved — merged to main

---

## CI-2: Undersign entries without prior scaffolding go through slow path unreliably

Undersign entries like `{M: S}` (from `S = M`) go through candidate retrieval
as S3 against existing entries. Without prior scaffolding (identities
pre-grounded in earlier lessons), the slow path may not resolve them. This is
correct behaviour (undersign = connotate reversed), but the LLM cogitator
may not reliably bridge the gap. The question is whether the rationaliser's
graph expansion can resolve these without LLM assistance.

**Evidence:** `auto-tune/mhall-svo/runs/005/harness.log` — 15/18 satisfied
in single-lesson curriculum. The 3 unsatisfied are undersign/connotate entries.

**Curriculum:** `curricula/mhall-svo-single.md`
**How to invoke:**
> /auto-tune — goal: make the rationaliser resolve undersign entries like {M: S} through graph expansion without relying on the LLM cogitator. Use curricula/mhall-svo-single.md. Open session.

### ✅ Resolved — `auto-tune/rationaliser-graph-expansion` (2026-06-06)

Added Phase 3b to `KAgent.rationalise()` (`src/kalvin/agent.py`): when a
single-node entry references an unknown node, the rationaliser checks if the
signature participates in a grounded structure (via `is_s1`). If so, the
unknown node is grounded as a frame and the entry is accepted as S1.

Result: 18/18 entries resolved via fast path with zero LLM calls.
Branch: `auto-tune/rationaliser-graph-expansion`

---

## CI-3: promote_participating promotes unrelated STM entries

`promote_participating()` promotes ALL STM entries whose signatures appear in
the node union of the ratified query and candidate. This means connotate
entries like `{L: O}` get promoted to LTM during countersign ratification just
because their signature (L) appears in MHALL's node set. This causes them to
ground trivially on re-rationalisation rather than going through proper graph
expansion. The fix is to scope promotion to klines that actually participated
in the ratification event.

**Evidence:** `auto-tune/mhall-svo/runs/005/harness.log` —
`GROUND L > O → S1 (fast path)` where L>O is S3 connotate that grounded
because it was promoted to LTM as a side effect.

**Curriculum:** `curricula/mhall-svo-single.md`
**How to invoke:**
> /auto-tune — goal: scope promote_participating to only promote klines that structurally participated in the ratification event, not all STM entries sharing a signature. Use curricula/mhall-svo-single.md. Open session.

### ✅ Resolved — `auto-tune/promote-scope` (2026-06-06)

Scoped `promote_participating` (`src/kalvin/expand.py`) to only promote:
identity frames (`nodes=[]`), single-node entries (countersign/undersign),
and canonical compositions (canonization). Cogitator expansion proposals
(multi-node non-canonical klines) are excluded.

Result: 87% reduction in promotions (13 vs baseline's 101) while maintaining
the same 17/18 entry satisfaction rate.
Branch: `auto-tune/promote-scope`
