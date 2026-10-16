---
type: source
title: "Observation: TRS candidate consolidation implemented — earned state removed"
tags:
  - trs
  - rationalise
  - redesign
  - candidate-consolidation
  - earned-removal
status: observation
created: 2026-10-16
updated: 2026-10-16
slug: obs-2026-10-16-trs-candidate-consolidation-implemented-earned-state-removed
relevance: high
observed_at: 2026-10-16T13:17:44.428Z
source_context: "TRS rationalise redesign, step 1: candidate consolidation"
---

# ⭐ Observation: TRS candidate consolidation implemented — earned state removed

Redesign step 1 implemented in src/trs/model.py (branch trs2, uncommitted). candidates(c_e, nbs) helper: covered witnesses (nb.adj non-empty and can(nb.adj) ⪯ c_e) aggregate into ONE candidate ⟨c_e, ⋃adj⟩; uncovered witnesses stand alone; terminal (empty-adj) neighbours consolidate into a single ⟨c_e,∅⟩ S4 ask (dedup of the old N duplicate acquire asks). earned parameter deleted from rationalise; S2 split's fit kline ⟨F,N_F⟩ stores in its own right via admit() when S1-canonical, else re-enters as its own ask; product halts propagate to outer ends but product stores do NOT (restores 2f0934b semantics). admit(): d.add + report once per run even on idempotent re-store. Fixed the 2 tests that were already failing on HEAD (test_r1_stores_the_s1_candidate, test_canon_memo_is_per_run — broken by b4355e0's earned threading). mhall.ks both-steps: OLD 9 stores/170 ends → NEW 2 stores/6 ends. 6 new tests added. All 166 pass.

*Relevance: high*
*Context: TRS rationalise redesign, step 1: candidate consolidation*
*Tags: trs rationalise redesign candidate-consolidation earned-removal*

---
*Observed: 2026-10-16T13:17:44.428Z*
