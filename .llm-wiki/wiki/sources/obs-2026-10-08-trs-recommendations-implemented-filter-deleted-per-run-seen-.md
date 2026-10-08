---
type: source
title: "Observation: TRS recommendations implemented: filter deleted, per-run seen memo, r2_products3 tested"
tags:
  - trs
  - rationalise3
  - r2-reconstruction
  - seen-memo
  - tests
status: observation
created: 2026-10-08
updated: 2026-10-08
slug: obs-2026-10-08-trs-recommendations-implemented-filter-deleted-per-run-seen-
relevance: high
observed_at: 2026-10-08T22:27:31.200Z
source_context: Implementing the three recommendations from the trs-significance resync
---

# ⭐ Observation: TRS recommendations implemented: filter deleted, per-run seen memo, r2_products3 tested

All three flagged recommendations implemented on branch trs (uncommitted, on top of the staged rationalise3 rewrite). (1) Vacuous like-for-like filter DELETED from rationalise3's candidate formation — `[nb.sig] != nb.adj` (list vs frozenset, always True) had never been active; Definition 9 (every neighbour forms a candidate) and §6's collapse of Identity/Canon make identity witnesses ordinary canonical klines, and the mhall trace shows `Mary:[Mary]` productively earning its F-side. (2) Module-global `seen` replaced by a `seen: set[KNode] | None = None` parameter threaded through recursion — per-run memo, no cross-run contamination; the trs_harness's goal and ask steps no longer share the memo (each presented entry is its own run). (3) r2_products3 (the reconstruction R2) now exported from trs and tested: unit tests pin cross-wise pairing, crossing-node starvation (BC crossing node starves both F- and O-sides → ⟨B,∅⟩,⟨A,∅⟩,⟨C,{A}⟩), pure-overfit head skip; property tests pin the entropy theorem under reconstruction (Σμ(head)·H ≥ μ(σ∨â)·H, equality iff can(N_F) = F) and per-product well-formedness; rationalise3 pinned for the R4 no-adjacency guard and per-run memo. Suite 159 → 166 green; probe + harness green. Witness-variant r2_products and its tests kept (still used by the LIFO rationalise).

*Relevance: high*
*Context: Implementing the three recommendations from the trs-significance resync*
*Tags: trs rationalise3 r2-reconstruction seen-memo tests*

---
*Observed: 2026-10-08T22:27:31.200Z*
