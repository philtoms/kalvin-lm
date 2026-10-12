---
type: source
title: "Observation: R2 same-side pairing settled; spec §5 resynced"
tags:
  - trs
  - r2
  - spec
status: observation
created: 2026-10-12
updated: 2026-10-12
slug: obs-2026-10-12-r2-same-side-pairing-settled-spec-5-resynced
relevance: high
observed_at: 2026-10-12T16:05:47.672Z
source_context: Resyncing docs/trs-significance.md §4/§5 to adjusted R2 handling in rationalise3
---

# ⭐ Observation: R2 same-side pairing settled; spec §5 resynced

R2 semantics settled (user-confirmed) and spec resynced: (1) pairing is same-side — products are ⟨F,N_F⟩ (earned, not re-entered), ⟨U,N_U⟩ (entry's U-side), ⟨O,N_O⟩ (candidate's O-side); cross-wise pairing withdrawn. (2) Split return dropped: R2 residual products ground for their own sake, no product return stores the parent candidate; only R3's re-entry returning with stores stores it. (3) R1 stores C ∨ earned — the fit kline merges (sig and adj) into an earned composition riding down the recursion, spent at the store. Code: src/trs/model.py r2_products3 returns (products, k_f); rationalise3 threads earned. Legacy r2_products (cross-wise, witness lookup) remains for the old stack runner.

*Relevance: high*
*Context: Resyncing docs/trs-significance.md §4/§5 to adjusted R2 handling in rationalise3*
*Tags: trs r2 spec*

---
*Observed: 2026-10-12T16:05:47.672Z*
