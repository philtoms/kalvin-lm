---
type: source
title: "Observation: Worked example fixed: queue enters as [w,d,m,h], not [w,dh,m]"
tags:
  - kalvin-algebra
  - worked-example
  - docs
  - correction
status: observation
created: 2026-09-12
updated: 2026-09-12
slug: obs-2026-09-12-worked-example-fixed-queue-enters-as-w-d-m-h-not-w-dh-m
relevance: high
observed_at: 2026-09-12T12:32:54.334Z
source_context: Correcting the worked example in docs/kalvin-algebra.md
---

# ⭐ Observation: Worked example fixed: queue enters as [w,d,m,h], not [w,dh,m]

User identified that docs/kalvin-algebra.md §9's worked example (WDMH => MHALL) presumed the queue arrived as wdmh:[w,dh,m] — the verb phrase "did have" already composed as node dh. Corrected to the honest entry state wdmh:[w,d,m,h] (four bare word-bits). The derivation now includes the necessary gather+contract steps: an ordering move makes d,h contiguous (strategy degree of freedom, Def 12), then reverse on canon dh:[d,h] contracts [d,h]⇉[dh] (witnessed, σ-preserving, no targeting budget) before the denotation shed dh⇉[h] can fire. No held kline has signature d, so there is no other route to shed it. Relationship C, Δ₀=4, end state, Ĥ=9/5, and T1's two-replace count are unchanged (C reads σ(ν_A), which is wdmh either way). Side benefit: the example now exercises §7/§8's witnessed granularity-exposure machinery, previously unillustrated. Same fix applied to the expanded version in docs/kalvin-for-agents.md (§9 table, steps renumbered 1–4). Also corrected an arithmetic slip in for-agents' pricing: J at [w,m,h] is 2/5 (not 1/2) per J(x,y)=|x∧y|/|x∨y|; γ dip is 0.4→≈0.29 at δ=½.

*Relevance: high*
*Context: Correcting the worked example in docs/kalvin-algebra.md*
*Tags: kalvin-algebra worked-example docs correction*

---
*Observed: 2026-09-12T12:32:54.334Z*
