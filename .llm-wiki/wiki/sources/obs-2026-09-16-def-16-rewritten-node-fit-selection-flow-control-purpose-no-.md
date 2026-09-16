---
type: source
title: "Observation: Def 16 rewritten: node-fit selection, flow-control purpose, no pre-supposed derivation results"
tags:
  - docs
  - kalvin
  - def16
  - selection
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-def-16-rewritten-node-fit-selection-flow-control-purpose-no-
relevance: high
observed_at: 2026-09-16T08:58:32.011Z
source_context: Reworking Def 16 (Selection) in docs/kalvin-algebra.md
---

# ⭐ Observation: Def 16 rewritten: node-fit selection, flow-control purpose, no pre-supposed derivation results

Rewrote Def 16 per Phil's critique: the old condition (candidate's signature t must occur as a node in ν_A) pre-supposed the result of derivation — a signature present as a node is typically an arrival the derivation itself had to materialise (e.g. node dh exists only after a contract), which is why canonicalisation needed a separate "survey" channel to bootstrap Step 1. New condition: a node of the candidate's witness occurs among the current queue's nodes (n ∈ ν_K and n ∈ ν_A, multiset-wise per Def 12) — fit between nodes, never signatures, grounded in Def 5 (the queue holds a piece of the candidate's claim at the state's own resolution). Purpose stated first: selection controls the order and scope of candidate flow into the derivation model — a gate, not a licence; Defs 13–14 apply or decline. Ratchet re-based on nodes ("held klines sharing one of those nodes flow in next"); "the path is the guard" kept. Independence paragraph reworked: node fit ⇒ content overlap (shared node carries atoms in both), not conversely (below-node-resolution overlap: A=abc:[abc] vs K=bc:[b,c]), neither fixes the band (K=x:[a] vs A=bc:[a] selectable, disjoint, S3). Design choice made: shared-node fit rather than full-witness occurrence — the appendix's m:[m,all] at [m,h] only flows in under shared-node. Appendix line 1346 updated ("its node m occurs"); CONTEXT.md Candidates entry updated in same change. Both worked examples cohere: §9 Step 1's dh:[d,h] now selectable at entry. Not committed.

*Relevance: high*
*Context: Reworking Def 16 (Selection) in docs/kalvin-algebra.md*
*Tags: docs kalvin def16 selection*

---
*Observed: 2026-09-16T08:58:32.011Z*
