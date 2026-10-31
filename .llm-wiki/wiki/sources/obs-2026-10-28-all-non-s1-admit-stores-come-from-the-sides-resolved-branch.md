---
type: source
title: 'Observation: All non-S1 _admit stores come from the "sides resolved" branch'
tags:
  - trs
  - r1
  - admit
  - nonsense-stores
  - cogitator
status: observation
created: 2026-10-28
updated: 2026-10-28
slug: obs-2026-10-28-all-non-s1-admit-stores-come-from-the-sides-resolved-branch
relevance: high
observed_at: 2026-10-28T12:17:59.963Z
source_context: Investigating unusual _admit results from dev/dialogue/trs_harness.py
---

# ⭐ Observation: All non-S1 _admit stores come from the "sides resolved" branch

Instrumenting Cogitator._admit on mhall.ks: 2311 stores, 1705 (74%) are non-S1 (sig != can(adj)) — ALL from the "sides resolved — it grounds" branch at src/trs/model.py:358. The S1-checked branch (line 362, `cand.sig == can(cand.n_f)`) admits 3, all canons. Mechanism of Verb:[Subject,Verb,Object]: entry ⟨Verb,{Verb}⟩ vs neighbour ⟨SVO,{S,V,O}⟩ gives F=Verb, O=Subject|Object, n_o={S,O}; since signature Subject|Object is grounded in D (⟨SO,{S,O}⟩ held), the loop merges n_o into n_f and admits ⟨Verb,{V,S,O}⟩ with the signature unmoved — adjacency ballooned over the claim. 1676 of the 1705 are S2 "claim smaller than its witnesses", 29 S3 disjoint, at least one empty-adjacency store (Query:[]) which the spec routes to R4 halt. Per docs/trs-significance.md R1 stores fire at S1 (or R3 licensed claim C ∨ earned which COMPOSES the signature); R2 says split and re-enter — "no product return stores the candidate that split". r2_products/r3_substitute are never called by the Cogitator (dead in the run path); tests/test_trs_model.py is stale (imports module-level rationalise, fails at import).

*Relevance: high*
*Context: Investigating unusual _admit results from dev/dialogue/trs_harness.py*
*Tags: trs r1 admit nonsense-stores cogitator*

---
*Observed: 2026-10-28T12:17:59.963Z*
