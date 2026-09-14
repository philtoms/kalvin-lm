---
type: source
title: "Observation: Def 17 restated: two-party slots, anchor, refinement; earlier work committed"
tags:
  - kalvin-algebra
  - def17
  - slot-derivation
  - reverse-derivation
  - commit
status: observation
created: 2026-09-14
updated: 2026-09-14
slug: obs-2026-09-14-def-17-restated-two-party-slots-anchor-refinement-earlier-wo
relevance: high
observed_at: 2026-09-14T14:55:19.523Z
source_context: Committing earlier work and drafting Def 17 symmetric slot derivation
---

# ⭐ Observation: Def 17 restated: two-party slots, anchor, refinement; earlier work committed

Drafted the Def 17 restatement in docs/kalvin-algebra.md (uncommitted, for review), implementing the "2a" design: slots defined on both parties (underfit slot = node of ν_A carrying a u-atom; overfit slot = node of ν_B carrying an o-atom; duality via C(B,A) stated in the def). Walks from ν_B are goal-less, occurrence-licensed as before, ending at arrival in σ(ν_A) — the "anchor" (new term). Unified write rule: head = A-side end (slot fixed at departure, anchor discovered on arrival), witness = A-side end's atoms shared with the goal + B-side end's overfit-covering nodes. Verified it reproduces the §9 worked example exactly (w:[a,l,l], depth 3). The probe's pending refine-gap is closed in-def: "Arrival is not absorption" paragraph sanctions post-arrival Canon refinement to the consuming resolution (goal's witness for overfit, ν_A's nodes for anchor), each refinement edge counted; §9 renamed "absorption expansion" → "refinement edge". Consequential edits: Def 16 "not selected for replacement … may seed slot walks"; Def 15 stuck clause two-sided. CONTEXT.md Slot + Candidates entries updated. Done unchanged (σ(ν_A)=σ(ν_B)); T1 untouched. Also committed earlier work: 62ed466 (underfit/overfit rename), 108ba6e (cogitator connotateY→connotate + dead canonicalise removal + completed probe sweep — 3 probes + 2 copies inside coverage_d missed by the earlier sweep). Finding: probe_what_hub, probe_reverse_edges, probe_what_consumed and coverage_b/c/d/rule were already broken at HEAD by an older KPath refactor (tuple-unpack vs KPath yields) — stale, candidates for deletion/repair.

*Relevance: high*
*Context: Committing earlier work and drafting Def 17 symmetric slot derivation*
*Tags: kalvin-algebra def17 slot-derivation reverse-derivation commit*

---
*Observed: 2026-09-14T14:55:19.523Z*
