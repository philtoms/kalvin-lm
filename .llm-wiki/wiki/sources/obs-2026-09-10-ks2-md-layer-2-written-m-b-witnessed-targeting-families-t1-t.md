---
type: source
title: "Observation: ks2.md Layer 2 written: ⊢_{M,B}, witnessed/targeting families, T1/T2 termination"
tags:
  - formalisation
  - rewrite-system
  - ks2
  - derivations
  - termination
status: observation
created: 2026-09-10
updated: 2026-09-10
slug: obs-2026-09-10-ks2-md-layer-2-written-m-b-witnessed-targeting-families-t1-t
relevance: medium
observed_at: 2026-09-10T13:32:40.413Z
source_context: Writing Layer 2 (rewrite system) into docs/ks2.md per the review
---

# 🔍 Observation: ks2.md Layer 2 written: ⊢_{M,B}, witnessed/targeting families, T1/T2 termination

Applied the Layer 2 review to docs/ks2.md as new §§6–9 (defs 12–15), renumbering the scope section to §10 and updating the status line to "Layers 1–2". §6 Derivations: memory-relative one-step relation ⊢_{M,B} written out (expand/contract/remove/add/replace), multiset-wise membership and difference convention (order used only by contract's pattern match), no rule reads a kline's own fit. §6 Def 14: licensing table keyed on fit(C(A,B)) at the current state; S3 replace forced total by node-disjointness; S4 = ν_B = [] (ask propagates); witnessed moves need no table license. §7: the two families stated by invariant — witnessed moves preserve σ(ν_A) (licensed by M), targeting moves change σ(ν_A) toward σ(ν_B) (licensed by B, difference-set nodes only); replace is the only composite with licensed interiors; model-theoretic congruence remark (held canons generate ≈_M, targeting operates on representatives); terminals are targeting-closed not rule-closed; no retain move. §8: done (value-equality σ(ν_A) = σ(ν_B), not node-equality) vs stuck (no target selected vs ν_B = [] ask); licensed ≠ safe with the ab:[ab] vs a:[a] dead-end example; termination as two true statements — (T1) targeting runs terminate in ≤ D₀ steps (multiset symmetric difference, decreases by exactly one per move, computable bound), (T2) witnessed moves σ-preserving and cyclical, so mixed-derivation termination is a strategy property; confluence renounced; decidability via finite V + finite acyclic M. §9: done proves content-equality with held content, not A's own head-claim; solver reading restated; feedback strategy-level; worked micro-example (mall) with D₀ = 2 achieved exactly. §10 scope updated: candidate selection (where B comes from), budgets with D₀ as natural unit, tiers, graded distance, KScript, countersigning.

*Relevance: medium*
*Context: Writing Layer 2 (rewrite system) into docs/ks2.md per the review*
*Tags: formalisation rewrite-system ks2 derivations termination*

---
*Observed: 2026-09-10T13:32:40.413Z*
