---
type: source
title: "Observation: Refine-before-absorb: mechanism general, eager policy not — goal-resolution-directed is the criterion"
tags:
  - algebra
  - def17
  - refinement
  - policy
  - granularity
  - acquisition-depth
status: observation
created: 2026-09-14
updated: 2026-09-14
slug: obs-2026-09-14-refine-before-absorb-mechanism-general-eager-policy-not-goal
relevance: high
observed_at: 2026-09-14T06:29:23.777Z
source_context: Designing the Def 17 refinement amendment for kalvin-simplified.md
---

# ⭐ Observation: Refine-before-absorb: mechanism general, eager policy not — goal-resolution-directed is the criterion

Phil asked whether refine-before-absorb (the step needed to reproduce the documented w:[a,l,l] absorption) is generalisable and whether it applies to every derivation. Resolution, verified by a two-policy probe run: the MECHANISM is fully general (it is just the forward canon replace — a witnessed move licensed by memory alone, T2-bounded; no new rule). The eager POLICY is not generalisable: (1) cost invariance — δ discounts D̄ and Ĥ identically ("both are denominated in edges"), so the expansion edge costs 2^-1 whenever crossed; eager absorption (w:[a,l,l], Ĥ=9/5, γ≈0.287) equals lazy absorption plus a later in-derivation expansion in total γ, and lazy (w:[all], Ĥ=2/3, γ≈0.63, same done by value equality) is strictly cheaper when nothing ever needs the finer granularity — which is exactly the Mary case (nothing in the doc memory applies at bare a or l). (2) Permanence — absorbed klines carry their depth and consuming composes (§11), so eager refinement stamps +1 on every future consumption of the reusable correspondence; lazy pays +1 only per deriving derivation that expands. §11's granularity-monotonicity (gratuitous expansion detectable) exists to catch exactly the eager policy. The generalisable criterion is demand-driven: the composed correspondence delivers the excess at the GOAL'S WITNESS RESOLUTION of it ("absorption speaks the goal's language for the excess") — reproduces the documented example (B holds {a,l} bare), no-ops when the goal holds its excess coarsely; it is canonicalisation's doctrine read in mirror (contract to compose, expand to deliver). It applies only at slot-walk absorptions, never mid-derivation, never to non-excess content. Probe gained a refine_at_absorption flag + policy-comparison run. Proposed Def 17 amendment updated accordingly + T2 keying fix; awaiting Phil's go-ahead.

*Relevance: high*
*Context: Designing the Def 17 refinement amendment for kalvin-simplified.md*
*Tags: algebra def17 refinement policy granularity acquisition-depth*

---
*Observed: 2026-09-14T06:29:23.777Z*
