---
type: source
title: "Observation: Goal vs target disambiguated: Def 22 selects goals, == gives the target"
tags:
  - terminology
  - goal
  - target
  - def22
  - ==
  - declaration
status: observation
created: 2026-09-23
updated: 2026-09-23
slug: obs-2026-09-23-goal-vs-target-disambiguated-def-22-selects-goals-gives-the-
relevance: critical
observed_at: 2026-09-23T08:04:52.684Z
source_context: Resolving the goal/target terminology conflation
---

# 🔴 Observation: Goal vs target disambiguated: Def 22 selects goals, == gives the target

Phil resolved the "goal-by-declaration" contradiction as a TERM CONFLATION: "goal" in Def 22 (the derivation's B, one party of a relationship construct — selected by coverage, γ-ordered) vs "goal" in the `a == b => c d` table row (goal-targeted TRAINING's target — the answer key). The == operator is not structural: it constructs no relationship between the ask and the target; it declares the training pair. Implemented: (1) doc — table row now says "the target b:[c,d]" / "S4 ask; target S1..."; §13 paragraph rewritten with the two-senses distinction ("The training target is not selected and never a candidate: for the ask it is given — the declared B of the training pair, ahead of the selected list"); §4's S4 sentence says "held against its declared target"; (2) code — hop.candidate_goals returns the declared target FIRST (given, never scored), coverage candidates γ-ordered after (previously the target was γ-scored into the candidate list, ranking last at γ=0 and risking the top-8 cut). Verified: mhall proposes+grounds MHALL:[Subject, Verb, Object] (chain now 2 hops, canonicalisation happens inside the target derivation — the ALL-canon goal detour is gone), wdmh unchanged, 90/90 tests.

*Relevance: critical*
*Context: Resolving the goal/target terminology conflation*
*Tags: terminology goal target def22 == declaration*

---
*Observed: 2026-09-23T08:04:52.684Z*
