---
type: source
title: "Observation: Layer 2 review: termination claims inverted, one-step relation undefined"
tags:
  - formalisation
  - rewrite-system
  - termination
  - derivation
  - review
status: observation
created: 2026-09-10
updated: 2026-09-10
slug: obs-2026-09-10-layer-2-review-termination-claims-inverted-one-step-relation
relevance: high
observed_at: 2026-09-10T13:16:24.194Z
source_context: Reviewing kalvin-symbolic.md §2 (rewrite system) for sense, gaps, and vocabulary
---

# ⭐ Observation: Layer 2 review: termination claims inverted, one-step relation undefined

Layer 2 (rewrite system, kalvin-symbolic.md §2) review findings: (1) the one-step relation is never defined — should be a memory-relative relation ⊢_{M,B} with expand/contract licensed by held canons and add/remove licensed by the targeting table keyed on fit(C(A,B)) at the current state; (2) ν_A ∖ ν_B is undefined (set vs multiset difference changes which moves exist, e.g. [a,a] vs [a]); contract's "subsequence" needs contiguous-exact-order vs multiset decision; (3) §2.3 termination claims inverted: add/remove oscillation is impossible (D = |ν_A∖ν_B| + |ν_B∖ν_A| multiset strictly decreases per targeting move — any targeting run terminates in ≤ D₀ steps, no monotonicity needed, computable bound), while the real hazard is expand/contract loops (σ-preserving, e.g. bc ↔ [b,c] against held canon bc:[b,c], constant band) — so "significance-monotone derivations terminate" is false as stated; true pair: targeting always terminates + witnessed moves need strategy bounds; (4) "expand/contract are composites of add/remove" is false — different invariant (σ-preserving vs σ-changing) and license source; replace is the only composite; (5) "retain: a terminal stands" is false for witnessed moves — s:[s] expands to s:[a,b] under held canon; terminals are targeting-closed, not rule-closed; retain duplicates the table's S1 "none" row; (6) licensed ≠ safe: licensed remove can empty ν_A → Unknown/ask stuck state (ab:[ab] vs a:[a]) — needs guard or explicit strategy-prunes statement; (7) §2.5 gloss "(nothing shared)" mis-assigns S4: nothing-shared is No-fit/S3 (σ-disjointness provable from node-disjointness); S4-as-Unknown means ν_B = [] (ask propagates); (8) "normalising moves" misnomer given renounced confluence — rename witnessed moves; (9) goal predicate is value-equality σ(ν_A) = σ(ν_B), not node-equality — explains Identity-relationship done with ν_A ≠ ν_B; success proves content-equality with held content, not A's own head-claim.

*Relevance: high*
*Context: Reviewing kalvin-symbolic.md §2 (rewrite system) for sense, gaps, and vocabulary*
*Tags: formalisation rewrite-system termination derivation review*

---
*Observed: 2026-09-10T13:16:24.194Z*
