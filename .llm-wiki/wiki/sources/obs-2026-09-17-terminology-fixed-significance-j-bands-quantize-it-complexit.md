---
type: source
title: "Observation: Terminology fixed: significance = J (bands quantize it); complexity = 1−δ^(D̄+Ĥ); γ = composite"
tags:
  - kalvin-algebra
  - context-md
  - definition-20
  - significance
  - complexity
  - gamma
  - terminology
status: observation
created: 2026-09-17
updated: 2026-09-17
slug: obs-2026-09-17-terminology-fixed-significance-j-bands-quantize-it-complexit
relevance: critical
observed_at: 2026-09-17T11:13:03.633Z
source_context: "Terminology fix: significance vs complexity split; structural significance absorbed"
---

# 🔴 Observation: Terminology fixed: significance = J (bands quantize it); complexity = 1−δ^(D̄+Ĥ); γ = composite

Terminology restructure landed across docs/kalvin-algebra.md and CONTEXT.md (user's directive: the depth-laden γ was never "significance" — it is complexity; significance is the Jaccard-anchored understanding measure absorbing "structural significance"). New vocabulary, normative in Def 20 (retitled "Significance and complexity"): SIGNIFICANCE = J(σ(ν_A),σ(ν_B)) — path-independent understanding, 1.0 exactly at done, 0 at disjointness, γ at entry depths; bands are its QUANTIZATION (S1=1.0, S2=overlap short of equality, S3=zero overlap, S4 off-scale vacuous) — "structural significance" retired as redundant, kept only as a historical note + CONTEXT _Avoid_. COMPLEXITY = 1 − δ^(D̄+Ĥ) — the work axis, independent of significance, prices moving between embedded concepts, never selects a band (formula choice flagged to user: normalized complement so the example decimals p1 sig 0.8/complexity 0.2 vs p2 0.9/0.4 read as direct cost). γ = significance × (1 − complexity) — the composite, formula unchanged; compares derivations of equal significance, rate-of-change steers strategy. Dependent updates: Def 16 done = "significance 1.0"; §9 closing paragraph; §0 intro (significance value + complexity value); 0.3 Measurement; Def 22 candidate ordering ("γ, not band, sets the order"); invariants restated (band-consistency now a property of significance; granularity invariants in complexity terms); Exchange (proposal travels at significance, complexity stays as acquisition record); §10 example (γ 0.29 = significance 1.0 at complexity 0.71). CONTEXT: Significance rewritten, Complexity entry added, Band = "quantization of significance", Done/Abandoned/Cogitation/KValue/== updated. Code comments aligned (engine._propose docstring, DerivationResult field comments); identifiers unchanged — j1 IS significance, result.gamma IS γ composite; a code rename (j1→significance etc.) is a future refactor. Future payoff noted by user: S2 proposal selection can trade significance vs complexity as independent axes. Tests 69/69.

*Relevance: critical*
*Context: Terminology fix: significance vs complexity split; structural significance absorbed*
*Tags: kalvin-algebra context-md definition-20 significance complexity gamma terminology*

---
*Observed: 2026-09-17T11:13:03.633Z*
