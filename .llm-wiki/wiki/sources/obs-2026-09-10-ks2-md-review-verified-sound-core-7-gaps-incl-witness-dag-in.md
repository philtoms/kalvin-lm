---
type: source
title: "Observation: ks2.md review: verified sound core, 7 gaps incl. witness-DAG invariant and γ underdetermined"
tags:
  - formalisation
  - algebra
  - ks2
  - review
status: observation
created: 2026-09-10
updated: 2026-09-10
slug: obs-2026-09-10-ks2-md-review-verified-sound-core-7-gaps-incl-witness-dag-in
relevance: high
observed_at: 2026-09-10T14:30:10.304Z
source_context: Reviewing docs/ks2.md formal algebra draft for sense, gaps, and implementability
---

# ⭐ Observation: ks2.md review: verified sound core, 7 gaps incl. witness-DAG invariant and γ underdetermined

Reviewed docs/ks2.md (2nd formalisation draft, 4 tracts). Verified sound: Def 10 8-case partition complete/disjoint; T1 targeting bound computable and correct (D decreases 1/move); worked example correct. Found gaps: (G1) witness-DAG invariant asserted not enforced — Def 10 case 3 admits self-containing canons (a:[a,a], ab:[a,ab]) which break expand well-foundedness; absorb (§10) unspecified and could violate the invariant. (G2) §7 replace-interior claim false — remove-all-then-add passes through S4-stuck empty interior; adds must precede last remove. (G3) Def 15 "stuck = no licensed move" conflicts with witnessed moves being licensed at all bands. (G4) γ underdetermined (two qualitative requirements don't pin a family); unweighted mean-of-α is granularity-sensitive so witnessed moves change γ at constant content even without δ. (G5) ask atom has no operational semantics (declared S4 vs structural fit precedence undefined). (G6) unclear whether an Unknown can be a grounded target (Def 16 vs S4 licensing row). (G7) factual error in §10 example: abc:[a] vs x:[c,a] gives C = a:[c,a] = two-node Overfit, not Denotation. Conformance drift: retain removed (CONTEXT.md table still lists it); glossary Candidates "at least S2" vs Def 16 signature-in-node + S3 routing; engine is_connotation (sig⊆node, S2 shapes) diverges from ks2 Connotation (case 4/S3). Acid test verdict: tracts 1–2 implementable verbatim; tracts 3–4 blocked on absorb semantics, progressive-path STM writes, γ formula, ask precedence.

*Relevance: high*
*Context: Reviewing docs/ks2.md formal algebra draft for sense, gaps, and implementability*
*Tags: formalisation algebra ks2 review*

---
*Observed: 2026-09-10T14:30:10.304Z*
