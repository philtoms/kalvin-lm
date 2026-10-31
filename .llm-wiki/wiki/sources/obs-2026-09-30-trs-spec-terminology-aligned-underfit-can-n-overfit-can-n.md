---
type: source
title: "Observation: TRS spec terminology aligned: underfit=σ∖can(N), overfit=can(N)∖σ"
tags:
  - trs
  - significance
  - terminology
  - underfit
  - overfit
  - alignment
status: observation
created: 2026-09-30
updated: 2026-09-30
slug: obs-2026-09-30-trs-spec-terminology-aligned-underfit-can-n-overfit-can-n
relevance: high
observed_at: 2026-09-30T16:31:23.924Z
source_context: Reversing overfit/underfit terminology in docs/trs-significance.md to align with kalvin-algebra.md
---

# ⭐ Observation: TRS spec terminology aligned: underfit=σ∖can(N), overfit=can(N)∖σ

Terminology reversal applied to docs/trs-significance.md: overfit/underfit now follow kalvin-algebra.md Definition 9. Underfit U = σ ∖ can(N) (signature-side: claimed by the signature, not supplied by the adjacency; flow signature→adjacency); overfit O = can(N) ∖ σ (adjacency-side: supplied by the adjacency, not claimed by the signature; flow adjacency→signature). The mnemonic letters swapped with the names so U/O still match their terms, and N_U/N_O follow. All derived statements updated consistently: decomposition σ = F ∨ U, â = F ∨ O; P-2 containments F,U ⪯ σ and F,O ⪯ â with U ∧ â = ⊥ and O ∧ σ = ⊥. Per user instruction the change reads as never-broken: D1 restated as the aligned convention (no amendment trace), §0.2 retitled plain 'Terminology mapping' with identical-readings rows (no warning), the Def 12 inversion parenthetical removed, §7 residuals row now 'identical'. The four-round amendment note was left untouched as it never mentioned the inversion. Note for future sessions reviewing data/trs.md (the design dialogue): the dialogue itself still uses the OLD orientation (overfit = signature-heavy); the spec is now the authority and diverges from the dialogue on these two names.

*Relevance: high*
*Context: Reversing overfit/underfit terminology in docs/trs-significance.md to align with kalvin-algebra.md*
*Tags: trs significance terminology underfit overfit alignment*

---
*Observed: 2026-09-30T16:31:23.924Z*
