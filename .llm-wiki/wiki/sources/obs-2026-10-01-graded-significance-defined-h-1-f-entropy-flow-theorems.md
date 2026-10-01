---
type: source
title: "Observation: Graded significance defined: H = 1 − μ(F)/μ(σ∨â), entropy flow theorems"
tags:
  - trs
  - graded-significance
  - entropy
  - definition-13
  - jaccard
  - r2-theorem
status: observation
created: 2026-10-01
updated: 2026-10-01
slug: obs-2026-10-01-graded-significance-defined-h-1-f-entropy-flow-theorems
relevance: high
observed_at: 2026-10-01T13:24:53.072Z
source_context: Defining graded significance as entropy in the TRS spec
---

# ⭐ Observation: Graded significance defined: H = 1 − μ(F)/μ(σ∨â), entropy flow theorems

Graded significance DEFINED as entropy (amendment 12, docs/trs-significance.md Definition 13): H(C) = 1 − μ(F)/μ(σ ∨ â) — surprise = the unexplained fraction of the pairing; g(C) = μ(F)/μ(σ ∨ â) is its complement ('the more significant, the less entropy'). User framing: measure of cognitive effort, entropy-directed; no surprise in canon or identity at S1 (BOTH shapes — this kills any witness-quality reading of the grade, consistent with the parked I-2 finding), 100% surprise in S4. KEY FORMAL FACTS: (1) g is exactly the engine's J (Jaccard), independently reached — added to §6 mapping table; (2) entropy does NOT separate S3 from S4 (both g=0, H=1) — both carry full surprise, the band splits them by repairability (R3 vs terminal), the empty adjacency is structural (arrival condition, not a point on the correspondence axis); (3) entropy flow under the rules — R1 fires exactly at H=0 (the proof of full understanding IS the discharge of all surprise — coheres with the R1 rewrite); R2 NEVER REDUCES entropy: Σ μ(head)·H(product) ≥ μ(σ∨â)·H(C), equality iff the fit head is held or ⊥ (misfit products always H=1 — disjoint or empty; an unheld fit head degrades to R4 at full surprise) — projection alone creates no understanding; R3 discards the surprised claim (prior signature dropped — entropy eliminated by abandonment, not explanation), product self-canonical H=0, surprise re-enters only via Definition 9 re-pairing; R4 = full-surprise arrival. Spec mechanics: μ RESTORED to Definition 1 (measure: μ(⊥)=0, additive on disjoint parts; word-bit popcount; 'μ weighs, it does not decompose'), §4 preamble rewritten (both species now defined), D9 amended, graded-significance bullet REMOVED from §7. Implementation: mu(), significance() → Fraction (exact), entropy() = 1−g in src/trs/model.py, exported. Tests: endpoint units (canon AND identity H=0, S4 H=1, S2 interior with g(⟨AB,{A}⟩)=1/2), band-grade correspondence, R2 never-reduces-entropy theorem verified over random S2 scenarios (with equality-iff-fit-held biconditional). 141 tests passing. Entropy work UNCOMMITTED.

*Relevance: high*
*Context: Defining graded significance as entropy in the TRS spec*
*Tags: trs graded-significance entropy definition-13 jaccard r2-theorem*

---
*Observed: 2026-10-01T13:24:53.072Z*
