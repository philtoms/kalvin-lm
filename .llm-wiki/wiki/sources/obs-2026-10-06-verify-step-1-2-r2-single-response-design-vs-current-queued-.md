---
type: source
title: "Observation: Verify step 1.2: R2 single-response design vs current queued products"
tags:
  - trs
  - redesign
  - reentry
  - r2-products
  - recursion
  - diverge
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-verify-step-1-2-r2-single-response-design-vs-current-queued-
relevance: high
observed_at: 2026-10-06T15:14:09.050Z
source_context: Step-by-step rationalisation reentry redesign; verify step 1.2
---

# ⭐ Observation: Verify step 1.2: R2 single-response design vs current queued products

TWO NOTES from the reentry redesign walkthrough. (1) USER NOTE FOR LATER: neighbour-signature-ascending candidate order is NOT an algebraic requirement — added a NOTE comment at the sorted() site in rationalise() (model.py), to revisit. (2) VERIFY STEP 1.2 ANALYSIS (new R2 design stated, not yet implemented): new design = r2_Products returns ⟨F,N_F⟩,⟨U,N_O⟩,⟨O,N_U⟩ as a SINGLE response; no iteration; F is IGNORED ALWAYS; U is RECURSIVELY applied; O dropped when ⊥. Current build for the step-1.1 candidate ⟨SVO,{Subject}⟩ (F=Subject, U=Verb|Object, O=⊥, witness(F)=witness(U)=∅): r2_products returns TWO products ⟨Subject,∅⟩ and ⟨Verb|Object,∅⟩, both QUEUED on the LIFO stack, each popping to an R4 no-adjacency halt E=1 (hence step 1's 6 halts = 3 candidates × 2 products). DIVERGENCES of new vs current: (a) F-product — current queues it (here it just halts, but when witness(F)≠∅ it can do REAL WORK: re-enter with adjacency, reach S1, store — the new design kills this path unconditionally); (b) mechanism — current queues products as work entries interleaving with other candidates' work on the LIFO stack, new design recurses depth-first within the response (equivalent here, divergent in general — recursion is the redesign's stated goal); (c) O when ⊥ — both drop, same. Halts shrink under new design: 6 → 3 for step 1 (the F-products' R4 halts vanish).

*Relevance: high*
*Context: Step-by-step rationalisation reentry redesign; verify step 1.2*
*Tags: trs redesign reentry r2-products recursion diverge*

---
*Observed: 2026-10-06T15:14:09.050Z*
