---
type: source
title: "Observation: Verify step 1.2: R2 consumption diverges — recursion on U, F accumulates"
tags:
  - trs
  - redesign
  - verify-step-1-2
  - r2
  - recursion
  - witness-accumulation
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-verify-step-1-2-r2-consumption-diverges-recursion-on-u-f-acc
relevance: high
observed_at: 2026-10-06T16:05:44.099Z
source_context: Step-by-step rationalisation reentry redesign; verify step 1.2
---

# ⭐ Observation: Verify step 1.2: R2 consumption diverges — recursion on U, F accumulates

VERIFY STEP 1.2 — R2 single-response vs current mechanics (instrumented against primed mhall domain, goal step). CURRENT BUILD on first candidate ⟨SVO,{Subject}⟩ (S2, F=Subject, U=Verb|Object, O=⊥): r2_products already returns the products as ONE response (matches the new design at the function boundary) — ⟨Subject∧SVO, witness(F)⟩ and ⟨VO, witness(O)⟩, third dropped (O=⊥). BUT witness(F)=∅ and witness(O)=∅: Subject is NOT a held head (the kline with adjacency {Subject} has head MarySubject — witness is exact-signature lookup), so both products carry empty adjacency, queue as separate work entries, pop, and die as 'R4: entry kline with no adjacency'. Goal run total: 6 R4 ends, 0 stores. U=Verb|Object NEVER meets a neighbour. THREE DIVERGENCES of the new design: (1) U is RATIONALISED (own neighbour lookup, own decomposition: VO → hadVerb → F₂=Verb, U₂=Object → recurse → Object via QueryObject → S1 stored always) — not re-entered as an entry carrying N_O=∅; (2) F ACCUMULATES WITNESS ON UNWIND — stored sigs ⊆ residual (Object⊆VO, then VO⊆SVO) get added to F: [Subject] → [Verb,Object] → [Subject,Verb,Object], each enriched F pushed and iterated to S1 stores (VO:[Verb,Object] then SVO:[Subject,Verb,Object]) — current build has zero accumulation, independent per-product traces; (3) products consumed as one response (F accumulator, U recursion, O/N-cross pairing terms) vs queue.append of separate work entries with per-entry openers. Expected new-design stores for the goal: Object:[Object], VO:[Verb,Object], SVO:[Subject,Verb,Object]. User protocol continues step-by-step, NO implementation yet.

*Relevance: high*
*Context: Step-by-step rationalisation reentry redesign; verify step 1.2*
*Tags: trs redesign verify-step-1-2 r2 recursion witness-accumulation*

---
*Observed: 2026-10-06T16:05:44.099Z*
