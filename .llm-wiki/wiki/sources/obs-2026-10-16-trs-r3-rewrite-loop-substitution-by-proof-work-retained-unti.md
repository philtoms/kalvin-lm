---
type: source
title: "Observation: TRS R3 rewrite loop — substitution by proof, work retained until R1"
tags:
  - trs
  - rationalise
  - redesign
  - r3
  - substitution-by-proof
  - rewrite-loop
  - licensed-claim
status: observation
created: 2026-10-16
updated: 2026-10-16
slug: obs-2026-10-16-trs-r3-rewrite-loop-substitution-by-proof-work-retained-unti
relevance: high
observed_at: 2026-10-16T15:30:55.332Z
source_context: "TRS rationalise redesign, step 3: R3 rewrite loop with proof substitution"
---

# ⭐ Observation: TRS R3 rewrite loop — substitution by proof, work retained until R1

Redesign step 3 implemented in src/trs/model.py (uncommitted): R3 rewrite loop. Problem: S3 candidate MHALL:[Verb] dead-ended at like-for-like — node Verb is never a held head (only a witness of hadVerb:[Verb], SVO:[S,V,O]), so forward substitution can never fire, and each candidate's verified work was thrown away (no cross-link after earned removal). Mechanism: (1) candidates() now returns (kline, suppliers) provenance pairs; terminal neighbours halt once pre-loop; (2) S3 substitutes forward (held heads expand, r3_substitute) else BY PROOF — the supplying witness node ↦ the supplier's head (the neighbour that established the candidate is the proof-link); (3) rewrites accumulate into `work`, suppliers leave `consumed`, rounds regenerate the candidate list from live domain until empty/no-progress — the while-loop IS the reentry (avoids seen-memo conflict on same canon); every processed candidate consumes its supplier → monotone termination without relying on seen; (4) exhaustion-with-work admits Kline(c_e, work) — R1 when S1 (exact cover: test_r3_loop_runs_until_r1), a licensed claim otherwise (overfit accumulated, e.g. MHALL claim). Old product-returns-stores licensing and identity-mint path (⟨AB,{AB}⟩) replaced by the accumulated claim. mhall.ks step 2 now stores MHALL:[a,little,lamb,MarySubject,hadVerb,ALLQuery] — the SVO-structured goal claim — plus Mary|had:[MarySubject,hadVerb] and ALL claim via product recursion; 2 ends total (E=1 starved-fit acquires). Full trajectory: HEAD 9 stores/170 ends → step1 2/6 → step3 3/2. 3 tests rewritten, 3 added (by-proof, cross-link accumulation, loop-until-R1, duplicate-product seen-bound). 170 pass.

*Relevance: high*
*Context: TRS rationalise redesign, step 3: R3 rewrite loop with proof substitution*
*Tags: trs rationalise redesign r3 substitution-by-proof rewrite-loop licensed-claim*

---
*Observed: 2026-10-16T15:30:55.332Z*
