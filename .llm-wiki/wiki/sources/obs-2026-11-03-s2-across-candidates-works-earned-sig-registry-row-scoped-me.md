---
type: source
title: "Observation: S2 across candidates works — earned-sig registry + row-scoped memo ground both scripts"
tags:
  - trs
  - cogitator
  - s2-alignment
  - grounded-registry
  - row-memo
  - trial
status: observation
created: 2026-11-03
updated: 2026-11-03
slug: obs-2026-11-03-s2-across-candidates-works-earned-sig-registry-row-scoped-me
relevance: high
observed_at: 2026-11-03T23:21:30.432Z
source_context: "Trial: S2 alignment across candidates via earned-sig registry + row-scoped memo"
---

# ⭐ Observation: S2 across candidates works — earned-sig registry + row-scoped memo ground both scripts

S2-across-candidates trial SUCCEEDED (user's idea: like S3 alignment across parents, U/O S2 alignment across different candidates is acceptable; U fork always precedes O fork). Two mechanisms landed on top of the user's recode: (1) closes register their earned sig in a global self.grounded list (framed() consults it) — earned evidence crosses candidates, MINTS stay per-candidate (the old conflation was unearned F==E mints crossing); (2) memo moved from explore-param (inherited copy) onto the ROW (heap object): 'once per parent' = once per row across all forks. The inherited memo was accidentally load-bearing as the drain's loop guard — fresh memo per explore caused unbounded re-forking through drain re-pairings (RecursionError at 987 frames); row-scoped memo restores the guard while letting DIFFERENT rows fork the same sig — the S3 witness path: WDH's row parked DH, HALL's row later forked DH (different parent) → exchange → joint DH. RESULTS: tests 162/15/16; mhall 39 pairings, 8 grounded, claim via depth-1 MHALL×MS:Subject (U=HALL, O=Subject both earned); wdmh 84 pairings, 13 grounded, 1 joint, claim via TWO depth-1 closes — WDMH×what:Object AND WDMH×MHALL:had ALL (U=WDH interrogative + O=HALL answer, both earned — nearly the real opening story). Open: the claim rule still accepts any depth-1 close (neither grounding pairing is the goal pairing — the goal pairing itself never closes); one honest unwitnessed DH stall halt in wdmh step 2; mint pairings still emit no events (user's earlier edit).

*Relevance: high*
*Context: Trial: S2 alignment across candidates via earned-sig registry + row-scoped memo*
*Tags: trs cogitator s2-alignment grounded-registry row-memo trial*

---
*Observed: 2026-11-03T23:21:30.432Z*
