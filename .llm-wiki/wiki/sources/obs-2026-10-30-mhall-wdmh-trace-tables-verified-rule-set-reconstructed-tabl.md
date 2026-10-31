---
type: source
title: "Observation: mhall/wdmh trace tables verified — rule set reconstructed, tables consistent"
tags:
  - trs
  - trace-table
  - verification
  - mhall
  - wdmh
  - framing
  - exchange
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-mhall-wdmh-trace-tables-verified-rule-set-reconstructed-tabl
relevance: high
observed_at: 2026-10-30T16:25:05.525Z
source_context: Verifying mhall/wdmh trace tables before rewriting Cogitator.rationalise
---

# ⭐ Observation: mhall/wdmh trace tables verified — rule set reconstructed, tables consistent

Walked both user-supplied trace tables row by row against the compiled domains (mhall: MS:Subject=MarySubject:[Subject], HV:Verb, QO:Object, ALLQ:Query, SVO:[S,V,O] canon held at step 1; wdmh adds what:Object, had:[did,have], goal=MHALL canon held). Every F/U/O decomposition checks out; both tables are internally consistent and exactly reproducible under one rule set: (1) opening pairing ask×goal, goal is the only depth-1 candidate; (2) reentry on U (entry-side witness nodes) and O (candidate-side), LIFO, u-child leads; (3) F==E at pairing frames E's content; (4) side settled = empty | framed by proper ancestor | child reentry resolved (stalls only via exchange); (5) row resolves → reentry resolves → FRAMES THE REENTRY'S REGION (this mints WDH/HALL/MHALL/SVO — the final row's frame column "MHALL, SVO"/"WDH, HALL" is exactly the two child-reentry frames) → BP cascades; (6) stall = no candidates, parked by content signature; (7) exchange: two stalls same sig different parents → frame sig, admit ⟨sig, adj_a∪adj_b⟩, resolve both, BP both in stall order; (8) opening row resolves → admit ⟨ask,{goal}⟩ grounded. Two untabled-but-load-bearing interludes verified NECESSARY: mhall's V-descent (V×HV mirror→had×HV mirror→resolved) must run between the ALL-chain BP and row 3r else Verb never frames; mhall's SVO-identity row and wdmh's Object-descent (Object×QO mirror→Query×QO mirror) likewise before their parents resolve.

*Relevance: high*
*Context: Verifying mhall/wdmh trace tables before rewriting Cogitator.rationalise*
*Tags: trs trace-table verification mhall wdmh framing exchange*

---
*Observed: 2026-10-30T16:25:05.525Z*
