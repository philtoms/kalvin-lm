---
type: source
title: "Observation: wdmh grounding restored: S2 rotates, exchange notifies witness, live gates finish"
tags:
  - cogitator
  - reentry
  - exchange
  - grounding
status: observation
created: 2026-11-19
updated: 2026-11-19
slug: obs-2026-11-19-wdmh-grounding-restored-s2-rotates-exchange-notifies-witness
relevance: high
observed_at: 2026-11-19T22:12:01.643Z
source_context: Restoring wdmh grounding under the seen-rotation redesign
---

# ⭐ Observation: wdmh grounding restored: S2 rotates, exchange notifies witness, live gates finish

User corrected my analysis of wdmh grounding under the seen-rotation redesign: h(ad)=>did have IS reachable — as a HALL/had candidate (sig `had` overlaps HALL), whose O-side is exactly DH (did|have). My claim that it "can never serve" conflated the stall sink's signature-overlap filter (which indeed cannot see had:[did,have] from DH's seat) with the candidate path (which reaches it via HALL). The real defect: an entry reaching S2 stopped rotating — HALL framed via HV:Verb and never tried had:[did,have]. Fix 1: S2 (framed or claimed) now rotates to the ask's next unseen candidate; the parent resumes only when the rotation exhausts ("grounding does not stop the loop" applies to the ask, not just the run). Fix 2: the S3 exchange now resumes the witness's parent too — "witness each other" means both sides learn; without it DH frames but WDH×what (departed earlier) never re-evaluates and no claim fires. Fix 3 (consequence): the witness notify pushes a second live entry of the run, breaking the one-live-entry-per-run invariant that made eager root-departure finishes safe — the root could tail-depart and finish while WDH×what sat queued beneath (double finish, double claim, duplicated step). Added _Run.live (run entries on the stack): finish fires only when the root departs with live==0. Result: mhall and wdmh ground in both default and -s modes, claim once per scenario, Det/Mod stalls now honestly explored; steps count correct.

*Relevance: high*
*Context: Restoring wdmh grounding under the seen-rotation redesign*
*Tags: cogitator reentry exchange grounding*

---
*Observed: 2026-11-19T22:12:01.643Z*
