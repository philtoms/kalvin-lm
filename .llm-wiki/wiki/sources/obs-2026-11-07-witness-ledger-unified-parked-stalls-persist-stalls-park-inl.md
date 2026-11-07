---
type: source
title: "Observation: Witness ledger unified: parked stalls persist; stalls park inline without reentry"
tags:
  - cogitator
  - frames
  - stalls
  - ledger
  - exchange
  - witnesses
status: observation
created: 2026-11-07
updated: 2026-11-07
slug: obs-2026-11-07-witness-ledger-unified-parked-stalls-persist-stalls-park-inl
relevance: high
observed_at: 2026-11-07T14:13:54.197Z
source_context: Unifying frames and stalls into one persistent witness ledger
---

# ⭐ Observation: Witness ledger unified: parked stalls persist; stalls park inline without reentry

Implemented the witness-ledger unification (user's ruling: "witnessing is a procedure that generates frames — if DH is framed it shouldn't need to look for candidates; DH does not need to stall because it does not need to reenter"). The earlier "running frame refuted" experiment was the RIGHT idea half-implemented: frames (completed witnesses) persisted but stalls (parked witnesses) did not — an incoherent half-ledger. Coherent version now in src/rationalisation/cogitator.py: (1) self.frame AND self.stalled initialize in __init__ — the Runner's steps share the whole ledger like they share the domain; (2) inline parking — an unsettled side with no candidates parks AT THE PAIRING SITE (no push, no reentry, no stall iteration); a park meeting an opposite-side parked witness exchanges on the spot, mints the frame, and can close the pairing in the same iteration; the loop-top stall branch survives only for a bare goal-less opening; (3) the close check moved after the settle pass (a completed park settles sides mid-iteration), with the candidate pop restored after close (a dropped pop caused an immediate-close pairing to re-pair itself to the 4096 cap). Results: mhall 58→46, wdmh 66→50 and 93→3 pairings; wdmh step 4 grounds in FIVE events (pair, pair, exchange at the inline park, resolved, grounded) — step 1 had parked did|have side O (HALL×had:did have), step 3's WDH×what U-side park completes it, frame(DH), WDH closes same-iteration, claim WDMH:[MHALL]. Tests 162/15/16; all four stores land in both scripts. Notes: unwitnessed parks (Det, Mod) persist and re-report as halts on every later step — honest but a changed reporting shape; a park from an old context completing against a new-context park is sound by the same content-level logic as cross-step frames; D is static within a run (stores only at claim/S1), so inline parking never misses candidates that appeared mid-run.

*Relevance: high*
*Context: Unifying frames and stalls into one persistent witness ledger*
*Tags: cogitator frames stalls ledger exchange witnesses*

---
*Observed: 2026-11-07T14:13:54.197Z*
