---
type: source
title: "Observation: Parent-origin bar added to Cogitator candidate selection"
tags:
  - trs
  - cogitator
  - candidates
  - reentry
  - rule
status: observation
created: 2026-11-03
updated: 2026-11-03
slug: obs-2026-11-03-parent-origin-bar-added-to-cogitator-candidate-selection
relevance: high
observed_at: 2026-11-03T13:14:48.526Z
source_context: "Strengthening Cogitator candidate selection: no parent entry / parent candidate"
---

# ⭐ Observation: Parent-origin bar added to Cogitator candidate selection

Cogitator._candidates now bars a reentry's origin: candidates sharing the PARENT ENTRY's signature or the PARENT CANDIDATE's signature are excluded (immediate parent only; depth-1 goal exempt). Implemented by passing psig/pcsig through push() into _candidates (cogitator.py); _Entry unchanged. Closes the hole the like-for-like fingerprint bar missed: fingerprint==E only catches candidates whose whole composition is the entry — MHALL-sig klines (fingerprint MHALL) were free to pair HALL, re-deriving the parent pairing's decomposition (old wdmh step-4 rows 2/5/6). Verification: tests 162/15/16 unchanged; both scripts ground identically with identical stores (mhall: MHALL:[SVO]; wdmh: WDMH:[MHALL]); mhall 160→148 rows (removed rows are exactly the origin-pairings: SV×SVO, SO×SVO, VO×SVO, Query×ALLQ, ALL×ALLQ); wdmh 365→420 rows — HALL's depth-2 pass no longer mints (frame HALL now minted by its close at the end: ALL canon + framed had, the reductive route only); a second HALL reentry at depth 6 (pushed by Mary×MHALL canon) resolves via ALL canons. Note for u-side children the parent-candidate bar is implied by overlap (U = E & ~cs cannot overlap cs); it bites on o-side children. KNOWN FORK: rule is immediate-parent; grandparent pairings still allowed (e.g. new wdmh `| 3 | HAL | MHALL:had ALL |` mints HAL — MHALL is HAL's grandparent claim); ReferenceCogitator._in_path bars the whole ancestor path (entries by kline, o-side parents' candidates by sig) — stricter variant if wanted.

*Relevance: high*
*Context: Strengthening Cogitator candidate selection: no parent entry / parent candidate*
*Tags: trs cogitator candidates reentry rule*

---
*Observed: 2026-11-03T13:14:48.526Z*
