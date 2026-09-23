---
type: source
title: "Observation: Maude template updated to Def 15 meeting walk + Def 13 inertness + one-ask structure"
tags:
  - maude
  - formal-spec
  - kalvin-algebra
  - def15-meeting
status: observation
created: 2026-09-23
updated: 2026-09-23
slug: obs-2026-09-23-maude-template-updated-to-def-15-meeting-walk-def-13-inertne
relevance: high
observed_at: 2026-09-23T16:56:37.307Z
source_context: Bringing docs/kalvin-algebra.maude up to spec with latest rules
---

# ⭐ Observation: Maude template updated to Def 15 meeting walk + Def 13 inertness + one-ask structure

Updated docs/kalvin-algebra.maude (599 lines) from spec commits a2aeb82..06881e7. Changes: (1) KALVIN-WALK (single-party occurrence-licensed walk) replaced by KALVIN-MEET — Def 15 two-party meeting walk: positions (value, delivering kline) with descent/ascent step rules, no-retrace on ascent, A-side single-node witnesses only, B-side every node; [meet] rule fires on a shared value delivered by distinct klines and emits bridged(slot_a:[slot_b]); MeetState/MeetOutcome share supersort Meeting (Maude rules must stay within one kind). (2) Def 13 evidence inertness (S =/= N, S not in νK) added to all four replace rules. (3) revLicensed consumed-nodes clause switched from σ-any overlap to allCarry (every consumed node carries the underfit — engine's reading of Def 14). (4) KSCRIPT: countersignature `<=>` removed (subjectivity out, 44e2a69); ask is now (A ; ASK) : A — one ask structure with canon nodes riding; ASK atom + mask() added to KALVIN-VALUE (excluded from FULL); fit returns unknown for marked heads (ask's own band S4). (5) MEASURE: sig/complexity ops (Def 20 split), γ = sig × (1 − complexity). (6) MARY-HAD bridge is now w:[all] (compound B-departure, depth 2), not w:[a,l,l]; MARY-FRAGMENT drops the M1F/Q1F adoption — the fragment is stuck at entry (anchor arrival is an open question, not a licence). All searches re-verified with Maude 3.5.1: W0 => bridged w:[all] (1 solution); Q0 no done; Q1 3 value-equal done witnesses ([all,h,m], [a,l,l,h,m], [mhall]); Q2 terminal [w,h,m]; Q0F stuck at entry. Not encoded (noted in header): §11 Defs 21–23 (hop/selection/scope) remain strategy-side.

*Relevance: high*
*Context: Bringing docs/kalvin-algebra.maude up to spec with latest rules*
*Tags: maude formal-spec kalvin-algebra def15-meeting*

---
*Observed: 2026-09-23T16:56:37.307Z*
