---
type: source
title: "Observation: Ask-attends probe: empty ask silently discharged by grounded canon; riding ask never"
tags:
  - mhall
  - ask
  - is_answered
  - canon
  - probe
  - fork
status: observation
created: 2026-09-22
updated: 2026-09-22
slug: obs-2026-09-22-ask-attends-probe-empty-ask-silently-discharged-by-grounded-
relevance: high
observed_at: 2026-09-22T14:18:43.312Z
source_context: Probing the ask-attends fork on the mhall stall
---

# ⭐ Observation: Ask-attends probe: empty ask silently discharged by grounded canon; riding ask never

Probe dev/dialogue/probe_mhall_canon_empty_ask.py (AskAttends rationaliser subclass: _fast_route returns False for ask-marked feeds, so questions attend instead of S4-refusing). Results: (1) User's exact spec — canon MHALL:[Mary,had,a,little,lamb] fed S1 (grounds in frame), then empty ask MHALL|ASK:[] fed S4: the ask attends and is IMMEDIATELY DISCHARGED by Memory.is_answered's bare-ask path ("any witness resolves a bare ask") — the grounded canon at the ask's canon key answers it. Zero emissions, empty work list, refused=0, stm=0: the question is satisfied by inspection, completely silently; the harness would never see anything to answer or grade. (2) Control (empty ask alone, no canon): attends, sits in work list with NO goals (empty nodes → Def 22 coverage vacuously false), never emits — genuine residue. (3) Riding-nodes ask (the compiled form) with canon grounded: NOT discharged — is_answered skips the witness with identical nodes as "its own riding canon"; the ask stays attending with an EMPTY goal list (its canon excluded by Def 22's ask-canon rule, SVO covers none of the riding nodes). Asymmetry: the empty ask is answered BY the canon; the riding ask never is — it demands a content answer beyond the question's own words (the design that forced wdmh to derive WDMH:[ALL,had,Mary]). (4) Full lesson (scaffolds+canon+empty ask): scaffolds unchanged — stuck toward SVO canon; QueryObject:[Object] and lamb:[Object] hop "done" without moving (ground-path done, filtered, no proposal). Conclusion: making asks attend does not unlock mhall by itself; the is_answered bare-ask path quietly eats the empty-ask+canon combination, and the riding ask still has no derivation path.

*Relevance: high*
*Context: Probing the ask-attends fork on the mhall stall*
*Tags: mhall ask is_answered canon probe fork*

---
*Observed: 2026-09-22T14:18:43.312Z*
