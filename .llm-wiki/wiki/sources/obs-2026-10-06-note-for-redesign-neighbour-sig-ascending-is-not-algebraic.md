---
type: source
title: "Observation: Note for redesign: neighbour-sig ascending is not algebraic"
tags:
  - trs
  - redesign
  - note
  - candidate-ordering
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-note-for-redesign-neighbour-sig-ascending-is-not-algebraic
relevance: medium
observed_at: 2026-10-06T15:26:55.832Z
source_context: Step-by-step rationalisation reentry redesign
---

# 🔍 Observation: Note for redesign: neighbour-sig ascending is not algebraic

REDESIGN NOTE (for later, user: 'neighbour-signature ascending is not an algebraic requirement'): the current rationalise() iterates candidate neighbours by SIG ASCENDING (sorted(neighbours, key=lambda k: k.sig)) — a determinism convenience, not algebra. Any redesign must not treat this order as meaningful; the mhall traces happen to put the MarySubject pairing first by coincidence (0x21 < 0x42 < 0x180). Filed for the redesign's ordering decision.

*Relevance: medium*
*Context: Step-by-step rationalisation reentry redesign*
*Tags: trs redesign note candidate-ordering*

---
*Observed: 2026-10-06T15:26:55.832Z*
