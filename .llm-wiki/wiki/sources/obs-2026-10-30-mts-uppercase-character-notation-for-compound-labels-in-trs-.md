---
type: source
title: "Observation: MTS Uppercase Character notation for compound labels in TRS table"
tags:
  - trs
  - harness
  - labels
  - mts
  - notation
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-mts-uppercase-character-notation-for-compound-labels-in-trs-
relevance: medium
observed_at: 2026-10-30T08:34:54.005Z
source_context: MTS notation for compound KNode labels in the TRS harness table
---

# 🔍 Observation: MTS Uppercase Character notation for compound labels in TRS table

TRS harness now renders compound labels in the Uppercase Character (MTS) notation: _humps() splits camelCase (acronyms kept whole), _initials() maps a compound of 2+ signature parts to each part whole when it is already all-caps (ALL→ALL) else its initial — MarySubject→MS, hadVerb→HV, QueryObject→QO, ALLQuery→ALLQ, did|have→DH, Mary|had|a|little|lamb→MHALL, had|a|little|lamb→HALL, what|did|have→WDH, Verb|Object→VO. Single signatures stay whole (Mary, had, Subject). Applied uniformly in the trace table (_tlabel for E/C-sig/F/U/O/frame/step) and the stores/halt lines (_tline: sig in MTS form, nodes spelled out — "DH:[did, have]"). Model-level labels deliberately unchanged: merge/residue/meet token arithmetic depends on full pipe-composed labels; MTS is presentation-only in dev/dialogue/trs_harness.py.

*Relevance: medium*
*Context: MTS notation for compound KNode labels in the TRS harness table*
*Tags: trs harness labels mts notation*

---
*Observed: 2026-10-30T08:34:54.005Z*
