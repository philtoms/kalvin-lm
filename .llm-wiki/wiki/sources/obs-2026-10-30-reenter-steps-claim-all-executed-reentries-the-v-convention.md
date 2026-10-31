---
type: source
title: "Observation: Reenter steps claim all executed reentries — the V convention"
tags:
  - trs
  - trace-table
  - steps
  - convention
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-reenter-steps-claim-all-executed-reentries-the-v-convention
relevance: medium
observed_at: 2026-10-30T10:23:19.093Z
source_context: Fixing missing reenter claims in the trace table step column
---

# 🔍 Observation: Reenter steps claim all executed reentries — the V convention

Step column convention corrected: reenter steps claim every EXECUTED reentry (child.popped — set in _run at pop), not only children with tabled rows. A reentry like ⟨had,∅⟩ (no witness nodes — HV:Verb's node lies in F, not O) pops, halts R4, produces no rows — its side resolves via frames from elsewhere — and is still claimed, matching the user's V convention (their HALL row: \"reenter ALL, V\" — no V row follows; V resolves via the VO frame). Never-popped children (run ended at the claim before their turn) are still omitted. Result: wdmh HALL row reads HALL×had:did-have | F=had | U=ALL | O=DH | reenter ALL, DH — exactly the expected table's H:did have row shape; SV row shows \"reenter Subject, had\"; VO row \"reenter Verb, Query\". mhall rows 3/6/8 similarly claim both sides. 134 tests pass, deterministic.

*Relevance: medium*
*Context: Fixing missing reenter claims in the trace table step column*
*Tags: trs trace-table steps convention*

---
*Observed: 2026-10-30T10:23:19.093Z*
