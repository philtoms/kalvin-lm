---
type: source
title: "Observation: Orthogonal trace: one row per reentry; exchange feeds hold their slots"
tags:
  - trs
  - trace-table
  - orthogonality
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-orthogonal-trace-one-row-per-reentry-exchange-feeds-hold-the
relevance: high
observed_at: 2026-10-30T09:57:51.252Z
source_context: Making the trace table orthogonal — one candidate row per entry
---

# ⭐ Observation: Orthogonal trace: one row per reentry; exchange feeds hold their slots

Orthogonal trace tables: _trace now tables AT MOST ONE row per reentry (tabled: reentry-id → holding row-id; any other candidate row of that reentry is another path and is skipped — fixing conflated rows like Object×QO:Object + Object×SVO). Exchange feeds pre-seed their slots: the parent rows of exchanged stalls (the S3 path — e.g. HALL×had:DH) hold their reentries' slots BEFORE the walk, so the canon shortcut that merely resolved first (HALL×MHALL) does not displace them — the exchange row keeps its visible parent. Final tables: wdmh 15/42 rows — WDMH→WDH→DH stalled; Object-side SVO peel; HALL×had:did-have (F=had, U=ALL, O=DH); had→MALL→ALL→MH chain; DH resolved (exchange, last); stores DH:[did,have]+WDMH:[MHALL]. mhall 12/114 rows ending stores MHALL:[SVO]. Both claims ground, 134 tests pass, deterministic.

*Relevance: high*
*Context: Making the trace table orthogonal — one candidate row per entry*
*Tags: trs trace-table orthogonality*

---
*Observed: 2026-10-30T09:57:51.252Z*
