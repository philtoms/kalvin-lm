---
type: source
title: "Observation: Exchange visible: temporal stalls, parent-chain trace, no self-covering"
tags:
  - trs
  - trace-table
  - exchange
  - self-covering
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-exchange-visible-temporal-stalls-parent-chain-trace-no-self-
relevance: high
observed_at: 2026-10-30T09:15:52.796Z
source_context: Making the wdmh exchange causally visible in the trace table
---

# ⭐ Observation: Exchange visible: temporal stalls, parent-chain trace, no self-covering

Three fixes to make the wdmh trace table causally faithful: (1) Temporal stall presentation — stall rows now render \"stalled X\"; only the ARRIVING exchange trigger (Row.exchange, set on b.stalled in _exchange) renders \"resolved X\" — the first DH stall no longer pretends to be resolved at its own row. (2) _trace closes over the reentry parent chain (row.re.parent) — the (HALL, had:DH) pairing joined the table with F=had, U=ALL, O=DH, making the exchange partner's origin visible. (3) Self-covering removed: frames carry their producing reentry (frame_re parallel list); _covered(x, exclude) never lets a frame cover the sides of rows in the reentry that produced it — previously the sibling (HALL, MHALL-canon) row's HALL frame silently satisfied the had:DH row's U=ALL, suppressing the ALL reentry. mhall grounds with 25/35 rows on the path, wdmh 9/20; wdmh reads: WDMH→WDH→DH stalled; HALL→had:DH→reenter DH→resolved DH (exchange); stores DH:[did,have] + WDMH:[MHALL].

*Relevance: high*
*Context: Making the wdmh exchange causally visible in the trace table*
*Tags: trs trace-table exchange self-covering*

---
*Observed: 2026-10-30T09:15:52.796Z*
