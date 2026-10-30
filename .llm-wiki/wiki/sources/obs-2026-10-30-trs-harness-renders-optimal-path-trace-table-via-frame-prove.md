---
type: source
title: "Observation: TRS harness renders optimal-path trace table via frame provenance"
tags:
  - trs
  - harness
  - trace-table
  - provenance
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-trs-harness-renders-optimal-path-trace-table-via-frame-prove
relevance: high
observed_at: 2026-10-30T07:37:37.476Z
source_context: Rendering the optimal-path trace table in the TRS harness
---

# ⭐ Observation: TRS harness renders optimal-path trace table via frame provenance

Cogitator now records frame provenance (frame_src: frame value → source rows) at every _frame site (fit-on-match, resolution framing F and the reentry sig, and the exchange framing both stall rows). Rationalisation gained path + claim fields; _trace() walks the provenance DAG backwards from the goal row (for each side region, the frames covering it and their source rows) — yielding exactly the rows that carried the opening pairing to resolution, in generation order. dev/dialogue/trs_harness renders this as the markdown table form: depth | E(ntry) | C(andidate) | F(it) | U(nderfit) | O(verfit) | frame | step — with word-order decomposition for derived regions ("had a little lamb"), sig:nodes for candidates whose nodes don't compose the sig (had:did have), bare sig otherwise (SVO), and steps "reenter U, O" / "stalled X" / "resolved X" (exchange) / "resolved F". When the claim does not ground, the harness falls back to all rows with a "(no path — the goal was not reached)" note and the stalls surface as connect proposals. mhall path: 12 rows ending "resolved SVO" → stores MHALL:[SVO]; wdmh path: 10 rows with the two did-have exchange rows at depths 3/4 → stores WDMH:[MHALL].

*Relevance: high*
*Context: Rendering the optimal-path trace table in the TRS harness*
*Tags: trs harness trace-table provenance*

---
*Observed: 2026-10-30T07:37:37.476Z*
