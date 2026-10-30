---
type: source
title: "Observation: Root pairs with goal only — trace table shows a true path"
tags:
  - trs
  - trace-table
  - path
  - root-pairing
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-root-pairs-with-goal-only-trace-table-shows-a-true-path
relevance: high
observed_at: 2026-10-30T09:09:32.581Z
source_context: Fixing the false path in the wdmh trace table
---

# ⭐ Observation: Root pairs with goal only — trace table shows a true path

Root-cause of the false wdmh path: the root reentry paired with ALL overlapping candidates, so the goal row's step said "reenter WDH, HALL" while the resolution actually flowed through a root-level alternate pairing (what:Object → DMH branch) — the claimed children never ran. Two fixes: (1) the depth-1 reentry pairs with the goal ONLY (the ask==goal claim; the U/O residuals do the exploring) — this also removes the seen-starvation where the root's U-child reentry (same kline as the root) found all pairings already seen; the user's table structure (root row → depth-2 WDH/HALL rows) now emerges naturally. (2) _Reentry.resolved_by records the row whose resolution closed it; the harness's _on_path_children claims a child in the step text only when its resolving row, exchange stall, or any kept row of the child is on the path — so every "reenter X, Y" is followed by rows with E=X/E=Y. Also fixed: the immediate F==side(E) fit frame now sets row.frame (was only set at resolution, leaving the frame column blank). wdmh path: WDMH→WDH→DH(exchange) + WDMH→HALL→DH(exchange), 6 of 20 rows; mhall: 11 of 35.

*Relevance: high*
*Context: Fixing the false path in the wdmh trace table*
*Tags: trs trace-table path root-pairing*

---
*Observed: 2026-10-30T09:09:32.581Z*
