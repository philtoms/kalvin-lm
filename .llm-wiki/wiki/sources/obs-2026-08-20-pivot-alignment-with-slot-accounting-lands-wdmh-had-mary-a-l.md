---
type: source
title: "Observation: Pivot alignment with slot accounting lands WDMH:[had,Mary,a,little,lamb]"
tags:
  - dialogue
  - engine
  - pivot-alignment
  - proposals
status: observation
created: 2026-08-20
updated: 2026-08-20
slug: obs-2026-08-20-pivot-alignment-with-slot-accounting-lands-wdmh-had-mary-a-l
relevance: critical
observed_at: 2026-08-20T15:05:03.668Z
source_context: Updating behaviour notes + wiki before commit
---

# 🔴 Observation: Pivot alignment with slot accounting lands WDMH:[had,Mary,a,little,lamb]

Session advanced the misfit proposal algorithm to pivot alignment with slot accounting. New mechanisms in src/dialogue/expand_fit.py since f48112d: (1) reentry — propose recurses on each proposal (depth 2), widening connotation sets; (2) concrete-first grading — _fill_distance: fill that is a node of the entry's own canon gets distance 1, connotational fills get hops+1 ("grounded terminal" discriminators fail because grammar types have identities); (3) drop rule — fill-derived proposals drop on uncovered bit residual; (4) _pivot_proposals — entry canon aligned against a sharing grounded canon: shared node = S2 (canonical, NOT S1 per user correction), edge-hop path = S3 (node replaced by pivot counterpart), no path = gap slot; canon nodes forming a grounded sub-canon resolve as a group (did+have→DH→had); gap slots take pivot leftovers: one gap takes the whole residual group (operand-pairings convention), N gaps one each, gaps>leftovers drops, leftovers with no open gap EXCLUDED (this was the ALL:[a,little,lamb,Mary,had] bug — surplus graft survived inside the gap-fill loop); slot accounting, not bit residual, decides pivot survival. Result: mhall proposes WDMH:[had, Mary, a, little, lamb] — full alignment, did+have→had at S3, Mary shared, what gap filled by grouped [a,little,lamb] at S4. docs/behaviour-notes.md rewritten to match. Open: find()-returns-last bucket-order fragility (wdmh greedy variant); self-canon echo (zero-gap pivot proposes entry's own canon); S5T2 duplicate-STM asks+proposes of same kline; refused-set lifetime; reentry widening/refusal circumvention.

*Relevance: critical*
*Context: Updating behaviour notes + wiki before commit*
*Tags: dialogue engine pivot-alignment proposals*

---
*Observed: 2026-08-20T15:05:03.668Z*
