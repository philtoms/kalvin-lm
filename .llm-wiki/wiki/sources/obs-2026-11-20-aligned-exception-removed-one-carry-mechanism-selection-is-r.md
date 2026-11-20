---
type: source
title: "Observation: Aligned exception removed — one carry mechanism, selection is read-time"
tags:
  - cogitator
  - events
  - handover
  - design-principle
  - path-energy
status: observation
created: 2026-11-20
updated: 2026-11-20
slug: obs-2026-11-20-aligned-exception-removed-one-carry-mechanism-selection-is-r
relevance: high
observed_at: 2026-11-20T10:53:32.373Z
source_context: Removing the aligned exception after empirical comparison
---

# ⭐ Observation: Aligned exception removed — one carry mechanism, selection is read-time

The aligned-only story handover (Entry.aligned flag, commit ec4bbe99) was REMOVED in d867b926 after Phil's pushback with evidence: he ran a version without aligned tracking and the trace was essentially the same. My own experiment confirmed and sharpened this: unconditional carry gives wdmh step 2 in 21 rows vs 14 with the flag — the difference is exactly the U-side prefix (WDH attempts, DH stall, Object/what mints); the winning suffix (exchange → WDH framed → rotations → claim) is identical. Decisive findings: (1) NEITHER mechanism is a clean account — the aligned version's stories have unexplained mints (provenance gap), and the unconditional version LOSES rows at branch joins: when the exchange-resumed WDH copy departs into the root, its story REPLACES the root's accumulated one, dropping the O-side rows that ran in between (had framed, first HALL completion vanished from the report). The carried story is the carrying lineage's account, not a global log. (2) The flag baked selection policy into the carry mechanism and made stalled paths unreportable (a probe's story died with it) — contradicting the aim of rationalising any path, aligned or stalled. Resolution: ONE mechanism — frame and story ride every forward pass unconditionally (push: carry(parent); pop: always copy both); selection (pruned candidate path, derivation with provenance) is read-time, and provenance is needed anyway for path energy. Docs updated: §2.2 story paragraph, §2.3 exchange, T1, §2.4 invariant; report.py docstring. 163 passed. Design principle recorded: carrying is mechanism, selection is policy — keep them separate.

*Relevance: high*
*Context: Removing the aligned exception after empirical comparison*
*Tags: cogitator events handover design-principle path-energy*

---
*Observed: 2026-11-20T10:53:32.373Z*
