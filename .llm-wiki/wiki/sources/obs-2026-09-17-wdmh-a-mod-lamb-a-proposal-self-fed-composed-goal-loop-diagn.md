---
type: source
title: "Observation: WDMH [a,Mod,lamb,a] proposal: self-fed composed-goal loop diagnosed"
tags:
  - dialogue
  - engine
  - wdmh
  - derivation
  - bug
status: observation
created: 2026-09-17
updated: 2026-09-17
slug: obs-2026-09-17-wdmh-a-mod-lamb-a-proposal-self-fed-composed-goal-loop-diagn
relevance: high
observed_at: 2026-09-17T11:59:01.347Z
source_context: dialogue-dev investigation of WDMH run repetition
---

# ⭐ Observation: WDMH [a,Mod,lamb,a] proposal: self-fed composed-goal loop diagnosed

Investigated why the WDMH-on-reloaded-mhall run proposes `little:[a, Mod, lamb, a]` (shown as 0x800000e7d hex) 8 times. Causal chain: (1) `a:[Det]`'s hop chain writes composed correspondence `Det:[a, Mod, lamb]` (acq=4, a slot-walk product, overlap-loose arrival); (2) that write enters the candidate goal pool for queued `little:[Mod]` and ranks FIRST (J=1/3, the only positive — "Mod"'s word bit appears in no script kline, only in the engine's own writes); (3) derivation 1 toward it is stuck but _walk_b departs goal slot a, walks a→Det→[a,Mod,lamb] THROUGH THE GOAL'S OWN BRIDGE, and writes self-containing `Mod:[a, Mod, lamb, a]`; (4) the DUPLICATE goal copy (engine._writes accumulates exact duplicates across passes — 17 writes/12 distinct at run end) re-derives the same goal in the same hop, its scope trawls derivation 1's fresh writes, forward targeting on the self-containing bridge fires (misfit 2→0) → done at j1=1.0; (5) proposal stamped S1/255, supervisor declines, but refusal only filters emission (is_refused in _propose) — little:[Mod] never grounds (Mod never receives an identity — genuine script residue) so cogitate re-proposes it every turn: 10 _propose calls, 8 reaching the identical done. Probes: dev/dialogue/probe_wdmh_goals.py (TARGET env), probe_wdmh_decode.py, probe_derivation_monitor.py.

*Relevance: high*
*Context: dialogue-dev investigation of WDMH run repetition*
*Tags: dialogue engine wdmh derivation bug*

---
*Observed: 2026-09-17T11:59:01.347Z*
