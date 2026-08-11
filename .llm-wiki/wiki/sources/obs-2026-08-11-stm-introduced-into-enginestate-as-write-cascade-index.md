---
type: source
title: "Observation: STM introduced into EngineState as write-cascade index"
tags:
  - engine
  - stm
  - dialogue
  - engine-state
status: observation
created: 2026-08-11
updated: 2026-08-11
slug: obs-2026-08-11-stm-introduced-into-enginestate-as-write-cascade-index
relevance: high
observed_at: 2026-08-11T16:18:58.495Z
source_context: "Lean harness: introducing STM into EngineState for O(1) cross-reference"
---

# ⭐ Observation: STM introduced into EngineState as write-cascade index

> **⚠️ Superseded (2026-08-11).** The write-cascade integration this
> observation describes was reversed later the same day. STM is now reserved
> for the expansion strategies' exclusive use and wired into no logic; the
> helpers and accessors named below (`add_work`/`note_grounded` cascading to
> STM, `_stm_drop_if_orphaned`, `has_seen`, `find_by_nodes`) were removed. See
> [obs-2026-08-11-enginestate-four-store-model-grounded-ltm-stm-disconnected](/sources/obs-2026-08-11-enginestate-four-store-model-grounded-ltm-stm-disconnected.md)
> for the corrected model. Kept as a record of the dead-end.

Added `kalvin.stm.STM` to `dialogue.engine_state.EngineState` as the lowest tier of the write cascade. Every add to the work-list, grounded model, or frame now reaches STM (`add_work`, `note_grounded`, `frame_kline` helpers); removals mirror via orphan-checked `_stm_drop_if_orphaned`. Engine.py's direct `work_list.append`/`del work_list[idx]`/`grounded.setdefault` calls were rerouted through these helpers. New O(1) cross-store accessors: `has_seen(kline)` (sig+nodes dedup) and `find_by_nodes(nodes_sig)`. STM is derived on load (`from_dict` re-indexes from the three stores). mhall trace byte-identical before/after; all 1220 tests pass.

Key gotcha: STM removals must be orphan-checked. A kline can live in `grounded` AND `frame` simultaneously; `unframe` or `remove_work_at` must not evict the STM entry while a twin remains in another store. The naive `stm.remove` on every store-removal under-indexed STM (31 store entries → 24 STM entries on mhall).

Second gotcha (caught by reading the trace): `is_seen`/`signature_seen` are NOT "any kline with this sig seen" — they specifically mean "grounded OR pending as an `{X:[]}` Unknown ask." STM's signature key is a superset (any add). Routing them through STM broke the S4-ask discovery mechanism (slow route stopped emitting `{X:[]}` because the query kline's own `add_work` immediately populated STM). Those scoped reads deliberately keep their single-store semantics; `has_seen`/`find_by_nodes` are the new cross-store accessors.

Files: src/dialogue/engine_state.py, src/dialogue/engine.py, CONTEXT.md (STM glossary), docs/behaviour-notes.md (Engine-state rules).

*Relevance: high*
*Context: Lean harness: introducing STM into EngineState for O(1) cross-reference*
*Tags: engine stm dialogue engine-state*

---
*Observed: 2026-08-11T16:18:58.495Z*
