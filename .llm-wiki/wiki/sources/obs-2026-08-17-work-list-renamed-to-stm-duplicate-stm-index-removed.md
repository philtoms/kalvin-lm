---
type: source
title: "Observation: work_list renamed to stm; duplicate STM index removed"
tags:
  - engine
  - stm
  - refactor
  - engine-state
  - rename
status: observation
created: 2026-08-17
updated: 2026-08-17
slug: obs-2026-08-17-work-list-renamed-to-stm-duplicate-stm-index-removed
relevance: high
observed_at: 2026-08-17T12:14:00.145Z
source_context: Renaming work_list to stm — work_list was already the attention store
---

# ⭐ Observation: work_list renamed to stm; duplicate STM index removed

Renamed EngineState.work_list to EngineState.stm (first refactor growing the lean engine toward the normative glossary). Recognition: the work_list already WAS the attention store — incoming entries and ungrounded sigs/nodes unpacked from them sit there while cogitation attends to them; that is exactly STM's written-by-attention relation. No need for both, so the separate kalvin.stm.STM index field (reserved-for-expansion, never wired) was removed entirely, along with the STM import and __post_init__.

Mechanics: field rename, add_work→add_stm, remove_work_at→remove_stm_at, ground(work_idx)→ground(stm_idx), canon_nodes/pop_identity/is_seen/signature_seen internals updated, persistence key "work_list"→"stm". rationalise.py (the verbatim reference) untouched. Harness summary label now "stm (attending to at end of run):". mhall trace byte-identical except that label; all 1220 tests pass.

Docs synced: behaviour-notes Engine-state rules (three stores now), wiki k-engine.md (three-store model), stm entity page lean note (work_list IS STM; the index field removed). CONTEXT.md needed no change — it names the relations, not the lean field names.

Remaining gap to the glossary: Frame/LTM are not yet split by promotion in the lean engine (ltm populated directly on grounding), and grounding is still S1-shaped only.

*Relevance: high*
*Context: Renaming work_list to stm — work_list was already the attention store*
*Tags: engine stm refactor engine-state rename*

---
*Observed: 2026-08-17T12:14:00.145Z*
