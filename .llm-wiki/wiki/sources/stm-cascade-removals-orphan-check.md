---
type: source
title: STM cascade removals must be orphan-checked
status: insight
category: backend
created: 2026-08-11
updated: 2026-08-11
slug: stm-cascade-removals-orphan-check
---

# STM cascade removals must be orphan-checked

> **⚠️ Superseded (2026-08-11).** The EngineState STM write cascade this
> insight describes was removed the same day — STM is now reserved for the
> expansion strategies' exclusive use and wired into no logic. The code
> references below (`_stm_drop_if_orphaned`, `has_seen`, `find_by_nodes`,
> `note_grounded`) no longer exist. The generalisable lesson — *a union index
> over multiple stores needs a quorum check on removal* — stands; see
> [obs-2026-08-11-enginestate-four-store-model-grounded-ltm-stm-disconnected](/sources/obs-2026-08-11-enginestate-four-store-model-grounded-ltm-stm-disconnected.md)
> for the corrected four-store model.

When `kalvin.stm.STM` is the union index over multiple stores (work-list + grounded + frame in `dialogue.engine_state.EngineState`), a removal from one store must NOT unconditionally call `stm.remove(kline)` — the same `(signature, nodes)` kline may still live in another store.

The naive cascade (every store-add → `stm.add`; every store-remove → `stm.remove`) under-indexes STM: a kline grounded and then unframed disappears from STM while still grounded. On `mhall` this showed as 31 store entries mapping to only 24 STM entries; 7 pairs wrongly evicted.

The fix: store-removals call `_stm_drop_if_orphaned(kline)`, which checks the other two stores and only removes from STM when no twin remains. This keeps STM a faithful union index.

Related trap: do NOT route the scoped reads (`is_seen`, `signature_seen`) through STM's signature key. Those mean "grounded OR pending `{X:[]}` Unknown ask," not "any add under this sig." STM's signature key is a superset; using it silently breaks the S4-ask discovery path (the slow route stops emitting `{X:[]}` because the query kline's own `add_work` populates STM immediately, so `is_seen(sig)` returns True prematurely). Keep scoped reads scoped; expose `has_seen` / `find_by_nodes` for genuine cross-store O(1) queries.

See [[entities/harness]], [[concepts/non-judging-harness]]. Discipline: when adding a union index, the removal path needs a quorum check across the member stores.

*Category: backend*

---
*Captured: 2026-08-11*

## Related

_Add links to related pages._
