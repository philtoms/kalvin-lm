---
type: source
title: "Observation: Walk ruling implemented: whole-witness descents, containment ascents — Query chain now the only ALL→O path"
tags:
  - walk
  - ascent
  - containment
  - witness
  - accounting
  - mhall
  - algebra
status: observation
created: 2026-09-22
updated: 2026-09-22
slug: obs-2026-09-22-walk-ruling-implemented-whole-witness-descents-containment-a
relevance: critical
observed_at: 2026-09-22T17:47:46.041Z
source_context: Closing the partial-witness walk leak per Phil's ruling
---

# 🔴 Observation: Walk ruling implemented: whole-witness descents, containment ascents — Query chain now the only ALL→O path

Phil's ruling on the ALL→lamb→Object shortcut implemented and verified: the walk cannot partially consume a multi-node witness ("the algebra cannot just leave a and little dangling — all nodes must be accounted for"). Two code refinements in derivation._walk_neighbours: (1) A-side descents cross single-node witnesses only (a step licenses the kline's whole witness; multi-node witnesses would dangle siblings when cherry-picked) — B's side still descends the goal's own canon (ν_B is walk material; its witness enumerates B's slots, every node accounted); (2) the ascent requires CONTAINMENT not overlap — the walk enters a head whose content CONTAINS the current value (residual(v, head)==0), not merely shares a bit. The overlap form was the actual leak: ALL rode lamb:[Object]'s head on the shared lamb-bit then legally descended to Object. With containment: M ⊆ MarySubject ✓, ALL ⊆ ALLQuery ✓, Query ⊆ QueryObject ✓; ALL ⊄ lamb-head (a, little absent) — closed. Verified: mhall bridges now Mary:[Subject] acq=3, had:[Verb] acq=3, ALL:[Object] acq=5 (the 4-edge ALL→ALLQuery→Query→QueryObject→Object chain — the 2-connotation path grades below the 1-connotation paths, the designed 0xfc/0xfb distinction); chain reaches done, witness [Subject, Verb, Object]; harness proposes MHALL:[Subject, Verb, Object] S1 255 and grounds; wdmh unchanged (WDMH:[ALL, had, Mary] S1 255); 90/90 tests. Doc updated in the same edit: ascent wording changed from "covered by" (Def 8 overlap) to "whose content contains the current value's"; descent sentence gains the whole-witness clause; connotation paragraph notes B-side ν_B material. Probes: probe_all_walk.py added.

*Relevance: critical*
*Context: Closing the partial-witness walk leak per Phil's ruling*
*Tags: walk ascent containment witness accounting mhall algebra*

---
*Observed: 2026-09-22T17:47:46.041Z*
