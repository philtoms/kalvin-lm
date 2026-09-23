---
type: source
title: 'Observation: Fit table cases 4–5 fixed; Def 14 consumed-nodes clause tightened to "every"'
tags:
  - kalvin-algebra
  - docs
  - fit
  - def14
status: observation
created: 2026-09-23
updated: 2026-09-23
slug: obs-2026-09-23-fit-table-cases-4-5-fixed-def-14-consumed-nodes-clause-tight
relevance: high
observed_at: 2026-09-23T17:36:48.133Z
source_context: Fixing fit-table inconsistency and tightening Def 14 reverse-licence wording in kalvin-algebra.md
---

# ⭐ Observation: Fit table cases 4–5 fixed; Def 14 consumed-nodes clause tightened to "every"

Fixed docs/kalvin-algebra.md Def 10 fit table: removed the accidental "u = ∅, o = ∅" from cases 4–5 (Denotation/No-fit), reintroduced in commit 06881e7's table reformat. Under Def 9, u=∅ ∧ o=∅ ⟺ s=σ(ν), so case 3 (Canon) would always fire first — making cases 4–5 unreachable and contradicting the doc's own python classifier, the species section (a:[b] → Denotation) and the KScript table. Also tightened Def 14's S2 restriction clause from "the consumed nodes carry the underfit" to "every consumed node carries underfit content" — resolving the any/all ambiguity in favour of the engine's existing all(...) reading in Derivation.targetings (derivation.py) and the Maude template's allCarry. Rationale: departures remove content, so every removed node must be in the misfit region (removing matched content is witness work, not targeting); arrivals only need to touch the region, since the strict-decrease condition polices anything beyond. Doc, engine and Maude template now agree on both points.

*Relevance: high*
*Context: Fixing fit-table inconsistency and tightening Def 14 reverse-licence wording in kalvin-algebra.md*
*Tags: kalvin-algebra docs fit def14*

---
*Observed: 2026-09-23T17:36:48.133Z*
