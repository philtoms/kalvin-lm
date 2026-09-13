---
type: source
title: "Observation: Step 3 Mary example: o→all hop licensed by reverse read of all:[o]"
tags:
  - docs
  - kalvin
  - slot-walk
  - connotation
  - reverse-replace
status: observation
created: 2026-09-13
updated: 2026-09-13
slug: obs-2026-09-13-step-3-mary-example-o-all-hop-licensed-by-reverse-read-of-al
relevance: high
observed_at: 2026-09-13T16:07:57.071Z
source_context: Fixing the Step 3 slot-walk demonstration in kalvin-simplified.md
---

# ⭐ Observation: Step 3 Mary example: o→all hop licensed by reverse read of all:[o]

Phil reported the Step 3 slot walk (w:[w] → w:[o] → w:[all] → w:[a,l,l]) in the docs/kalvin-simplified.md "what did Mary have?" worked example as unlicensed: no memory kline traverses o → all. Resolution: the licence is the Connotation all:[o] read in REVERSE — Def 13's mirror clause ("the same kline read from the other side licenses the mirror derivation"), since its witness [o] occurs as the node multiset at w:[o]. The doc's Step 3 never named any licence, so a forward-only reading (or Def 16's t ∈ ν_A selection) concluded stuck at w:[o]. Fixed: Step 3 rewritten with a per-hop licence table (forward/reverse/Canon-expand), explicit crossover prose (w and all meet only at shared node o; arrival orients the correspondence), the T2 no-revisit policy forcing the walk through all:[o] rather than back through w:[o], and acquisition depth 3 = the three edges crossed (corroborated by §11's Ĥ example). Def 17 tightened to "either side of a held kline occurring in the walk's nodes", matching CONTEXT.md's Slot gloss (no glossary change needed). Engine already implements the crossover: src/dialogue/cogitator.py expand() rev_paths + hops + rev.hops. Not committed.

*Relevance: high*
*Context: Fixing the Step 3 slot-walk demonstration in kalvin-simplified.md*
*Tags: docs kalvin slot-walk connotation reverse-replace*

---
*Observed: 2026-09-13T16:07:57.071Z*
