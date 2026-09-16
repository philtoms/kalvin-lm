---
type: source
title: "Observation: kalvin-algebra.md restructured: Measurement §10 (Defs 16-19) before Strategy §11 (Defs 20-21)"
tags:
  - docs
  - kalvin
  - structure
  - measurement
  - renumbering
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-kalvin-algebra-md-restructured-measurement-10-defs-16-19-bef
relevance: high
observed_at: 2026-09-16T09:59:29.365Z
source_context: "Restructuring kalvin-algebra.md: measurement promoted to definitions, moved before strategy"
---

# ⭐ Observation: kalvin-algebra.md restructured: Measurement §10 (Defs 16-19) before Strategy §11 (Defs 20-21)

Restructured kalvin-algebra.md per Phil's structural challenge ("why does measurement not have definitions, and why is it grouped at the end with terminology; older decisions need constant review"). Root cause: measurement was written as elaboration of §0.3 notation and sat AFTER Strategy — but Def 16 Selection now consumes γ, a backwards dependency. Fix: sections swapped — §10 is now Measurement with numbered definitions carved from the old prose (Def 16 Jaccard overlap J incl. the forced-not-chosen argument; Def 17 Resolution depth D̄; Def 18 Acquisition depth Ĥ with provenance rules; Def 19 Graded significance γ with δ folded in), §11 is Strategy (Selection → Def 20, Slot derivation → Def 21). Dependency order now flows forward: Def 20 references Def 19, both after Def 8. All cross-refs updated: doc header map ("divided as follows" — five entries), §6/§8 §-refs, worked-example Def 17→21 refs, Selection's "graded significance of §11"→"(Definition 19)", appendix. CONTEXT.md: 9 entries updated (Def 20/21/19 refs; §10/§11 swapped; bands now §10; γ gets Def 19). Maude header: KALVIN-MEASURE §10 Defs 16–19, KALVIN-WALK §11 Def 21, §11 loop. Probes: comment-only updates (Def 17→21, §11→§10), both re-run PASS. Doc now has 21 numbered definitions, none forward-referencing a later section.

*Relevance: high*
*Context: Restructuring kalvin-algebra.md: measurement promoted to definitions, moved before strategy*
*Tags: docs kalvin structure measurement renumbering*

---
*Observed: 2026-09-16T09:59:29.365Z*
