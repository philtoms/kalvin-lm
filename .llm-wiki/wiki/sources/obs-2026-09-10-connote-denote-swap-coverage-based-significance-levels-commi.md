---
type: source
title: "Observation: Connote/denote swap + coverage-based significance levels committed (0f8b56a)"
tags:
  - kscript
  - compiler
  - significance
  - commit
status: observation
created: 2026-09-10
updated: 2026-09-10
slug: obs-2026-09-10-connote-denote-swap-coverage-based-significance-levels-commi
relevance: high
observed_at: 2026-09-10T08:31:59.240Z
source_context: Committing the connote/denote structure swap and significance-level redefinition
---

# ⭐ Observation: Connote/denote swap + coverage-based significance levels committed (0f8b56a)

Session committed as 0f8b56a on branch `dialogue` (18 files, +194/−79): CONNOTE/DENOTE compiled structures swapped (A > B ⇒ A:[B], A < B ⇒ B:[A], A = B ⇒ AB:[B] — compound signature, node is the denoted value; concat moved to the sig slot in SymbolicEntry/TokenEncoder), and sig_level redefined — significance levels derive purely from the signature–nodes relationship (S1 exact, S2 ≥1 covered node, S3 none, S4 empty), moving no-coverage misfits S2→S3. Containment mirrors flipped in EngineState.is_connotation and harness._structure_class; curriculum prompt, CONTEXT.md, and wiki concept page updated. Pre-existing breakage left untouched: sig_in calls in reentry.py/expand_fit.py, dead probes (tuple-unpacking connotateY, dialogue.actors import), "S2 strategy" historical names in pivot_fill/reentry.

*Relevance: high*
*Context: Committing the connote/denote structure swap and significance-level redefinition*
*Tags: kscript compiler significance commit*

---
*Observed: 2026-09-10T08:31:59.240Z*
