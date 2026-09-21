---
type: source
title: "Observation: Harness sub-script delineation simplified to scope-0 entry groups"
tags:
  - dialogue
  - harness
  - kscript
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-harness-sub-script-delineation-simplified-to-scope-0-entry-g
relevance: high
observed_at: 2026-09-21T14:01:41.968Z
source_context: Simplifying sub-script delineation in src/dialogue/harness.py
---

# ⭐ Observation: Harness sub-script delineation simplified to scope-0 entry groups

Harness sub-script delineation simplified: a group opens at every scope-0 entry; expansion (scope≠0) entries join the current group positionally. Removed: first_by_ann expansion routing, the unused annotation-occurrence key in groups/steps, the opener-selection chain (opener is now group[0] since each group holds exactly one scope-0 entry), and the step filter that skipped unannotated non-ASK openers. Consequence: compiled output is source-then-expansion, so all expansion entries now trail into the last scope-0 group (they previously routed to the first group of their annotation). mhall.ks now yields 8 steps (one per authored entry); 95 tests pass. Annotations remain only in presentation (section headers in present()).

*Relevance: high*
*Context: Simplifying sub-script delineation in src/dialogue/harness.py*
*Tags: dialogue harness kscript*

---
*Observed: 2026-09-21T14:01:41.968Z*
