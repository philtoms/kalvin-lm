---
type: source
title: "Observation: Obsolete dialogue modules removed (d9be56f): −2105 lines, 95 tests green"
tags:
  - dialogue
  - cleanup
  - committed
  - modules
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-obsolete-dialogue-modules-removed-d9be56f-2105-lines-95-test
relevance: medium
observed_at: 2026-09-21T12:20:59.606Z
source_context: Committing obsolete dialogue module removal
---

# 🔍 Observation: Obsolete dialogue modules removed (d9be56f): −2105 lines, 95 tests green

Committed d9be56f: five obsolete dialogue modules removed (cogitator.py, derivation.py, expand_fit.py, pivot_fill.py, reentry.py, −2105 lines) plus import-bound orphaned tests test_pivot.py and test_selection.py. This also retroactively explains the earlier working-tree incident — the files had been deleted externally before the user requested it; my restore + this commit make the removal explicit and clean. Pre-removal verification: dialogue/__init__ re-exports only decoder symbols; no kept module imports the five (rationaliser imports kalvin/cogitator.py — a DIFFERENT module, the slow-path Cogitator also re-exported there and referenced by training supervisor's own Cogitator class); test_hop's h.reentry is a harness-state attribute not the module. Engine/harness docstrings naming :class:`ExpandFit` were stale (engine imports hadn't included it) — reworded to actual behavior. 95 tests green.

*Relevance: medium*
*Context: Committing obsolete dialogue module removal*
*Tags: dialogue cleanup committed modules*

---
*Observed: 2026-09-21T12:20:59.606Z*
