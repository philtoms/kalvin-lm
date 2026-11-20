---
type: source
title: "Observation: decompose removed; effort inlines U and O"
tags:
  - rationalisation
  - cleanup
status: observation
created: 2026-11-17
updated: 2026-11-17
slug: obs-2026-11-17-decompose-removed-effort-inlines-u-and-o
relevance: medium
observed_at: 2026-11-17T13:33:32.807Z
source_context: Removing algebra.decompose and Decomposition
---

# 🔍 Observation: decompose removed; effort inlines U and O

algebra.Decomposition and decompose() removed. The one production consumer was effort() (halts/proposals) — its U and O now computed inline via underfit(); the wasted fit() computation decompose did is gone too. Dependent cleanup: two decomposition-law tests removed (model + properties); properties' oracle_band rewritten directly on fit/underfit/overfit (same independent formulation, no Decomposition record); dataclass import dropped from algebra; overfit added to the rationalisation package exports to complete the fit/underfit/overfit trio the package already two-thirds exported. effort values verified byte-identical to HEAD over both canonical scripts. Suite: 153 passed, 16 xfailed, 2 pre-existing goal-less failures.

*Relevance: medium*
*Context: Removing algebra.decompose and Decomposition*
*Tags: rationalisation cleanup*

---
*Observed: 2026-11-17T13:33:32.807Z*
