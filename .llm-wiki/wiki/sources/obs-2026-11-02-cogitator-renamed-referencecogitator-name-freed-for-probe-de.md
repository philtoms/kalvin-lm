---
type: source
title: "Observation: Cogitator renamed ReferenceCogitator; name freed for probe-derived successor"
tags:
  - rationalisation
  - reference
  - rename
  - cogitator
status: observation
created: 2026-11-02
updated: 2026-11-02
slug: obs-2026-11-02-cogitator-renamed-referencecogitator-name-freed-for-probe-de
relevance: high
observed_at: 2026-11-02T17:38:30.217Z
source_context: Renaming the reference model before deriving the new model from the probe
---

# ⭐ Observation: Cogitator renamed ReferenceCogitator; name freed for probe-derived successor

Renamed the reference implementation: class Cogitator → ReferenceCogitator (src/rationalisation/model.py), freeing the name Cogitator for the successor model to be derived from the probe (dev/probes/probe_simple_reentry.py). Updated: rationalisation __init__ exports, verify.py comment, dev/dialogue/harness.py + trace.py (attribute self.cogitator kept — the role name), the three test_rationalisation_* files, and docs/rationalisation.md §2.2 ("One ReferenceCogitator per rationalisation"). Class docstring now opens with the reference-status note. NOT touched: src/kalvin/cogitator.py (a cogitate() function — the older engine) and src/training/* whose "Cogitator" mentions are prose about that older pipeline. CONTEXT.md glossary keeps "Cogitator" as the concept term (ubiquitous language) — the class rename marks the implementation. Verified: suite 162 passed / 15 skipped / 16 xfailed unchanged; harness grounds wdmh; probe reports 3× "✓ agrees with ref"; training imports OK. No commit made — changes pending user confirmation.

*Relevance: high*
*Context: Renaming the reference model before deriving the new model from the probe*
*Tags: rationalisation reference rename cogitator*

---
*Observed: 2026-11-02T17:38:30.217Z*
