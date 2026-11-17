---
type: source
title: "Observation: ReferenceCogitator retired; instruments and witness index removed"
tags:
  - rationalisation
  - cleanup
  - retirement
status: observation
created: 2026-11-17
updated: 2026-11-17
slug: obs-2026-11-17-referencecogitator-retired-instruments-and-witness-index-rem
relevance: critical
observed_at: 2026-11-17T13:14:34.483Z
source_context: Retiring the ReferenceCogitator baseline
---

# 🔴 Observation: ReferenceCogitator retired; instruments and witness index removed

ReferenceCogitator retired and removed per user decision. Deleted: src/rationalisation/model.py (ReferenceCogitator, _Pairing, _Reentry), verify.py (certificate checker), existence.py (existence checker — both instrumented only the reference), dev/dialogue/trace.py (narrated reference records), tests/test_rationalisation_certificate.py and test_rationalisation_existence.py (parked, skip-marked). Reference-only support removed with it: algebra.fingerprint (free function added one turn earlier — no user left), Domain.witness + the _canonical first-stored index (the Cogitator never consults witnesses), and the doc §1 fingerprint/witness sentences. probe_simple_reentry.py demoted from comparison driver to trace driver (its _ref_runner/ReferenceCogitator half removed); its _labels_for/_tlabel exports remain live for probe_head_events. docs/rationalisation.md §2 rewritten to describe the Cogitator's actual machine (entry chain, at-map fresh rule, MAX_ENTRIES cap, most-recently-held-first candidate consumption); §2.4/§3 rewritten for its invariants and bounds; §4 now records the instruments as retired.

*Relevance: critical*
*Context: Retiring the ReferenceCogitator baseline*
*Tags: rationalisation cleanup retirement*

---
*Observed: 2026-11-17T13:14:34.483Z*
