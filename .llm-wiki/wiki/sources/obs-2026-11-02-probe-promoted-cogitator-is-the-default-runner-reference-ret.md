---
type: source
title: "Observation: Probe promoted: Cogitator is the default runner; reference retained as baseline"
tags:
  - rationalisation
  - cogitator
  - promotion
  - default
status: observation
created: 2026-11-02
updated: 2026-11-02
slug: obs-2026-11-02-probe-promoted-cogitator-is-the-default-runner-reference-ret
relevance: critical
observed_at: 2026-11-02T17:53:51.634Z
source_context: Promoting the probe model to the default Cogitator
---

# 🔴 Observation: Probe promoted: Cogitator is the default runner; reference retained as baseline

Promoted the probe to the default model: new module src/rationalisation/cogitator.py defines Cogitator (the simple reentry loop, productionised from dev/probes/probe_simple_reentry.py) and rationalise; rationalisation/__init__.py now exports those as the defaults, with ReferenceCogitator (model.py) retained as baseline (model.py's own one-shot rationalise is no longer exported). dev/dialogue/harness.py Runner defaults to the new Cogitator; trace() (the reference's optimal-path reading) runs only under isinstance ReferenceCogitator. The engine reuses the reference's report types — Rationalisation/Row/End — emitting one Row per pairing (plus stall rows), events as (row,"pair"/"close"), path = the frame-minting witness rows, so the harness's existing trace table renders the new engine unchanged. Boundary rules (R4, S1-on-receipt) and the claim store mirror the reference preamble. Bounds: MAX_DEPTH=10, MAX_PAIRINGS=4096. Two bugs caught while wiring: (1) a goal-less run can ground at depth 1 via a domain candidate — the claim rule must apply only when a goal was given; (2) I reintroduced the pseudocode's `break` after inner grounds — exactly the probe's starves-the-exchange bug; only the claim breaks. Measured: all canonical scenarios agree with the reference (mhall MHALL:[SVO] 168 rows; wdmh MHALL 178 rows; WDMH:[MHALL] 138 rows, claim row last). Notable difference: the new engine files ends=0 on grounding runs where the reference files 64-74 exhaustion halts — it has no path bar, so nothing halts to propose; whether the training protocol needs those proposals is an open question. Property tests (no-fabrication, is-honest) now run against the new engine via rationalise and pass. Suite 162/15/16. dev/probes/probe_simple_reentry.py rewritten as a pure comparison driver over the production engine (its embedded loop deleted).

*Relevance: critical*
*Context: Promoting the probe model to the default Cogitator*
*Tags: rationalisation cogitator promotion default*

---
*Observed: 2026-11-02T17:53:51.634Z*
