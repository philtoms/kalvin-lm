---
type: source
title: "Observation: Engine ask-atom branching removed; structural routing + refusal guard landed"
tags:
  - ask
  - engine
  - ks2
  - cogitator
status: observation
created: 2026-09-12
updated: 2026-09-12
slug: obs-2026-09-12-engine-ask-atom-branching-removed-structural-routing-refusal
relevance: high
observed_at: 2026-09-12T10:38:07.724Z
source_context: Removing engine branching on the ask atom per settled ks2 position
---

# ⭐ Observation: Engine ask-atom branching removed; structural routing + refusal guard landed

Committed 74eca06 (wiki: ks2 conformance review + settled ask-atom position), then implemented the engine update: removed all engine branching on the ask atom. Changes: (1) engine.py cogitate gate now `is_misfit` only — ask-marked klines with nodes reach the strategy as misfits by their ask-atom gap, no mark read; docstring updated to structural routing; (2) cogitator.py query selection now structural: `find_canon(entry.signature) or entry` — held canon is the known decomposition, else the entry's own nodes (asks find no canon under their marked key, so they fall to their own nodes, identical behaviour); (3) expand_fit.py same pattern fixed (also repaired dead findCanons→find_canon); (4) NEW refusal guard in Cogitator's yield: `is_refused` check, honouring EngineState.refuse's existing "not to be re-proposed" contract, which Cogitator violated latently. Verified: 35 tests pass; ask flows byte-identical on all three scripts (wdmh-underfit, mhall, wdmh-underfit-o). Behaviour deltas are the ks2-specified consequences, previously hidden by mark-gated silence: goal-less misfits now emit the structural ask (e.g. aDet:[]) plus ungrounded proposals (e.g. aDet:[a] one-hop, Mary:[a] 30-hop), each exactly once (refusal guard), declined by supervisor/harness S4 replies. Deep-walk proposals like Mary:[a] show as S3 due to the known raw-hop-count-as-byte inversion bug (review item 7) — separate pending fix. Engine no longer reads is_ask anywhere except harness display (? prefix, surface presentation, correct).

*Relevance: high*
*Context: Removing engine branching on the ask atom per settled ks2 position*
*Tags: ask engine ks2 cogitator*

---
*Observed: 2026-09-12T10:38:07.724Z*
