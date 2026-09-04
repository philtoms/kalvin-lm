---
type: source
title: "Observation: Cogitator.expand dead: connotateY frontier never seeded"
tags:
  - dialogue
  - cogitator
  - engine
  - bugfix
status: observation
created: 2026-09-04
updated: 2026-09-04
slug: obs-2026-09-04-cogitator-expand-dead-connotatey-frontier-never-seeded
relevance: high
observed_at: 2026-09-04T09:18:40.570Z
source_context: "Dialogue-dev tuning: WDMH underfit proposal investigation"
---

# ⭐ Observation: Cogitator.expand dead: connotateY frontier never seeded

Two bugs found and fixed in src/dialogue/cogitator.py expand/connotateY (uncommitted refactor fallout): (1) connotateY's frontier was initialised empty — the old `frontier=[sig]` seed was dropped in the KLine-based rewrite, so the generator yielded nothing and expand ALWAYS returned ([], 0); no proposal was possible for any query. Fixed by seeding `frontier = [KLine(sig, [])]`. (2) The node-containment guard `all(True for n in kl.nodes if n in m_nodes)` is a filter-comprehension, vacuously True — replaced with `all(n in m_nodes for n in kl.nodes)`. Remaining structural gap: connotateY traverses only signature→kline edges (find_sig), so gap nodes like 'what'/'did'/'have' whose only resolutions are identities die at hop 0; there is no reverse (node-in-nodes → kline) index, so the WDMH→DH→hadDH→…→MHALL bridge is invisible to expand. Probe scripts: dev/dialogue/probe_wdmh_expand.py, probe_wdmh_pair.py.

*Relevance: high*
*Context: Dialogue-dev tuning: WDMH underfit proposal investigation*
*Tags: dialogue cogitator engine bugfix*

---
*Observed: 2026-09-04T09:18:40.570Z*
