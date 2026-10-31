---
type: source
title: "Observation: Harness runs multiple ask==goal scenarios"
tags:
  - dialogue-dev
  - harness
  - kscript
status: observation
created: 2026-10-31
updated: 2026-10-31
slug: obs-2026-10-31-harness-runs-multiple-ask-goal-scenarios
relevance: high
observed_at: 2026-10-31T22:54:39.729Z
source_context: Modifying the dialogue harness to run multiple ask==goal scenarios
---

# ⭐ Observation: Harness runs multiple ask==goal scenarios

dev/dialogue/harness.py now runs every resolvable ask==goal pair, not just the last. `_ask_goal_pair` → `_ask_goal_pairs` (list, ask order); `run()` primes all non-pair entries once, then for each pair runs goal step then ask step with continuous step numbering over the one live Domain — an earlier scenario's stores/proposals persist into later ones. wdmh.ks (imports mhall) now runs 2 scenarios: steps 1-2 mhall's, steps 3-4 wdmh's; the accumulation is visible — wdmh's HALL now resolves via known HV:Verb before leaning on had:did have. mhall.ks standalone output byte-identical. Same-file multi-ask verified via dev/probes/probe_two_scenarios.py. Purpose: test partial goal solutions generating S2/S3 proposals from grouped scenarios. Two dev/probes callers updated to `_ask_goal_pairs(entries)[0]`.

*Relevance: high*
*Context: Modifying the dialogue harness to run multiple ask==goal scenarios*
*Tags: dialogue-dev harness kscript*

---
*Observed: 2026-10-31T22:54:39.729Z*
