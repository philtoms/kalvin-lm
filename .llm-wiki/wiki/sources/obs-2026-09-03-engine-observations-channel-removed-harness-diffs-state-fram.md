---
type: source
title: "Observation: Engine observations channel removed; harness diffs state; frame in summary"
tags:
  - dialogue
  - engine
  - harness
  - refactoring
status: observation
created: 2026-09-03
updated: 2026-09-03
slug: obs-2026-09-03-engine-observations-channel-removed-harness-diffs-state-fram
relevance: high
observed_at: 2026-09-03T16:34:47.730Z
source_context: Simplifying the engine↔harness observation channel in the dialogue harness
---

# ⭐ Observation: Engine observations channel removed; harness diffs state; frame in summary

Removed Engine.rationalise's second return value (internal S1 groundings). rationalise now returns only the dialogue batch; Engine.observations deleted. The harness's _drive builds turn.grounds by diffing a (frame ∪ ltm) snapshot keyed by (signature, nodes) before/after each rationalise call (harness._grounded_snapshot). Side effect: the doubled "grounds" lines from the cascade loop are gone (diff is idempotent). Also added a "frame (framed, not yet grounded)" section to the run summary (_render_frame), listing frame entries with no isomorphic LTM entry — visible on wdmh-underfit (identities sit in frame only), empty on mhall where everything promotes to LTM. Dev probes probe_block_filter.py / probe_rationalise.py updated to the single-return API.

*Relevance: high*
*Context: Simplifying the engine↔harness observation channel in the dialogue harness*
*Tags: dialogue engine harness refactoring*

---
*Observed: 2026-09-03T16:34:47.730Z*
