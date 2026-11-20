---
type: source
title: "Observation: Claim broadcast relaxed: every S2 alignment broadcasts"
tags:
  - rationalisation
  - cogitator
  - claims
  - streaming
status: observation
created: 2026-11-17
updated: 2026-11-17
slug: obs-2026-11-17-claim-broadcast-relaxed-every-s2-alignment-broadcasts
relevance: high
observed_at: 2026-11-17T14:37:48.024Z
source_context: Relaxing claim broadcast to many-per-run
---

# ⭐ Observation: Claim broadcast relaxed: every S2 alignment broadcasts

Claim broadcast relaxed (user-directed): the once-per-run gate (self.claimed) is removed — every S2 alignment with the goal broadcasts KLine(ask.signature, [goal.signature]) through on_claim. A run may broadcast many identical claims; the channel assumes no uniqueness, the sink receives whatever arrives. Measured: the simple ABC/AB test scenario broadcasts 4; mhall 1; wdmh step 2 broadcasts 5. The harness ledger records all (display unchanged — any() lookup). Doc §2.2/§2.3/§2.4 updated to 'a run may broadcast many'. Committed before this change: 5fdf4d6 (storeless Cogitator). This relaxation is uncommitted.

*Relevance: high*
*Context: Relaxing claim broadcast to many-per-run*
*Tags: rationalisation cogitator claims streaming*

---
*Observed: 2026-11-17T14:37:48.024Z*
