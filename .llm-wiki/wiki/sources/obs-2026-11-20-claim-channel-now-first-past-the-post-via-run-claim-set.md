---
type: source
title: "Observation: Claim channel now first-past-the-post via run claim set"
tags:
  - cogitator
  - claims
  - semantics
status: observation
created: 2026-11-20
updated: 2026-11-20
slug: obs-2026-11-20-claim-channel-now-first-past-the-post-via-run-claim-set
relevance: medium
observed_at: 2026-11-20T07:36:49.985Z
source_context: Adding first-claim-past-the-post filtering to the claim channel
---

# 🔍 Observation: Claim channel now first-past-the-post via run claim set

First-past-the-post claim semantics added to the Cogitator: the claim kline is a pure function of the run (KLine(run.ask.signature, [run.goal.signature]) — constant per run), so every duplicate is an exact duplicate by construction. Filter at the source: _Run.claims (set of fingerprint(claim) = sig | can(nodes), using the user's algebra.fingerprint) gates the broadcast — the first S2 alignment with the goal broadcasts, later ones fall through to a plain "framed" event so the trace still records the alignment. Verified with a synthetic run producing three goal alignments: one claim broadcast, two framed rows. Cost: one set lookup per goal alignment (set holds ≤1 element per run today, generalizes if claims ever become per-entry). Considered sink-side dedupe (harness _claim set) but rejected: the claim is the run's output ("the claim is the ask's, whatever entry aligned") so first-past-the-post is run-level bookkeeping like run.entries/live, not sink policy; the sink stays a dumb holder. docs/rationalisation.md ALIGN/WALK/§2.4 updated.

*Relevance: medium*
*Context: Adding first-claim-past-the-post filtering to the claim channel*
*Tags: cogitator claims semantics*

---
*Observed: 2026-11-20T07:36:49.985Z*
