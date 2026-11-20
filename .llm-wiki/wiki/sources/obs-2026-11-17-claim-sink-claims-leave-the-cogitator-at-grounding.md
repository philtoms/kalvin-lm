---
type: source
title: "Observation: Claim sink: claims leave the Cogitator at grounding"
tags:
  - rationalisation
  - cogitator
  - interface
  - claims
  - streaming
status: observation
created: 2026-11-17
updated: 2026-11-17
slug: obs-2026-11-17-claim-sink-claims-leave-the-cogitator-at-grounding
relevance: critical
observed_at: 2026-11-17T13:59:04.041Z
source_context: Claim sink interface between Cogitator and harness
---

# 🔴 Observation: Claim sink: claims leave the Cogitator at grounding

Claim sink interface landed (first step toward continuous cogitation, per user): Rationalisation no longer carries `claim` — claims leave the Cogitator at grounding through an `on_claim` callback (Cogitator.__init__ arg), emitted once per run at first grounding, store-then-announce; later re-alignments keep tracing `grounded` events but re-announce nothing. The harness Runner is the claim sink: `Runner._claim` appends (step_index, claim) to `runner.claims` mid-run (`_step_index` set before rationalise); display reads the ledger (present() takes a claims param). ProbeRunner.probe sets the same bookkeeping. probe_head_events and probe_simple_reentry reworked onto the ledger. Verified: ledger ≡ old return value step-for-step on both scripts; claim demonstrably lands before rationalise returns. Doc §2.1/§2.2/§2.3/§2.4 now speak the claim sink. Design decisions made: constructor-arg sink (Cogitator outlives runs), emit-once-per-run (sink need not dedupe).

*Relevance: critical*
*Context: Claim sink interface between Cogitator and harness*
*Tags: rationalisation cogitator interface claims streaming*

---
*Observed: 2026-11-17T13:59:04.041Z*
