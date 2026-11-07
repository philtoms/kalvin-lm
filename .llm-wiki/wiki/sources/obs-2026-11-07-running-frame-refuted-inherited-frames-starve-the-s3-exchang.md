---
type: source
title: "Observation: Running frame refuted: inherited frames starve the S3 exchange economy"
tags:
  - cogitator
  - frames
  - exchange
  - witnesses
  - experiment
status: observation
created: 2026-11-07
updated: 2026-11-07
slug: obs-2026-11-07-running-frame-refuted-inherited-frames-starve-the-s3-exchang
relevance: high
observed_at: 2026-11-07T13:53:49.758Z
source_context: Investigating running frame shared between harness steps
---

# ⭐ Observation: Running frame refuted: inherited frames starve the S3 exchange economy

Experiment: running frame (self.frame initialized in Cogitator.__init__, shared across rationalise calls / harness steps) REFUTED the expectation that outcomes stay the same with only pairing counts varying. wdmh step 3 collapsed 93→5 pairings and LOST the claim (exhausted + unwitnessed DH stall). Mechanism: step 1 (MHALL×SVO) leaves 10 frames — including frame(HALL)=had|a|little|lamb and frame(Object). Step 3's opening pairing WDMH×MHALL has O=HLAL(=HALL content) — settles INSTANTLY via the inherited frame; WDH×what's O=Object settles instantly too. The entire O-side descent (HLAL's 11 candidates) never runs, so `MH × had:did have → O=DH stall (side O)` never happens, the S3 exchange with the side-U DH stall never fires, frame(DH) is never minted, WDH never closes, the claim never grounds. Structural lesson: the S3 exchange economy is witness-starved by inherited frames — the DH exchange is a BYPRODUCT of re-deriving HLAL's construction within the same run; frame(HALL) earned in step 1's context cannot substitute because what step 3 needs is not HALL's settlement but the witnesses produced while earning it. Cross-step sharing of frames is therefore not a pure memoisation: per-run frame reset is load-bearing for grounding. (mhall alone is unaffected — single scenario; scenario-internal sharing is inert since goal steps are trivial S1 canon groundings.) Code reverted to per-run frames (git checkout); tests passed either way — the regression is invisible to the suite, only the scripts show it.

*Relevance: high*
*Context: Investigating running frame shared between harness steps*
*Tags: cogitator frames exchange witnesses experiment*

---
*Observed: 2026-11-07T13:53:49.758Z*
