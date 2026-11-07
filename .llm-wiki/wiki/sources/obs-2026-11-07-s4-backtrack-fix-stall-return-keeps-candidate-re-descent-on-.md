---
type: source
title: "Observation: S4 backtrack fix: stall-return keeps candidate, re-descent on frame growth"
tags:
  - cogitator
  - s4
  - backtrack
  - candidates
  - frames
status: observation
created: 2026-11-07
updated: 2026-11-07
slug: obs-2026-11-07-s4-backtrack-fix-stall-return-keeps-candidate-re-descent-on-
relevance: high
observed_at: 2026-11-07T09:21:52.590Z
source_context: "Fixing wdmh step-4 exhaustion: S4 backtrack loses the descended component"
---

# ⭐ Observation: S4 backtrack fix: stall-return keeps candidate, re-descent on frame growth

Fixed the Cogitator's S4-backtrack defect; wdmh.ks step 4 now grounds the claim WDMH:[MHALL] (all four stores land; tests 162/15/16). Three defects found: (1) stall-return was too aggressive — pop() discarded the caller's CURRENT candidate when a child stalled at S4, so the sibling side (O after U, or U after O) was never tried for that candidate, violating the module's own pseudocode (explore U → explore O → next candidate); (2) the seen string leaked across candidates (never reset on the stall path), permanently barring the stalled side's descent for every later candidate at that level — this suppressed honest stalls (Mod etc. never formally stalled before); (3) push() referenced the closure variable `call` before assignment — goal-less entries crashed with NameError (latent in HEAD too; test_no_fabrication was failing on it). Fix in src/rationalisation/cogitator.py: stall branch now does next(call) + continue (backtrack keeps the caller's candidate); `seen: str` replaced by `at: dict` (side → frame count at its last descent); fresh(side) allows re-descent only when new frames arrived since the last descent (progress condition — preserves T1 termination, hard-capped by MAX_PAIRINGS anyway). The grounding path: d1 WDMH×MHALL → U=WDH stalls DH (side U); O=HLAL descent contains MH×had:did have → DH stalls side O → S3 exchange mints frame(DH) → HLAL closes → backtrack to d1 → RE-descend U (frames grew) → WDH×what closes (DH framed) → frame(WDH) → d1 closes → claim grounds. Cogitator's stall-return and drain-return are now the same operation (next); pop() deleted.

*Relevance: high*
*Context: Fixing wdmh step-4 exhaustion: S4 backtrack loses the descended component*
*Tags: cogitator s4 backtrack candidates frames*

---
*Observed: 2026-11-07T09:21:52.590Z*
