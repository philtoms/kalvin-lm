---
type: source
title: "Observation: Grounding broadened to all significance levels; glossary is normative"
tags:
  - grounding
  - frame
  - ltm
  - promotion
  - glossary
  - domain-model
status: observation
created: 2026-08-17
updated: 2026-08-17
slug: obs-2026-08-17-grounding-broadened-to-all-significance-levels-glossary-is-n
relevance: critical
observed_at: 2026-08-17T11:52:25.090Z
source_context: Redefining Grounding in CONTEXT.md; glossary confirmed normative (code follows glossary)
---

# 🔴 Observation: Grounding broadened to all significance levels; glossary is normative

The user redefined the CONTEXT.md Grounding entry and confirmed a deliberate broadening with a normative framing I had wrong: the glossary defines what the code NEEDS TO FOLLOW, not what it already says.

New Grounding entry (now in the Rationalisation section, beside Model/Memory/Frame/LTM): "The model's mechanism for realising significance. If a signature is grounded, then Kalvin knows that all of its nodes are grounded also. KLines grounded in a Frame are available for cogitation. KLines grounded in LTM are frame promotions that Kalvin deems important enough to remember."

Key semantics:
1. Grounding realises ANY significance level, not just S1. S1 is what full commitment looks like; grounding is admission into memory at whatever level Kalvin understands something. A kline understood at S2/S3 can be Frame-grounded where cogitation re-traverses it.
2. The broadening explains the Frame/LTM split: Frame-grounded = available for cogitation; LTM-grounded = frame promotions deemed important enough to remember.
3. "Important enough to remember" in practice: LTM promotion is earned by PARTICIPATION IN FURTHER RATIONALISATION — the kline led through cogitation to a previously grounded or ratified proposal. Importance is demonstrated usefulness, not static criteria (replaces the discarded MIN_EDGES edge-count idea).
4. The invariant form ("if grounded then nodes grounded") is the better glossary form vs my precondition form; it absorbs the identity exception trivially ({S:[S]} node is itself).

Current engine implements only the S1-shaped special case (_is_groundable: identity or all-nodes-grounded). Frame-grounding at S2/S3 and promotion-by-participation are unimplemented — the direction the code must grow.

Synced wiki/concepts/grounding.md to the normative entry, with the engine's current state marked as the S1 special case.

*Relevance: critical*
*Context: Redefining Grounding in CONTEXT.md; glossary confirmed normative (code follows glossary)*
*Tags: grounding frame ltm promotion glossary domain-model*

---
*Observed: 2026-08-17T11:52:25.090Z*
