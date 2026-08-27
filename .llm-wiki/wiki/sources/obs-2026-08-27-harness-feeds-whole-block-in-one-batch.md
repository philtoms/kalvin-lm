---
type: source
title: "Observation: Harness feeds whole block in one batch"
tags:
  - harness
  - dialogue
  - feeding
status: observation
created: 2026-08-27
updated: 2026-08-27
slug: obs-2026-08-27-harness-feeds-whole-block-in-one-batch
relevance: high
observed_at: 2026-08-27T14:08:20.028Z
source_context: Refactoring src/dialogue/harness.py feeding behaviour
---

# ⭐ Observation: Harness feeds whole block in one batch

Committed edd64e8 on branch dialogue: Harness.run now feeds each annotation group's full entry list as the first batch (queue starts as list(group)) instead of only the opener; opener remains the display entry. Side effect: far fewer engine asks (mhall shows 1 ask total) since script answers arrive upfront; _answer/escalation only fire for cross-block asks. mhall.ks and wdmh-underfit.ks run clean; pre-existing test failures unchanged.

*Relevance: high*
*Context: Refactoring src/dialogue/harness.py feeding behaviour*
*Tags: harness dialogue feeding*

---
*Observed: 2026-08-27T14:08:20.028Z*
