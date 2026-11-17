---
type: source
title: "Observation: Word registry drives derived labels — fit/underfit/overfit decompose bits"
tags:
  - algebra
  - labels
  - mts
  - word-registry
  - rationalisation
status: observation
created: 2026-11-17
updated: 2026-11-17
slug: obs-2026-11-17-word-registry-drives-derived-labels-fit-underfit-overfit-dec
relevance: high
observed_at: 2026-11-17T16:41:48.194Z
source_context: Making _provenance usable — meaningful derived labels
---

# ⭐ Observation: Word registry drives derived labels — fit/underfit/overfit decompose bits

Derived-value labels (fit/underfit/overfit) now decompose the VALUE's word bits via a session word registry — not label-string arithmetic. kalvin.kline gained WordLabels (dict[int,str], content-space word value → word), the using_word_labels contextvar context (mirrors using_resolver), and mts_initials(value): one uppercase character per registered component word in allocation order; empty when no registry is active or bits remain unregistered. algebra._derived: zero → ⊥, else mts_initials(v) or the _provenance fallback ("WDMH∖MHALL") which needs both operand labels. The harness installs the registry in Runner.run around priming+steps (built from word_bits, {(bit<<32): word}). User's canonical example now holds exactly: A=WDMH vs C=MHALL → F=M, U=WDH, O=HALL.

*Relevance: high*
*Context: Making _provenance usable — meaningful derived labels*
*Tags: algebra labels mts word-registry rationalisation*

---
*Observed: 2026-11-17T16:41:48.194Z*
