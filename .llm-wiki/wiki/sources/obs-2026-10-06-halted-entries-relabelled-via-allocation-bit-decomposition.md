---
type: source
title: "Observation: Halted entries relabelled via allocation-bit decomposition"
tags:
  - trs
  - harness
  - display
  - labels
  - residuals
  - decomposition
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-halted-entries-relabelled-via-allocation-bit-decomposition
relevance: medium
observed_at: 2026-10-06T07:53:08.078Z
source_context: Restoring labels to halted entries in the TRS harness display
---

# 🔍 Observation: Halted entries relabelled via allocation-bit decomposition

DISPLAY FIX (trs_harness): halted entries had lost their labels — R2 residual values (U = σ∖â) rendered as bare hex (0x1e00001a00:[]) because the presenter only exact-matched script labels. Root cause found by probing the sig layout: word SIGNATURES are (alloc_bit << 32) | token_bits — the high region is a clean one-bit-per-word partition (from the compiler's word_bits allocation), the low token bits intersect and cannot cover. Fix: display-side decomposer _cover() in dev/dialogue/trs_harness.py — whole known sigs cover first (compound labels), then the allocation region (bit<<32 → word, authored order), leftover fragments as +0xhex. The algebra keeps nodes opaque — only the presenter looks inside. Effect: the halted residual now reads 'had|a|little|lamb|+0x1a00' — the sentence minus Mary, exactly what the misfit product carried. 158 tests green. NOTE: the illegal-store investigation (MHALL:[Mary] — vacuous R2 fit-branch proofs licensing S2 openers) REMAINS PENDING the user's ruling on the proposed fix (R1 licenses only via R3-derived groundings or self-S1 openers; plus the pure-overfit clause question and the harness seeding bypass of D10).

*Relevance: medium*
*Context: Restoring labels to halted entries in the TRS harness display*
*Tags: trs harness display labels residuals decomposition*

---
*Observed: 2026-10-06T07:53:08.078Z*
