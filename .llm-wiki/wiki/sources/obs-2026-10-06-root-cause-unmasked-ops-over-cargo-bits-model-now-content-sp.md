---
type: source
title: "Observation: Root cause: unmasked ops over cargo bits; model now content-space"
tags:
  - trs
  - realisation
  - word-mask
  - content-space
  - appendix
  - root-cause
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-root-cause-unmasked-ops-over-cargo-bits-model-now-content-sp
relevance: high
observed_at: 2026-10-06T08:09:22.235Z
source_context: Investigating why residues could strip type bits
---

# ⭐ Observation: Root cause: unmasked ops over cargo bits; model now content-space

ROOT CAUSE FOUND for the label-less halted entries (user's question: 'why can residues strip type bits?'): src/trs was running the RAW lattice — residue = x & ~y, meet = x & y, mu = popcount(v) — over FULL 64-bit compiled values. Per kalvin-algebra's appendix (The word-bit realisation), the token half (bits 0–31) is INERT ENCODING CARGO: 'masked off at every content read', MASK = 0x7FFF_FFFF_0000_0000; ∧, ∖, μ, and content equality are all masked operations; only the word half (bits 32–62) is content; bit 63 is the ASK marker. Unmasked ∖ let token-half debris participate as content, producing values like 0x5dc (a bare token-id fragment, no word bits) — which is why residues appeared to 'strip type bits' and why no word cover could label them. FIX: the TRS model operates in CONTENT SPACE — mask once at the harness boundary (dev/dialogue/trs_harness.py: WORD_MASK applied to entry sig and nodes in _to_kline); the model code itself is realisation-agnostic and unchanged (its tests use cargo-free values where masking is identity). Consequences: residues are now always exact word unions (had|a|little|lamb — no +0x debris, no token decoding needed — the reverted display hack is moot); script labels carried over under the mask so entries render MHALL:[...] again; μ is now genuinely the word-bit count (appendix's masked popcount). FURTHER EVIDENCE for the pending illegal-store ruling: step 2 now stores Subject:[Mary] — the S3 VARIANT of the vacuous proof: R3 with a no-op substitution (N={Mary} → N′={Mary}, nothing dereferenced) then grounds at the held identity, licensing 'Subject means Mary'. So the vacuous-proof class includes BOTH the R2 fit-branch fetch AND R3-without-substitution; the proposed 'R1 licenses only via R3-derived groundings' needs tightening to 'via an R3 substitution that actually changed N' (or an information-flow condition). 158 tests green. Uncommitted.

*Relevance: high*
*Context: Investigating why residues could strip type bits*
*Tags: trs realisation word-mask content-space appendix root-cause*

---
*Observed: 2026-10-06T08:09:22.235Z*
