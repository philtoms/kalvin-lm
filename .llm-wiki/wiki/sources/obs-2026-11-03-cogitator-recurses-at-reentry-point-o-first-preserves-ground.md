---
type: source
title: "Observation: Cogitator recurses at reentry point — O-first preserves grounding, trace collapses 4x"
tags:
  - trs
  - cogitator
  - recursion
  - refactor
  - mechanism
status: observation
created: 2026-11-03
updated: 2026-11-03
slug: obs-2026-11-03-cogitator-recurses-at-reentry-point-o-first-preserves-ground
relevance: high
observed_at: 2026-11-03T14:24:52.128Z
source_context: "Cogitator recursion step 1: stack-to-recursion refactor, mechanism only"
---

# ⭐ Observation: Cogitator recurses at reentry point — O-first preserves grounding, trace collapses 4x

Recursion step landed in Cogitator (mechanism only): explore() recurses into each unsettled side at the point of discovery instead of pushing onto the shared stack; _Entry dataclass and the while-loop are gone. Frames remain GLOBAL (path-local is the next step). Two findings: (1) U-first recursion BROKE wdmh grounding — WDH's subtree explored before HALL's, WDH's DH stall found no witness, WDH unwound exhausted, and the later exchange arrived too late: the stall-resumption fork is not deferrable, DFS order decides who sees the witness. Fix for this step: O-side recursed FIRST, reproducing the flat stack's LIFO discipline (last-pushed-first); with O-first, WDH's DH stall meets HALL's earlier stall, exchanges inline, frame(DH) mints, and the pairing's fresh framed() check closes it — both scripts ground IDENTICALLY (stores/halts/grounded rows match the flat run; tests 162/15/16). (2) The re-surface mechanism is structurally replaced: closes fire inline at the pairing's own check (framed(U) and framed(O) evaluated AFTER the descents return) — one pass per entry, no second-pass re-pairing. Trace collapsed: wdmh 676→158 rows, mhall 281→67; every pairing appears once; depth-first order now matches the reference's RESOLVE/BACK-PROPAGATE shape. Note: reference doc says "the u-side leading" but the flat Cogitator's LIFO was O-first; this step preserves Cogitator behaviour — u-vs-o leading order is a semantic question for a later ruling.

*Relevance: high*
*Context: Cogitator recursion step 1: stack-to-recursion refactor, mechanism only*
*Tags: trs cogitator recursion refactor mechanism*

---
*Observed: 2026-11-03T14:24:52.128Z*
