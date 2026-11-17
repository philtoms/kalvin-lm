---
type: source
title: "Observation: Cogitator ancestor walk was O(depth) per candidate — chain frozenset fixed 116s→3.5s"
tags:
  - performance
  - cogitator
  - tests
status: observation
created: 2026-11-17
updated: 2026-11-17
slug: obs-2026-11-17-cogitator-ancestor-walk-was-o-depth-per-candidate-chain-froz
relevance: high
observed_at: 2026-11-17T16:51:06.794Z
source_context: Diagnosing slow property tests
---

# ⭐ Observation: Cogitator ancestor walk was O(depth) per candidate — chain frozenset fixed 116s→3.5s

Test-suite slowness root-caused and fixed: the two hypothesis property tests (test_rationalisation_is_honest, test_no_fabrication) took ~115s of the 116s suite. cProfile showed 98% in Cogitator.ancestor — 167k calls at ~244µs each: an O(parent-chain-depth) walk per candidate per push, with MAX_ENTRIES=4096 allowing chains thousands deep → O(pushes × depth × |domain|). Fix: _Entry now carries `chain: frozenset` (the ask signatures of its strict parent chain), inherited at push time as parent.chain | {parent.A} (frozenset union, O(depth) once per push instead of per candidate). ancestor() is a membership test; common_ancestor() is chain(l) & chain(r.parent). KNode hashes as its int so set semantics match the old == walk exactly. Suite: 116s → 3.5s; both trace tables byte-identical (verified via git stash diff).

*Relevance: high*
*Context: Diagnosing slow property tests*
*Tags: performance cogitator tests*

---
*Observed: 2026-11-17T16:51:06.794Z*
