---
type: source
title: "Observation: Algebra probe: -o memory derives MHALL (done); base script stuck at what"
tags:
  - algebra
  - derivation
  - probe
  - acid-test
  - mirror-clause
  - slot-walk
status: observation
created: 2026-09-14
updated: 2026-09-14
slug: obs-2026-09-14-algebra-probe-o-memory-derives-mhall-done-base-script-stuck-
relevance: high
observed_at: 2026-09-14T05:30:03.096Z
source_context: Acid-testing the §9 worked example through implemented code
---

# ⭐ Observation: Algebra probe: -o memory derives MHALL (done); base script stuck at what

Rewrote dev/dialogue/probe_wdmh_expand.py as the acid test for the §9 worked example: drives the derivation purely by the algebra (Defs 12–17) in place of the Cogitator — replace with forward/reverse mirror occurrence, canonicalisation survey (exact witnessing, goal-directed, never whole-sequence), Def 14 targeting licence (relationship band scopes region + strict Δ decrease), Def 17 slot walks (either-side occurrence, no-revisit, BFS, end at excess overlap), absorption with acq_depth = edges crossed, done at value equality. Memory compiled fresh from mhall.ks + wdmh script (shared word_bits); A0 = scope-1 MTS canon WDMH:[what,did,Mary,have]; goal = max-overlap answer-side canon (MHALL). RESULTS: wdmh-underfit-o.ks DONE — witness WDMH:[ALL,had,Mary], γ≈0.165, early completion (§7); wdmh-underfit.ks STUCK at slot what. Findings vs the doc's idealization: (1) had/have are distinct word bits → Δ0=7 not 4; Step 2 is the mirror read [DH]⇉[had] of Connotation had:[DH] (real memory holds h:[dh], not the doc's forward Denotation dh:[h]) — the mirror clause is load-bearing, carrying the shed and both object walks' exits from Object. (2) Crossover at Object works as the doc describes w/all at o: arrive forward (what:[Object]), leave reverse (lambObject:[Object]/Query:[Object]). (3) Def 17 end-at-any-excess-overlap fires at partial overlap (lamb) → two slot walks instead of one; path dependence, still done. (4) Script bug found: (Object)W > O authored before the question group binds W to its own word (not "what") — fixed by authoring W > O(bject) inside the group (data/scripts/wdmh-underfit-o.ks rewritten). (5) `W = O` (DENOTES) compiles to whatObject:[Object] whose compound head never occurs from node what — structurally unusable; only the CONNOTES form bridges. All 35 tests pass; nothing else references the -o script. Not committed.

*Relevance: high*
*Context: Acid-testing the §9 worked example through implemented code*
*Tags: algebra derivation probe acid-test mirror-clause slot-walk*

---
*Observed: 2026-09-14T05:30:03.096Z*
