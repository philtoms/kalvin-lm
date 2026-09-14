---
type: source
title: "Observation: Pure-algebra probe: 7/7 exact match; Def 17 refine gap + T2 keying flagged"
tags:
  - algebra
  - verification
  - probe
  - worked-example
  - def17
  - refine-gap
  - t2-keying
status: observation
created: 2026-09-14
updated: 2026-09-14
slug: obs-2026-09-14-pure-algebra-probe-7-7-exact-match-def-17-refine-gap-t2-keyi
relevance: high
observed_at: 2026-09-14T06:19:43.905Z
source_context: Verifying the §9 worked example against pure algebra code
---

# ⭐ Observation: Pure-algebra probe: 7/7 exact match; Def 17 refine gap + T2 keying flagged

Stripped dev/dialogue/probe_wdmh_expand.py to a pure-algebra verification of the §9 worked example: no engine/compiler/tokenizer imports — values are plain ints (one bit per atom), klines a (signature, nodes, acq_depth) dataclass, the doc's 9-shape fit classifier implemented directly. Memory (7 klines), A0 = wdmh:[w,d,m,h], B = mhall:[m,h,a,l,l] injected verbatim; derivation runs Defs 12-17; verdict block asserts each documented expectation. RESULT: 7/7 PASS — exact reproduction: Step 1 {d,h}⇉[dh] under dh:[d,h] → [w,dh,m] Δ4; Step 2 dh⇉[h] under dh:[h] Denotation forward → [w,h,m] Δ4→3 (J=2/5 at this state, matching §11's example start); Step 3 walk w:[w]→w:[o](fwd w:[o])→w:[all](rev all:[o] mirror)→w:[a,l,l](canon expand), absorb w:[a,l,l] acq_depth 3; Step 4 w⇉[a,l,l] Δ3→0; done, C(A,B)=Canon, witness ≡[h,m,a,l,l] multiset, m never replaced, Ĥ=9/5, γ=2^(-9/5)≈0.287. TWO SPEC-LETTER GAPS the test exposed: (1) Def 17's end-condition ("reaches content overlapping the goal's excess") fires at w:[all] — content {a,l} already IS the excess — one canon edge before the documented absorption at w:[a,l,l]; the probe implements an explicit refine-before-absorb step (expand terminal's compound nodes under held well-founded witnesses) and flags this — Def 17 needs a sentence sanctioning it (or the example/§11 must move to depth 2). (2) T2's no-revisit ("each consumed correspondence signature used at most once") read as bare signature would block the documented walk, which consumes signature all twice (all:[o] then all:[a,l,l]); keying must be on correspondence identity (signature,witness). Presented the amendment fork to Phil; not committed.

*Relevance: high*
*Context: Verifying the §9 worked example against pure algebra code*
*Tags: algebra verification probe worked-example def17 refine-gap t2-keying*

---
*Observed: 2026-09-14T06:19:43.905Z*
