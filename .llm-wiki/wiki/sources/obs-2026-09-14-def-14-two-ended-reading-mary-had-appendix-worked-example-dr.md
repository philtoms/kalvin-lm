---
type: source
title: 'Observation: Def 14 two-ended reading + "Mary had" appendix worked example drafted'
tags:
  - kalvin-algebra
  - def14
  - appendix
  - worked-example
  - overfit
status: observation
created: 2026-09-14
updated: 2026-09-14
slug: obs-2026-09-14-def-14-two-ended-reading-mary-had-appendix-worked-example-dr
relevance: high
observed_at: 2026-09-14T15:37:01.036Z
source_context: Updating Def 14 wording and documenting the ν_B walk as appendix worked example
---

# ⭐ Observation: Def 14 two-ended reading + "Mary had" appendix worked example drafted

Def 14 wording updated in docs/kalvin-algebra.md (uncommitted): after the licensing table, "The restriction reads on both ends of the move: forward, the departed node carries underfit content or the arriving witness adopts overfit content; reverse, the consumed nodes carry the underfit or the arriving head lands in the overfit. An empty underfit therefore bars nothing — an overfit relationship is worked by adoption, on the arrival clause alone." Matches the ν_B-walk probe's generalized S2 guard exactly (probe finding: the literal A-centric reading blocked all forward moves in pure overfit). Also added "# Appendix — Worked example: 'Mary had'" after §14 (uncommitted): pure-overfit worked example in §9's format — memory mhall:[m,h,all]/all:[o]/all:[a,l,l]/o:[m]/m:[m], A0=mh:[m,h], C(A0,B)=mh:[m,h,all] Overfit u=∅ o={a,l} Δ0=2; stuck-entry analysis (why targeting alone fails, why the one-party ask is false since o:[m] connects the parties); Step 1 overfit-slot walk with hop table all:[all]→all:[o]→all:[m] (both forward on head occurrence), neither end refines (anchor already a ν_A node, departed end already at goal's witness resolution), head-ward write m:[m,all] Overfit depth 2; Step 2 adopt m⇉[m,all]→[m,all,h] done Canon; goal never rewritten; Ĥ=2/3 γ≈0.63 overfit sealed in [all]; closing mirror of §9's "important point". CONTENT trace matches dev/algebra/overfit-anchor-walk.py output exactly (7/7 PASS, exit 0; sibling probe also green). CONTEXT.md Licence entry updated with the two-ended reading. All uncommitted pending user go-ahead.

*Relevance: high*
*Context: Updating Def 14 wording and documenting the ν_B walk as appendix worked example*
*Tags: kalvin-algebra def14 appendix worked-example overfit*

---
*Observed: 2026-09-14T15:37:01.036Z*
