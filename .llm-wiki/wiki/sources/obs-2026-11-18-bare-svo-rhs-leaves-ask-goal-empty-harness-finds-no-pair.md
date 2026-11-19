---
type: source
title: "Observation: Bare `== SVO` RHS leaves ask goal empty — harness finds no pair"
tags:
  - kscript
  - harness
  - compiler
  - goal
status: observation
created: 2026-11-18
updated: 2026-11-18
slug: obs-2026-11-18-bare-svo-rhs-leaves-ask-goal-empty-harness-finds-no-pair
relevance: high
observed_at: 2026-11-18T17:01:23.994Z
source_context: Investigating why the dialogue harness reports no findable goal for mhall-ag.ks
---

# ⭐ Observation: Bare `== SVO` RHS leaves ask goal empty — harness finds no pair

dev/dialogue/harness.py rejects mhall-ag.ks (`(Mary had a little lamb) == SVO`, no `=>` body) with "no `==` ask with a findable goal kline". Root cause: ks/ast_emitter.py only stamps dbg.goal on a COUNTERSIGNS ask when the `==` RHS is an OperatorScope (`SVO =>` with indented body) — the goal string is read from the first OperatorScope item in the ask's scope.items. A bare signature RHS parses as a plain Signature item, so goal=''. The harness's _ask_goal_pairs skips any ask with an empty d.goal before it ever label-matches. Notably mhall-ag.ks DOES compile a goal-shaped kline (CANONICALISES 'SVO':[S,V,O] — the compound canon, non-ask), so a compiler-side fix (attach goal for bare RHS) would make the harness pair fine; the authored form (mhall.ks) instead yields SVO:[Subject,Verb,Object] from the `=>` block canon. Fix is either script-side (author a `SVO =>` body) or compiler-side (stamp goal on bare-RHS countersigns).

*Relevance: high*
*Context: Investigating why the dialogue harness reports no findable goal for mhall-ag.ks*
*Tags: kscript harness compiler goal*

---
*Observed: 2026-11-18T17:01:23.994Z*
