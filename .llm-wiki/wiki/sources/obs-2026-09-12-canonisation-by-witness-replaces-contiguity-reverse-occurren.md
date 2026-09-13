---
type: source
title: "Observation: Canonisation by witness replaces contiguity: reverse occurrence is multiset-wise"
tags:
  - kalvin-algebra
  - canonisation
  - replace-rule
  - design-decision
status: observation
created: 2026-09-12
updated: 2026-09-12
slug: obs-2026-09-12-canonisation-by-witness-replaces-contiguity-reverse-occurren
relevance: high
observed_at: 2026-09-12T12:58:43.167Z
source_context: "Redesigning replace occurrence semantics: contiguity → multiset-wise canonisation"
---

# ⭐ Observation: Canonisation by witness replaces contiguity: reverse occurrence is multiset-wise

Phil rejected both the ordering-rule and gather designs: the model should instead canonise its nodes into unordered configurations, by witness. Implemented across kalvin-algebra.md, kalvin-for-agents.md, CONTEXT.md. Formal reading: Def 13's reverse replace now matches occurrences multiset-wise (an unordered configuration of ν_A's nodes) — contiguity deleted everywhere; no rule reads sequence order, arrangement is witness structure pure. Canonisation = the reverse replace engaged position-free, with a bidirectional witnessing lock as cognitive reading: the configuration's nodes witness the compound (each covered by the candidate head, Def 8 — recognition from below) and the compound counter-witnesses the nodes (held canon n:ν_K whose witness is exactly that configuration — composition claim from above). Exactness implies coverage for canons (witness nodes are atom-subsets of head), so the licence is formally just ν_K's multiset occurrence. The survey is memory-bounded: held witnesses propose configurations; nothing unwitnessed contracts. Worked example now: canonise {d,h}⇉dh out of [w,d,m,h] position-free (discontinuous verb phrase invisible to the licence), then shed, then walk+traverse. CONTEXT.md gained a Canonisation glossary entry with _Avoid_: gather, reordering. Benefits: symmetric with forward occurrence (both sides of a correspondence selectable position-free); dissolves the ordering-rule question entirely; avoids factorial state-space blowup; matches the engine's observed grouped-resolve behaviour (behaviour-notes: did+have grouped-resolve through DH→had). Open puzzle flagged to Phil: §11's granularity-monotonicity (expand strictly increases D̄, contract strictly decreases) appears sign-inconsistent with D̄'s definition (0 for content held as itself) and the worked example's D̄=0 at the fully-bare end state — under every (ν_A,M)-computable convention I can construct, expand decreases nesting depth.

*Relevance: high*
*Context: Redesigning replace occurrence semantics: contiguity → multiset-wise canonisation*
*Tags: kalvin-algebra canonisation replace-rule design-decision*

---
*Observed: 2026-09-12T12:58:43.167Z*
