---
type: source
title: "Observation: Goal taken from top of list; hop runs multiple derivations down it"
tags:
  - docs
  - kalvin
  - selection
  - hop
  - loop
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-goal-taken-from-top-of-list-hop-runs-multiple-derivations-do
relevance: high
observed_at: 2026-09-16T12:05:56.860Z
source_context: Correcting goal-taking and hop's multiple derivations in kalvin-algebra.md
---

# ⭐ Observation: Goal taken from top of list; hop runs multiple derivations down it

Two refinements to the strategy model per Phil: (1) Def 22's "The goal B is chosen from the list... engine's queue strategy" was wrong — selection provides an ORDERED list; the goal is TAKEN FROM THE TOP, deterministically. Consequence: a hop runs MULTIPLE derivations — goals worked down the list in order, each with its own scope, a derivation ending without done yielding the next candidate; hop ends at done, list exhaustion, or a bound. Def 21 rewritten accordingly ("the strategy unit of one queued kline"), loop paragraph updated ("The first four phases repeat within a hop"), Bounds gained a fourth limit (the number of goals a hop takes from its list). (2) Re-entry phrasing fixed to his exact relationship: re-entry changes A, and A RESELECTS CANDIDATES for B — fresh list from the new A, top taken again (possibly the same kline). "Engine's queue strategy" phrase fully removed. CONTEXT Candidates/Hop/Reentry entries aligned. Probes 7/7. Note for future code work: hop loop = for goal in selection(A): scope, derive, break on done — versus current Derivation single-goal assumption.

*Relevance: high*
*Context: Correcting goal-taking and hop's multiple derivations in kalvin-algebra.md*
*Tags: docs kalvin selection hop loop*

---
*Observed: 2026-09-16T12:05:56.860Z*
