---
type: source
title: "Observation: Greedy pivot proposal was premature-alignment artifact; lone-gap fills only"
tags:
  - dialogue
  - engine
  - pivot-alignment
  - proposals
status: observation
created: 2026-08-21
updated: 2026-08-21
slug: obs-2026-08-21-greedy-pivot-proposal-was-premature-alignment-artifact-lone-
relevance: high
observed_at: 2026-08-21T07:51:57.689Z
source_context: Investigating greedy WDMH pivot proposal
---

# ⭐ Observation: Greedy pivot proposal was premature-alignment artifact; lone-gap fills only

The greedy pivot proposal WDMH:[Mary,had,a,little] (semantic nonsense — 'what did Mary have' answered with a truncated word form) was an artifact of premature alignment, not an inevitable step. Cause: in the early pass did/have had no resolution paths, leaving 3 gaps zipped one-each with leftovers [had,a,little,lamb] — arbitrary pairing presented as a proposal. Fix in src/dialogue/expand_fit.py _pivot_proposals: only a LONE gap takes a fill (the whole grouped residual); two or more open gaps means K cannot yet say which node answers which gap, so the pivot arm abstains (the pivot remains a reentry vehicle via _fills_through). Once DH:[did,have] grounds, the same alignment resolves: did+have grouped through DH→had, Mary shared, what lone gap takes [a,little,lamb]. Also corrected an earlier misreading: DH:[did,have] DOES count as a canon for is_canon (looser than sig(nodes)==signature), so grouped sub-canon resolution does fire in the working state. Both scripts now emit only ALL:[a,little,lamb] S1 255, WDMH:[had,Mary,a,little,lamb] S2 244, DH:[did,have] S1 255 — no declined-then-recover sequence needed. Rule: work is assigned only when its assignment is determined (shared, path-resolved, grouped, single-gap residual), never by permutation guess.

*Relevance: high*
*Context: Investigating greedy WDMH pivot proposal*
*Tags: dialogue engine pivot-alignment proposals*

---
*Observed: 2026-08-21T07:51:57.689Z*
