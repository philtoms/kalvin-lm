---
type: source
title: "Observation: User updates absorbed: next-candidate opener, R3 reentry on opener"
tags:
  - trs
  - redesign
  - rationalise2
  - opener
  - r3
  - reentry
  - user-updates
status: observation
created: 2026-10-07
updated: 2026-10-07
slug: obs-2026-10-07-user-updates-absorbed-next-candidate-opener-r3-reentry-on-op
relevance: high
observed_at: 2026-10-07T14:03:07.284Z
source_context: Absorbing the user's model.py updates to rationalise2
---

# ⭐ Observation: User updates absorbed: next-candidate opener, R3 reentry on opener

USER'S model.py UPDATES absorbed (4 changes): (1) remaining int→KNode (enrichment containment check now `s.sig & cand.sig == s.sig` directly); (2) OPENER IS THE NEXT CANDIDATE — resolve(cand, cand): each candidate is its own chain's opener; the resolving chain (possibly a later candidate after fallback) stores ITS opener; the enriched push and R3 reentry PROPAGATE the opener rather than resetting it; (3) R3 REENTRY ON OPENER — S3 branch: resolve(Kline(cand.sig, n_prime), opener, depth+1): the head stays the candidate's signature, only the adjacency is substituted (simplifies R3 algebra — no canonicalisation of the reentrant head); (4) depth increments on enriched/R3 pushes (bound by depth, not a rounds counter). NEW TRACES: GOAL (mhall) 8 rounds, 3 stores, 0 ends: Object:[Object], Verb|Object:[Verb], Subject|Verb|Object:[Subject] — the chains now store their OPENING pairings (single-node witnesses), NOT the enriched klines (previously [Verb,Object]/[Subject,Verb,Object]) — NOTE this supersedes the earlier 'Verb|Object:[Verb, Object]' expectation; ask-side: ASK (MHALL) 77 rounds, 5 stores, 1 end — a clean bottom-up sentence build: lamb:[lamb], lamb|little:[little], lamb|a|little:[a], had|lamb|a|little:[had], Mary|had|lamb|a|little:[Mary]; one depth-bound end at lamb|a|little:[Det] (the Det connotation cannot resolve). Updated the pinned test to opener semantics and refreshed rationalise2's stale docstring (still said 'FIRST candidate'/'opener is itself'). 159 tests passing.

*Relevance: high*
*Context: Absorbing the user's model.py updates to rationalise2*
*Tags: trs redesign rationalise2 opener r3 reentry user-updates*

---
*Observed: 2026-10-07T14:03:07.284Z*
