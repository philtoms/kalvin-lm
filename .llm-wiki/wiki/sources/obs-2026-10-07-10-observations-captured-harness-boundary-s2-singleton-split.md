---
type: source
title: "Observation: 10 observations captured: harness boundary, S2 singleton split, tail recursion"
tags:
  - trs
  - spec
  - observations
  - r2
  - singleton
  - rationalise3
  - design-brief
status: observation
created: 2026-10-07
updated: 2026-10-07
slug: obs-2026-10-07-10-observations-captured-harness-boundary-s2-singleton-split
relevance: high
observed_at: 2026-10-07T16:37:08.302Z
source_context: Capturing the user's 10 observations formally
---

# ⭐ Observation: 10 observations captured: harness boundary, S2 singleton split, tail recursion

TEN OBSERVATIONS CAPTURED FORMALLY in docs/trs-significance.md (the rationalise3 design brief): (1-4) new §7 subsection 'The harness boundary' — well-formed script holds ask + goal + all scaffolding to trace ask→goal; the point of rationalisation is the transformation (ask's content becomes the goal's form); the model neither knows nor cares — presentation and goal-reach are harness concerns; the model's own ask==goal configuration IS the candidate set (Defs 9+10) — the scripted pairing is at most one of them. (5) §5 preamble: the rules transform the candidate — S1 arrival is a full canon (Def 5). (6) tree section: candidates processed in turn, early termination on first S1; set exhausted without S1 ends at R4. (7) R2 REWRITTEN — products are now SINGLETON cross-pairings ⟨F,{F}⟩, ⟨U,{O}⟩, ⟨O,{U}⟩; the witness lookups (N_F/N_O/N_U) are DELETED from R2; bands fixed by construction: F-product arrives S1 (canonical by Def 5), cross products arrive S3 (U,O disjoint per Def 12); READING FLAGGED to user: [X] read as the singleton adjacency {X} per session-long sig:[adj] notation — deterministic bands match the unconditional parentheticals; awaiting confirmation. (8) no change — R3 already reenters on ⟨σ,N′⟩ = C:[S]. (9) §5: rationalisation is in principle tail-recursive, returning at S1 or S4. (10) R2 completion condition: the split tail-recurses to S1 only when the re-entrant branches return at S1 — at least two S1 returns, one from each cross product. Also noticed: user amended R2's header to 'project, re-enter' (parallel to R3's 'substitute, re-enter'). Diagram updated with product bands. OPEN: user floated 'candidate invariant is a misleading term?' for Def 10 — rename proposal pending ('Candidate pairing'). 159 tests passing.

*Relevance: high*
*Context: Capturing the user's 10 observations formally*
*Tags: trs spec observations r2 singleton rationalise3 design-brief*

---
*Observed: 2026-10-07T16:37:08.302Z*
