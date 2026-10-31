---
type: source
title: "Observation: TRS reentry stack — open-format frames, grounded merges, S4 empty-kline asks"
tags:
  - trs
  - rationalise
  - frame-stack
  - fresh-start
  - open-format
  - grounded-merge
  - s4-ask
status: observation
created: 2026-10-21
updated: 2026-10-21
slug: obs-2026-10-21-trs-reentry-stack-open-format-frames-grounded-merges-s4-empt
relevance: high
observed_at: 2026-10-21T15:54:29.235Z
source_context: "TRS rationalise fresh start: open-format frame-stack algorithm"
---

# ⭐ Observation: TRS reentry stack — open-format frames, grounded merges, S4 empty-kline asks

Fresh-start rationalise implemented in src/trs/model.py (uncommitted, on top of user's wip 84977ef): the open-format frame-stack algorithm per the user's spec. New pieces: candidate_list(d, entry) — each overlapping memory kline individually formed ⟨can(entry.adj), nb.adj⟩, identities filtered, sorted ascending (effort, sig); Frame dataclass (sig, C, F, U, O, n_f, n_u, n_o, depth); next_candidate(c, entry, depth) — n_f/n_o from the candidate's adj, n_u from the ENTRY's adj (k.adj in the spec = entry); grounded(d, s) = ∃k ∈ d : can(k.adj) == s; rationalise pops the FIRST candidate only (rest discarded), then a LIFO work_stack: F=∅ → S4 ask ⟨c.sig, ∅⟩; grounded u/o region merges into n_f, else reentry child pushed (children complete above the parent's requeue at index 0); exits: both-clear → admit, C == can(n_f) → admit (fires even with u pending — the worked example's ending), else stalled → End proposing ⟨sig, n_f ∪ n_o⟩ (S2-underfit → connect, S3-overfit → acquire); _reenter guards (empty adj → R4, seen → silent skip, depth+1≥10 → bound, no candidates → End) and marks seen on ALL paths (bug found+fixed: failed reentries must mark seen or parents retry and emit duplicate Ends). Semantics: grounded() is cross-frame through the shared domain — a child's admit makes the parent's region-check true next pass; identity mints arise naturally from the C-exit on trivial frames (entry ⟨A,{A}⟩, candidate ⟨A,{A,B}⟩ grounds ⟨A,{A}⟩). Domain.overlap iterates a SET not insertion order — equal-effort ties are hash-arbitrary, so tests pin only tie-robust facts. mhall: step 1 grounds the goal (stores Subject|Object:[S,O] + SVO:[S,V,O] canon); step 2 pops the ALL candidate (E 2/5), fit {a,little,lamb}, u=Mary|had reenters → its candidate ⟨Mary|had,{Subject}⟩ has F=∅ → S4 ask E1, base stalls S2 proposing ⟨MHALL,{a,little,lamb}⟩ connect — exactly the worked example's ask shape. r3_substitute/consolidated-candidates machinery deleted from the runner (r3_substitute kept exported). proposals(): stalled-at-S2 joins no-candidates → connect. 169 tests pass (14 rewritten, 2 deleted, honesty property extended with S4/stalled reasons).

*Relevance: high*
*Context: TRS rationalise fresh start: open-format frame-stack algorithm*
*Tags: trs rationalise frame-stack fresh-start open-format grounded-merge s4-ask*

---
*Observed: 2026-10-21T15:54:29.235Z*
