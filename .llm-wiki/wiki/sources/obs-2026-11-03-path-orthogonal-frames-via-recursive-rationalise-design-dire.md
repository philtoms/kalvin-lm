---
type: source
title: "Observation: Path-orthogonal frames via recursive rationalise — design direction and forks"
tags:
  - trs
  - cogitator
  - design
  - orthogonality
  - frames
  - recursion
status: observation
created: 2026-11-03
updated: 2026-11-03
slug: obs-2026-11-03-path-orthogonal-frames-via-recursive-rationalise-design-dire
relevance: high
observed_at: 2026-11-03T14:08:05.343Z
source_context: "Design: path-orthogonal frames vs stack-based reentry in the Cogitator"
---

# ⭐ Observation: Path-orthogonal frames via recursive rationalise — design direction and forks

Empirical: in new wdmh step 4, HALL's re-surface closes ([283]-[297]) — ALL EIGHT of its candidates close, each leaning on frames minted by its OWN children's subtrees: `had`/`ALL` frames were minted at depth 3/4 by the had/ALL reentries that HALL's earlier candidates pushed (HV pushed ALL, ALLQ pushed had), then every later candidate's U side settles on them. This is exactly the SETTLE violation the ReferenceCogitator's rule forbids ("covered by an ancestor's or joint frame — never the row's own subtree's"): the flat stack has no path structure so it cannot express the rule. User's design direction (agreed sound): recurse into rationalise instead of pushing to the shared stack — per-path frames passed through recursion (vertical inheritance only), rows in an append-only heap with parent-index paths. Sharpening: sequential DFS needs NO stack copying — frames as a small tuple arg (≤10 deep), parent idx as int; the integer-index snapshot becomes load-bearing only for (a) PARKING stalled paths for exchange resumption, (b) later parallel candidate reentry, (c) the future pruner (walk parent ints — trace.py's job without objects). Frame visibility classes: own-descent mints (settle own sides+children), ancestor mints (inherited down), joint/exchange frames (global, sanctioned horizontal). Cost honesty: orthogonal paths re-explore shared subregions; memo can't cross paths without horizontal borrowing — MORE work than the flat loop, same order as the reference with smaller constants. OPEN FORKS put to user: stall resumption (park-and-resume vs let-dead — DH/Det/Mod exchanges differ), first-wins vs explore-all after an entry grounds, parent bar depth (full ancestor path free in recursion), confirm own-subtree mint rule.

*Relevance: high*
*Context: Design: path-orthogonal frames vs stack-based reentry in the Cogitator*
*Tags: trs cogitator design orthogonality frames recursion*

---
*Observed: 2026-11-03T14:08:05.343Z*
