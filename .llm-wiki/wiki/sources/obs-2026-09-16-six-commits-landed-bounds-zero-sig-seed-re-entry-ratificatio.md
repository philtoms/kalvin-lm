---
type: source
title: "Observation: Six commits landed: bounds, zero-sig, seed, re-entry, ratification"
tags:
  - commit
  - milestone
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-six-commits-landed-bounds-zero-sig-seed-re-entry-ratificatio
relevance: high
observed_at: 2026-09-16T18:08:01.349Z
source_context: Committing the milestone batch
---

# ⭐ Observation: Six commits landed: bounds, zero-sig, seed, re-entry, ratification

Committed the WDMH→MHALL milestone batch as six commits on `dialogue`: ef2d90c kalvin: witness-driven canonicalisation survey; the walk's state bound enforced (the Hop.run hang — subset enumeration replaced by canon-multiset matching, MAX_WALK_STATES enforced); ebbe995 ks: lowercase multi-char canonicalises sigs are words, not deferred compounds (the zero-sig bug — sig 0 minted for `h(ad)`→'had', the atom vanishing from every contracted state); c3952e4 ks: cross-script word binding — compiles seed known_words (labels + value continuity, the MHALL drift gone); 3423800 dialogue: the ask's hop chain — _propose runs re-entry (run_hops finally wired; done-guard keyed on original nodes); f217244 dialogue: ratification — the goal grades proposals; the S1 stamp grounds (γ-of-content-vs-target grading; is_groundable gate removed from the S1 fast route); cc296aa dev+wiki (derivation monitor + observations). Hunk-split engine.py/harness.py/CONTEXT.md across the seed and ratification commits via awk + git apply --cached. 69/69 green, tree clean. Note: data/scripts/* stay uncommitted by project convention (data/ gitignored) — the working wdmh-underfit.ks on disk includes the user's own edit (a leading `(what Object)` / `W = O` scaffold group feeding what:[Object] as its own step).

*Relevance: high*
*Context: Committing the milestone batch*
*Tags: commit milestone*

---
*Observed: 2026-09-16T18:08:01.349Z*
