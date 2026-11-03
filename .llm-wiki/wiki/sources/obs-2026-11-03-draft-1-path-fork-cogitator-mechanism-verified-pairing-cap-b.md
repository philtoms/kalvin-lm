---
type: source
title: "Observation: Draft 1: path-fork Cogitator — mechanism verified, pairing cap blown"
tags:
  - trs
  - cogitator
  - recursion
  - fork
  - draft
  - orthogonality
status: observation
created: 2026-11-03
updated: 2026-11-03
slug: obs-2026-11-03-draft-1-path-fork-cogitator-mechanism-verified-pairing-cap-b
relevance: high
observed_at: 2026-11-03T14:49:15.332Z
source_context: "Cogitator draft 1: recursion into new state, heap + integer paths, path frames"
---

# ⭐ Observation: Draft 1: path-fork Cogitator — mechanism verified, pairing cap blown

Draft 1 of the path-fork Cogitator landed (per user ruling: mechanism over outcomes — 'I do not require it to deliver the end goal'): explore() recurses into NEW STATE — each candidate pairing's descent runs with its own fork of frames (frames + own F==E mint travels only with that candidate's descent), a shared append-only heap rows list (one _Row per reentry: sig/depth/parent/psig/pcsig), and a lightweight path = tuple of heap indices copied on descent. Stack semantics held: LIFO within a path, parent identity via heap indices, memo per parent across its candidates. Settle rules: empty | framed on path | framed joint | the recursion's return True (child grounded). Exchanges mint self.joint — the ONLY horizontal channel; stalls PARK the path (_Parked(parent, path, frames)) — resume is deferred. Forward-only: no parent-loop re-entry, no shared loop state. VERIFIED: 2223 heap rows all chain to root (max path 9 ≤ MAX_DEPTH); mhall step 2 got 2731 closes out of 4097 pairings — deep paths ground locally; Det/Mod parked once each then exchanged (joint frames minted). COST FINDING: strict orthogonality with zero cross-path reuse blows the 4096-pairing cap — mhall step 2 alone doesn't finish, no claim, 508+ exhausted halts (every non-grounding entry halts — no re-surface rescue). Next complexity reaction needed: reuse across forks (per-signature memo with frame-state keying?) or parked-path resume, else the cap is the only thing keeping the run finite. Tests still 162/15/16 (they don't pin step-2 grounding).

*Relevance: high*
*Context: Cogitator draft 1: recursion into new state, heap + integer paths, path frames*
*Tags: trs cogitator recursion fork draft orthogonality*

---
*Observed: 2026-11-03T14:49:15.332Z*
