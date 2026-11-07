---
type: source
title: "Observation: rationalisation split: algebra/domain/report modules; Event replaces Row"
tags:
  - rationalisation
  - refactor
  - modules
  - event
status: observation
created: 2026-11-03
updated: 2026-11-03
slug: obs-2026-11-03-rationalisation-split-algebra-domain-report-modules-event-re
relevance: high
observed_at: 2026-11-03T09:55:25.740Z
source_context: Separating the reference model from the cogitator
---

# ⭐ Observation: rationalisation split: algebra/domain/report modules; Event replaces Row

Refactored src/rationalisation for reference/cogitator separation (user-directed): new module layout — algebra.py (Kline, can, meet, residue, contained, mu, Decomposition/decompose, Band/band, significance, entropy, BETA, effort — the shared value vocabulary), domain.py (Domain + witness index), report.py (Event, End, Proposal, Rationalisation, proposals — the run's report vocabulary), cogitator.py (the default Cogitator), model.py (ReferenceCogitator + its private internals). Row and the (row,kind) event tuples reduced to a single structure Event(kind, depth, entry: KNode, candidate, F, U, O, frame) — kinds: pair|frame|stall|exchange|close|wait|known; "rows" as table-row concept expunged: Rationalisation is (stores, ends, events, claim) — no rows/path/goal_row. The reference keeps its internal pairing record as private _Pairing (public trace is Event); cogitator's _Frame renamed _Entry (parent-seq stall matching unchanged); harness TrsStep renamed Step. Harness table rewritten over events (columns depth/E/C/F/U/O/frame/event; plain exploration pairings elided — minting pairings, stalls, exchanges, resolutions, depth-1 carry the story); the reference's trace()/path selection is no longer called (trace.py dormant, imports fixed to _Pairing). verify.py: Row→_Pairing, Records gained goal_row; certificate test's RecordingRunner now swaps in ReferenceCogitator (it instruments the reference); T2 digest hashes len(result.events). Verified: suite 162/15/16; probe comparisons all agree (cogitator events 287/309/238 vs reference 240/269/196 on mhall/wdmh — reference also emits wait/close events); harness event tables render for both engines; no TrsStep/Row references remain.

*Relevance: high*
*Context: Separating the reference model from the cogitator*
*Tags: rationalisation refactor modules event*

---
*Observed: 2026-11-03T09:55:25.740Z*
