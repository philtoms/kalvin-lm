---
type: source
title: "Observation: Clock removed, notes in: events list separates model from reporting"
tags:
  - clock
  - events
  - separation
  - diagnostics
  - rendering
  - refactor
status: observation
created: 2026-11-02
updated: 2026-11-02
slug: obs-2026-11-02-clock-removed-notes-in-events-list-separates-model-from-repo
relevance: high
observed_at: 2026-11-02T11:57:09.455Z
source_context: Investigating whether _tick/clock is functional and separating rationalisation from diagnostic reporting
---

# ⭐ Observation: Clock removed, notes in: events list separates model from reporting

CLOCK INVESTIGATION — verdict and separation, superseding the user's `return 1` experiment found uncommitted in the tree (they had neutered `_tick` to empirically test what breaks). FINDINGS: (1) the SEARCH is entirely blind to the clock — with ticks returning a constant, mhall/wdmh still ground, all frames/halts identical; the only model-internal read was `row.waited is None` (a once-flag, not the value); (2) the TABLE consumes it — `_render_table` sorted (birth, waited, closed) into the pair/wait/close interleaving; with constant ticks the temporal shape collapses (all BP lines group at the end in creation order — the nested cascade `BP Query` right after its trigger row is lost), which the user's experiment demonstrated. SEPARATION LANDED: the model NOTES events — `self.events: list[(row, "pair"|"wait"|"close")]` appended at the six sites (pair in _pair/_known/stall-row, wait in _after, close in _resolve/_exchange) — call order ≡ the old tick order (ticks were strictly increasing), and `Rationalisation.events` carries them out; Row drops birth/waited/closed/instant entirely; `_tick`/`self.clock` deleted. The harness renders from the notes: per-row first pair/wait/close note index, sort by (index, kind-rank); "born closed" (the old `instant`) is inferred as close-note == pair-note+1 adjacency. mhall, wdmh, and mhall -v outputs BYTE-IDENTICAL to the real-tick baseline; suite 162/15/16; both scripts ground. Doc §2.2 updated (events = reporting only, never read). Net: -31/+31 across model+harness; the model's Row is now purely structural (resolved/bp/known/exchange/supports) and all temporality lives in the notes the harness interprets.

*Relevance: high*
*Context: Investigating whether _tick/clock is functional and separating rationalisation from diagnostic reporting*
*Tags: clock events separation diagnostics rendering refactor*

---
*Observed: 2026-11-02T11:57:09.455Z*
