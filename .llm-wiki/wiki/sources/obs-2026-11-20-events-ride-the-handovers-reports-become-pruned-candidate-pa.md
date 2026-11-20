---
type: source
title: "Observation: Events ride the handovers — reports become pruned candidate paths"
tags:
  - cogitator
  - events
  - framing
  - path-energy
  - report
status: observation
created: 2026-11-20
updated: 2026-11-20
slug: obs-2026-11-20-events-ride-the-handovers-reports-become-pruned-candidate-pa
relevance: high
observed_at: 2026-11-20T10:15:09.273Z
source_context: Events into entry — pruned candidate-path reports
---

# ⭐ Observation: Events ride the handovers — reports become pruned candidate paths

Events moved from _Run to Entry (commit ec4bbe99, on top of frame isolation 5567bb1d) — the story rides the same forward passes as the frame, with one refinement: a departure hands its story forward only when it ALIGNED (S2 framed, claimed, or the exchange's S3 — Entry.aligned flag); the frame hands forward unconditionally. Mechanics: Entry.carry(e) is the push handover (frame+story); _pop(parent, entry) always copies the frame, copies the story only if entry.aligned (fresh lists both ways — no aliasing); _Run.events removed; _Run.finish now takes the finishing Entry and the report is Rationalisation(entry.events) — the claiming path's story. Effect on traces: wdmh step 2 went 28→14 rows, step 1 29→16 — rival lineages' rows (the U-side WDH exploration, Object/what mints, the DH stalls, the MH rival candidate) drop at their unaligned departures; what survives is the O-side path through the exchange (exchanged → WDH framed at the witness-parent's S2) plus the winning rotations. Stalled events never appear in reports (probes never align) — stalls report only through the sink, consistent with the doc. Purpose (Phil's): plot a pruned graph traversal where rival candidates neither interfere with nor support each other — the report is the traversal the claim rode; this is the step towards path energy. KNOWN GAP for path energy: alignments use frame elements minted in dropped lineages (MH frames over Mary/had mints that live in the dropped rival's story; WDH's Object/what likewise) — the story does not track frame provenance; closing that needs a support graph (which alignment minted which element, recursively) — likely the next step. Also noted: the committed stall branch no longer has the else (a served-but-unretryable stall now files itself) — was in the tree at commit time, tests green.

*Relevance: high*
*Context: Events into entry — pruned candidate-path reports*
*Tags: cogitator events framing path-energy report*

---
*Observed: 2026-11-20T10:15:09.273Z*
