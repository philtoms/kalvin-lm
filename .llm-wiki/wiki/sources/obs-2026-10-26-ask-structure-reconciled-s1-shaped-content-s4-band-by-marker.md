---
type: source
title: "Observation: Ask structure reconciled: S1-shaped content, S4 band by marker; empty asks only in src/trs"
tags:
  - trs
  - asks
  - s4
  - s1
  - marker
  - kalvin-algebra
  - dialogue
status: observation
created: 2026-10-26
updated: 2026-10-26
slug: obs-2026-10-26-ask-structure-reconciled-s1-shaped-content-s4-band-by-marker
relevance: high
observed_at: 2026-10-26T11:21:32.560Z
source_context: Reconciling scripted-ask structure (S1 identity) vs dialogue-work S4 ASK descriptions
---

# ⭐ Observation: Ask structure reconciled: S1-shaped content, S4 band by marker; empty asks only in src/trs

Question raised: scripted asks look S1-identity-shaped yet dialogue docs say the ask is S4. Resolution: the ask has ONE structure — `s|ASK:[canon nodes]` (decoder.py ASK branch → KLine(mark_ask(sig), canon nodes); kalvin-algebra §13 "the identity shape, marked"). Content layer (marker masked, as every content read does): S1 — canon-exact for multi-node asks, identity for atom asks. Band layer: sig_level reads bit 63 from the value before any content check → S4 "the question, whatever its nodes". Empirically verified on mhall.ks: sig_level marker-aware = S4, marker masked = S1, is_canon(unmarked) = True. Also: the trace's fed byte for an ask is γ toward its goal (harness graded()), NOT the S4 stamp — "the structural band (the ask's S4 shape) is recomputed from structure by the rationaliser — never the fed byte" — which is the likely source of S1 sightings. _OP_TO_SIG["ASK"]=SIG_S4 is Target Significance (answer key), not a measurement. Structural (empty-form) S4 asks: generated ONLY in src/trs (Kline(c.sig, frozenset()) at no-fit halt, R4 ends, unrecognised-node acquire asks ⟨n,∅⟩ — Def 11 case 1); NEVER in src/kalvin or dev/dialogue — no empty-node KLine is constructed anywhere there (is_unknown: "No compiled ask takes the shape"); the empty form appears only as the refused unknown statement. The two truth docs deliberately diverge on the ask's construction: trs-significance.md = empty kline; kalvin-algebra.md = marked canon-rider.

*Relevance: high*
*Context: Reconciling scripted-ask structure (S1 identity) vs dialogue-work S4 ASK descriptions*
*Tags: trs asks s4 s1 marker kalvin-algebra dialogue*

---
*Observed: 2026-10-26T11:21:32.560Z*
