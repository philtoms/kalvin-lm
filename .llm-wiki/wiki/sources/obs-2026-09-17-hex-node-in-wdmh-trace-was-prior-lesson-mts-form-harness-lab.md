---
type: source
title: "Observation: Hex node in wdmh trace was prior-lesson MTS form; harness label maps now seed from state"
tags:
  - dialogue-dev
  - harness
  - labels
  - mts
  - wdmh
  - presentation
status: observation
created: 2026-09-17
updated: 2026-09-17
slug: obs-2026-09-17-hex-node-in-wdmh-trace-was-prior-lesson-mts-form-harness-lab
relevance: high
observed_at: 2026-09-17T17:48:24.863Z
source_context: "dialogue-dev: undecodable signature in wdmh.ks trace"
---

# ⭐ Observation: Hex node in wdmh trace was prior-lesson MTS form; harness label maps now seed from state

dialogue-dev: "undecodable signature" in wdmh.ks trace diagnosed. The hex node 0x1c00001efd was NOT an engine bug: it is ALL's MTS single-token form (word bits a|little|lamb OR synthetic token id 7933='acher'), persisted in the mhall.json prior (frame holds ALL(0x1c00001efd):[a,little,lamb], SVO(0xe000007ffe):[Subject,Verb,Object]). Engine path was algebra-legal throughout: hop 1's bridge walk (what→Object←ALL, commit 3351301's licence) descends and contracts through the held ALL canon, writes the composed correspondence what⇉[ALL]; hop 2 consumes it ([what,had,Mary]→[ALL,had,Mary]), content equals MHALL → done BY CALCULATION (j1=1.0, our Def 16 change); the proposal answers at compound granularity — same content as the earlier [a,little,lamb,had,Mary] answer, node sequences need not be equal. The actual defect was harness presentation: label maps were built per-script (_sig_to_label from the current source only), so prior-lesson MTS values rendered as raw hex in traces, frame, and work_list. Fix: _state_labels(state) recovers {value:label} from held klines (labels survive persistence on KSig/KNode values; engine-copied plain ints lose them, hence the value-keyed map), merged via setdefault into all four _sig_to_label sites (present, structural supervisor, interactive supervisor, graph) — script labels stay authoritative. After: proposes WDMH:[ALL, had, Mary] S1 255; MHALL:[MHALL], hadVerb:[hadVerb] all decode. Tests 69/69. Probe: dev/dialogue/probe_hexnode.py. Note: word-form token halves are synthetic ('acher', undecodable 32766) — no collision risk in the atom space (word bits are the atoms; token half masked).

*Relevance: high*
*Context: dialogue-dev: undecodable signature in wdmh.ks trace*
*Tags: dialogue-dev harness labels mts wdmh presentation*

---
*Observed: 2026-09-17T17:48:24.863Z*
