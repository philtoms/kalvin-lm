---
type: source
title: "Observation: Probe stalls now wait for S3 match; no-break for-run; all scripts agree"
tags:
  - reentry
  - probe
  - exchange
  - stalls
  - rationalisation
status: observation
created: 2026-11-02
updated: 2026-11-02
slug: obs-2026-11-02-probe-stalls-now-wait-for-s3-match-no-break-for-run-all-scri
relevance: high
observed_at: 2026-11-02T17:28:23.622Z
source_context: Fixing self-witness stalls in the reentry probe
---

# ⭐ Observation: Probe stalls now wait for S3 match; no-break for-run; all scripts agree

dev/probes/probe_simple_reentry.py final state of the self-witness fix: stalls no longer frame themselves. The stall branch pops the frame and records it in `waiting: dict[value, list[parent_seq]]`; a stall whose value already has a waiter under a DIFFERENT parent (frame seq via itertools.count — ids recycle, seqs don't) mints the joint frame (C1 amended). The for-run no longer breaks on an inner ground (C3 amended): every candidate pairs once per run, mirroring the reference's independent batches; the entry pops at its first grounding candidate; only the claim (depth 1) stops the run. Frames carry _Frame(sig, cands, depth, memo, seq, parent). ProbeRun gained .exchanges. Measured agreements vs reference Cogitator: mhall MHALL:[SVO] (161 pairs, 2 stalls, 5 exchanges, 42 grounds); wdmh step2 MHALL:[SVO] (170/3/5); wdmh step4 WDMH:[MHALL] (131/3/4); two-scenarios both. Runtime ~0.6s for all scripts; no iteration caps hit; suite 162/15/16 untouched. The no-break change roughly doubled pairing counts (mhall 84→161, wdmh 24→131) — the price of batch-pairing all candidates, as the reference pays in rows (148/168/120).

*Relevance: high*
*Context: Fixing self-witness stalls in the reentry probe*
*Tags: reentry probe exchange stalls rationalisation*

---
*Observed: 2026-11-02T17:28:23.622Z*
