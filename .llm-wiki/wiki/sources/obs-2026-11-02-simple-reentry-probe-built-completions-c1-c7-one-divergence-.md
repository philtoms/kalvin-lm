---
type: source
title: "Observation: Simple reentry probe built; completions C1–C7; one divergence isolated"
tags:
  - rationalisation
  - probe
  - reentry
  - pseudocode
status: observation
created: 2026-11-02
updated: 2026-11-02
slug: obs-2026-11-02-simple-reentry-probe-built-completions-c1-c7-one-divergence-
relevance: high
observed_at: 2026-11-02T15:48:25.969Z
source_context: Probing the simple reentry rules against the reference rationalisation
---

# ⭐ Observation: Simple reentry probe built; completions C1–C7; one divergence isolated

dev/probes/probe_simple_reentry.py implements the user's minimal reentry pseudocode (entries stack of (sig, candidates), decompose F/U/O, frame on F==E and on stall, push unresolved sides, ground when fit+both sides resolved) with completions C1–C7 documented in the module docstring: stall pops (C1); a for-run that pushes nothing new pops — the stack re-check is the poor man's frame cascade (C2); pop-after-ground removes the grounding frame by identity (C3); ground mints a frame (C4); per-parent push memo for termination (C5); opening entry pairs the goal alone and grounds the claim Kline(ask,{goal}) (C6); MAX_DEPTH=10 (C7). Driver runs fit_gate="framed" (as written) vs "free" (drop framed(F)) against the reference Runner on mhall.ks, wdmh.ks, and any script path passed as argv. Measured: mhall agrees under both gates (84/76 pairings vs reference 148 rows); wdmh steps 1–3 agree; step 4 (WDMH ask) diverges only under the framed gate. One global pushed-once guard was tried first and was WRONG: when F=0 the ask's underfit IS its own signature — the reference's "twins explore their own" means regions must be explorable per-parent. Probe stalls=5/grounds=40 in mhall; no exchange machinery anywhere. Test suite untouched: 162 passed, 15 skipped, 16 xfailed.

*Relevance: high*
*Context: Probing the simple reentry rules against the reference rationalisation*
*Tags: rationalisation probe reentry pseudocode*

---
*Observed: 2026-11-02T15:48:25.969Z*
