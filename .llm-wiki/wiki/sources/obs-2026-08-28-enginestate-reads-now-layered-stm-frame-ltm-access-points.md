---
type: source
title: "Observation: EngineState reads now layered STM→Frame→LTM access points"
tags:
  - dialogue
  - engine
  - memory
  - refactor
status: observation
created: 2026-08-28
updated: 2026-08-28
slug: obs-2026-08-28-enginestate-reads-now-layered-stm-frame-ltm-access-points
relevance: high
observed_at: 2026-08-28T15:16:14.154Z
source_context: Refactoring EngineState read semantics to layered access points
---

# ⭐ Observation: EngineState reads now layered STM→Frame→LTM access points

EngineState reads refactored into continuous layered access points (STM → Frame → LTM): find scans STM newest-first then Frame/LTM bucket[-1]; new find_sig(sig) = STM sig entries + Frame bucket + LTM bucket, replacing the public find_bucket (now removed); findCanons and where span all layers; canon_nodes now includes Frame (was LTM→STM); ltm_nodes (unused) replaced by layered sig_nodes. is_grounded/is_framed/_is_denoted/_is_groundable deliberately stay single-store (grounding/framing are store facts). Callers updated: reentry.py, expand_fit.py, pivot_fill.py (find_bucket→find_sig). Verified via direct assertions — dialogue tests remain blocked by the pre-existing is_s1 ImportError in kalvin/significance.

*Relevance: high*
*Context: Refactoring EngineState read semantics to layered access points*
*Tags: dialogue engine memory refactor*

---
*Observed: 2026-08-28T15:16:14.154Z*
