---
type: source
title: "Observation: ExpandFit docstrings rewritten: three-arm module doc, session narrative removed"
tags:
  - dialogue
  - docs
  - expand-fit
status: observation
created: 2026-08-21
updated: 2026-08-21
slug: obs-2026-08-21-expandfit-docstrings-rewritten-three-arm-module-doc-session-
relevance: medium
observed_at: 2026-08-21T13:54:38.360Z
source_context: Docstring drift pass over expand_fit strategy
---

# 🔍 Observation: ExpandFit docstrings rewritten: three-arm module doc, session narrative removed

expand_fit.py docstring pass: module docstring rewritten to cover all three proposal arms in emission order (pivots → fills → reentry) rather than only crossover fills; removed the DMHAL/MHALL session narrative from propose's gapless-canon comment; retired 'framing' vocabulary from BUDGET comment ('cogitation frames' → 'at most this many proposals'); fixed _grade's '0.0' to 'no credit' (returns int sig byte). Inline rule comments in _pivot_proposals/_fills were already current and kept.

*Relevance: medium*
*Context: Docstring drift pass over expand_fit strategy*
*Tags: dialogue docs expand-fit*

---
*Observed: 2026-08-21T13:54:38.360Z*
