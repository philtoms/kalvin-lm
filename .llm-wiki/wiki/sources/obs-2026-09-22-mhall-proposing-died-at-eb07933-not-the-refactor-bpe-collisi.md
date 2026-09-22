---
type: source
title: "Observation: mhall proposing died at eb07933, not the refactor — BPE-collision leak closed"
tags:
  - mhall
  - bisect
  - regression
  - def22
  - coverage
status: observation
created: 2026-09-22
updated: 2026-09-22
slug: obs-2026-09-22-mhall-proposing-died-at-eb07933-not-the-refactor-bpe-collisi
relevance: critical
observed_at: 2026-09-22T13:47:17.084Z
source_context: Diagnosing the silent canonical mhall run
---

# 🔴 Observation: mhall proposing died at eb07933, not the refactor — BPE-collision leak closed

Bisect result for "canonical mhall run does not propose any more": the recent refactor (f94d363 dissolve src/dialogue → b0d179e fast-path events) is behaviour-neutral for mhall — zero asks at every commit. Last proposing commit: 39fdbd5 (proposes SVO:[SVO] S1). First silent: eb07933 "Def 1 honored — measure/units/same_content on KSignifier". Root: Def 22 coverage in hop.candidate_goals changed from raw `int(n) & kc` to `signifier.signifies(n, kc)` (masked word-content overlap). The old test counted BPE token-id bit collisions as coverage, admitting 7 of 10 spurious goal candidates for the queued SVO canon scaffold; the lone SVO:[SVO] proposal rode that leak. Per docs/kalvin-algebra.md Def 8 + the realisation table (v ∧ w = (a & b) & MASK), eb07933 is spec-faithful — the old proposal was an artifact. wdmh.ks still proposes WDMH:[ALL, had, Mary] S1 255 at HEAD; 90/90 tests green. Bisect method: git worktrees in /tmp/kalvin-bisect with tokenizer data copied in (data/tokenizer is gitignored); probes kept at dev/dialogue/probe_mhall_stall.py and dev/dialogue/probe_mhall_ask_attends.py.

*Relevance: critical*
*Context: Diagnosing the silent canonical mhall run*
*Tags: mhall bisect regression def22 coverage*

---
*Observed: 2026-09-22T13:47:17.084Z*
