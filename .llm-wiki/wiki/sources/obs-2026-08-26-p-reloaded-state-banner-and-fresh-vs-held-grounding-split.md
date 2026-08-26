---
type: source
title: "Observation: -p reloaded-state banner and fresh-vs-held grounding split"
tags:
  - harness
  - persistence
  - presentation
status: observation
created: 2026-08-26
updated: 2026-08-26
slug: obs-2026-08-26-p-reloaded-state-banner-and-fresh-vs-held-grounding-split
relevance: medium
observed_at: 2026-08-26T16:22:30.692Z
source_context: Lean harness -p persistence presentation
---

# 🔍 Observation: -p reloaded-state banner and fresh-vs-held grounding split

Added to src/dialogue/harness.py (commit e123aa8): (1) a banner `── running on reloaded state: N grounded klines from PATH ──` when -p loads an existing state file; (2) the summary's grounded list now splits into freshly-grounded-this-run entries, followed by a `(reloaded, held before this run)` section listing pre-existing LTM klines — implemented by snapshotting `{(sig, nodes)}` from state.ltm before run() and passing `pre_grounded` through present()/_render_summary/_render_grounded. Non-persisted runs render a single undivided list. Presenter-only change; non-judging contract intact. Fresh-run traces unchanged; on a reloaded run everything is pre-grounded so no asks/grounds appear — that silence is expected, not a regression.

*Relevance: medium*
*Context: Lean harness -p persistence presentation*
*Tags: harness persistence presentation*

---
*Observed: 2026-08-26T16:22:30.692Z*
