---
type: source
title: "Observation: _trace extracted to dev/dialogue/trace.py — model searches, harness narrates"
tags:
  - refactor
  - trace
  - separation
  - model
  - harness
  - diagnostics
status: observation
created: 2026-11-02
updated: 2026-11-02
slug: obs-2026-11-02-trace-extracted-to-dev-dialogue-trace-py-model-searches-harn
relevance: high
observed_at: 2026-11-02T12:14:27.802Z
source_context: Exploring table-tracing responsibility moved out of model into harness via trace(cog)
---

# ⭐ Observation: _trace extracted to dev/dialogue/trace.py — model searches, harness narrates

TRACE RESPONSIBILITY MOVED OUT OF THE MODEL (exploration landed, verified, pending user commit ruling). _trace() — the ~150-line optimal-path builder (keep/tabled/via_frame/feeds walk, the repair pass, the supports/lean analysis) — was the model's ONLY narrator: its sole consumer was harness.py:413 (`result.path`); no test reads .path or .supports; dormant verify/existence don't either. New shape: (1) model.py −158 lines — Cogitator.rationalise returns the run without narrating it (comment: "the path is the harness's reading of the run"); Row.supports stays as a field but commented as the harness trace's annotation; rframes/frame_src/frame_re/stalls/goal_row stay on the Cogitator as run records the trace READS (written during search, narrated after); (2) NEW dev/dialogue/trace.py — `def trace(cog: Cogitator) -> list[Row]`, the body moved verbatim (self→cog), module docstring carries the path semantics (one row per reentry, orthogonal paths, generation order); (3) harness Runner._step: `result.path = trace(self.cogitator)` right after rationalise — the harness asks for the reading, fills Rationalisation.path itself. Verification: mhall, wdmh, mhall -v outputs byte-identical; suite 162/15/16. Model now 671 lines, trace 154, harness 526. Boundary: trace reads cog internals (frame_re, _chain via cog._chain...) — accepted as in-project diagnostics; the doc's §2.2 needs no change (it never described _trace). EXTRACTION LESSON: two failed attempts from stale string offsets after intermediate replaces corrupted the seam (unmatched paren / orphaned fragment) — always recompute slice offsets after edits, or extract from `git show HEAD:` (which worked).

*Relevance: high*
*Context: Exploring table-tracing responsibility moved out of model into harness via trace(cog)*
*Tags: refactor trace separation model harness diagnostics*

---
*Observed: 2026-11-02T12:14:27.802Z*
