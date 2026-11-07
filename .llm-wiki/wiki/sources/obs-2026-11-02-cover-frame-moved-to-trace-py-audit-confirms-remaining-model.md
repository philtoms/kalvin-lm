---
type: source
title: "Observation: _cover_frame moved to trace.py; audit confirms remaining model surface is semantic"
tags:
  - refactor
  - trace
  - audit
  - cover-frame
  - run-records
  - boundary
status: observation
created: 2026-11-02
updated: 2026-11-02
slug: obs-2026-11-02-cover-frame-moved-to-trace-py-audit-confirms-remaining-model
relevance: high
observed_at: 2026-11-02T12:18:38.479Z
source_context: Pulling trace-related methods and state out of model per user directive
---

# ⭐ Observation: _cover_frame moved to trace.py; audit confirms remaining model surface is semantic

TRACE-ONLY SURFACE AUDIT completed after the user pushed for more ("_cover_frame is only called from within trace so it does not belong in model"). MOVED: _cover_frame (smallest chain-excluding frame covering a region — pure narration analysis; sole caller was the trace's lean()) → module-level _cover_frame(cog, x, chain) in dev/dialogue/trace.py. AUDITED AND KEPT, with the ruling logic: _chain/_onpath/_covered/_pre_settled/frame_re are load-bearing in the search (bar checks, settle rules — model-internal callers verified programmatically: every remaining Cogitator method has ≥1 in-model call site); rframes/frame_src are RUN RECORDS — written during search at semantic moments (grounding goal-side O, resolution, known), read only by the trace afterward — the same standing as the events notes: the model records, the harness interprets; reconstructing resolution order in the trace from close-notes was considered and rejected (couples narration to resolution sequencing, saves 3 append lines). trace.py now touches exactly one model-private method (cog._chain ×2 — shared semantic helper). Final: model.py 659 lines (search + records only), trace.py 167 (narration), harness 526 (Runner + render). mhall/wdmh/-v byte-identical; suite 162/15/16. Cumulative uncommitted diff: model −165, harness +2, trace.py new.

*Relevance: high*
*Context: Pulling trace-related methods and state out of model per user directive*
*Tags: refactor trace audit cover-frame run-records boundary*

---
*Observed: 2026-11-02T12:18:38.479Z*
