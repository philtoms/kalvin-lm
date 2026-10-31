---
type: source
title: "Observation: Dialogue harnesses consolidated — TRS-only module"
tags:
  - harness
  - trs
  - refactor
  - consolidation
status: observation
created: 2026-10-31
updated: 2026-10-31
slug: obs-2026-10-31-dialogue-harnesses-consolidated-trs-only-module
relevance: high
observed_at: 2026-10-31T09:04:54.366Z
source_context: Consolidating the engine and TRS dialogue harnesses into one module
---

# ⭐ Observation: Dialogue harnesses consolidated — TRS-only module

Consolidated dev/dialogue/harness.py and dev/dialogue/trs_harness.py into a single TRS-only module at dev/dialogue/harness.py (user ruling: TRS is the winning solution carried forward; the engine-path distinction is retired). Deleted: dev/dialogue/trs_harness.py, dev/dialogue/structural.py (engine -e escalation supervisor), tests/test_ratification.py (engine _grade_proposal tests). Engine machinery removed from harness.py: Harness/Turn/StepResult, make/load_rationaliser, model-graph renderers, interactive/queue supervisors, persistence, scaffolding modes, --model/-s/-e/-p/-g flags, markdown-curriculum loop. New presentation rule: per-step trace table + stores/halts print by default; the end-of-run state sections (── domain held ──, ── pending proposals ──, discharged count) only under -v, which also keeps hex signatures. Renamed present_trs → present; _sig_to_label trimmed of the engine Memory/state path. Updated imports in tests/test_trs_harness.py and 6 probes (dump_mhall, probe_cogitator_trace, trace_ask, probe_mhall_walk, prototype_mhall_table, probe_ask_step2_trace). Stale pointers fixed: README tree, CONTEXT.md (Relational Tokens now points at the ask==goal pairing; Scaffolding entry points at domain priming; Semantic Evidence entry deleted — its machinery lived only in the deleted structural.py), rationaliser.py docstring factory sentence dropped. .pi/skills/dialogue-dev/SKILL.md intentionally left stale for later per user instruction. Suite: 162 passed, 16 xfailed.

*Relevance: high*
*Context: Consolidating the engine and TRS dialogue harnesses into one module*
*Tags: harness trs refactor consolidation*

---
*Observed: 2026-10-31T09:04:54.366Z*
