---
type: source
title: "Observation: README.md updated to match actual source tree"
tags:
  - docs
  - readme
  - project-structure
status: observation
created: 2026-08-11
updated: 2026-08-11
slug: obs-2026-08-11-readme-md-updated-to-match-actual-source-tree
relevance: high
observed_at: 2026-08-11T14:57:51.658Z
source_context: Updating README.md after initializing the LLM wiki
---

# ⭐ Observation: README.md updated to match actual source tree

README.md's "Project Structure" tree and CLI sections were stale. Updated to match the real `src/` layout: added `src/dialogue/` (engine.py, harness.py, runner.py, actors.py, rationalise.py, synthesize.py, decoder.py, pivot_fill.py, similar_fit.py, engine_state.py) and `src/training/auto_tune/`; corrected the `kalvin/`, `ks/`, `training/harness/`, `training/trainer/`, and `training/supervisors/` file listings. Replaced the bogus `python -m kscript` CLI section (no such module exists — KScript is a library, entry point is `ks.compiler.compile_source`) with the three real runnables: `python -m dialogue.harness` (lean harness, takes a .ks path, flags -v and -s {similar_fit,expand}), `python -m training.harness` (multi-agent server), and `python -m training.auto_tune`. Also added .llm-wiki/ to the Documentation table.

_Relevance: high_
_Context: Updating README.md after initializing the LLM wiki_
_Tags: docs readme project-structure_

---

_Observed: 2026-08-11T14:57:51.658Z_
