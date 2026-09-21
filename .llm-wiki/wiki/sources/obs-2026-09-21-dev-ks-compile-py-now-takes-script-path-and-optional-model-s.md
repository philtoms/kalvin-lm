---
type: source
title: "Observation: dev/ks/compile.py now takes script path and optional --model state"
tags:
  - kscript
  - dev-tools
  - compile
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-dev-ks-compile-py-now-takes-script-path-and-optional-model-s
relevance: medium
observed_at: 2026-09-21T13:36:38.418Z
source_context: Adding CLI args to dev/ks/compile.py
---

# 🔍 Observation: dev/ks/compile.py now takes script path and optional --model state

Rewrote dev/ks/compile.py with argparse: positional script path plus optional --model <state.json>. When --model is given, the state's word_bits are read from the JSON and passed to compile_source(word_bits=..., known_words=list(word_bits)), mirroring harness.load_engine (src/dialogue/harness.py:517-520): compiles must continue the loaded state's word→bit mapping and bind its words. Labels print the same with/without --model; the difference is in the uint64 encodings. Must be run via `uv run python dev/ks/compile.py ...` (pythonpath=src in pyproject).

*Relevance: medium*
*Context: Adding CLI args to dev/ks/compile.py*
*Tags: kscript dev-tools compile*

---
*Observed: 2026-09-21T13:36:38.418Z*
