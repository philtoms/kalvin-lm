---
type: source
title: "Observation: build_state.py dev tool added"
tags:
  - kscript
  - dialogue
  - devtools
status: observation
created: 2026-08-27
updated: 2026-08-27
slug: obs-2026-08-27-build-state-py-dev-tool-added
relevance: medium
observed_at: 2026-08-27T09:05:17.992Z
---

# 🔍 Observation: build_state.py dev tool added

New dev script `dev/ks/build_state.py` compiles a .ks source via `compile_source` (BPETokenizer + NLPSignifier, dev=True), grounds every entry kline into a fresh EngineState's ltm, and saves via `EngineState.save`. Usage: `PYTHONPATH=src .venv/bin/python dev/ks/build_state.py <src.ks> [out.json]`; default output is `data/dialogue/<stem>.json`. Verified on mhall.ks: 22 entries, 13 ltm signatures, stm/frame empty; round-trips through `EngineState.load`.

_Relevance: medium_
_Tags: kscript dialogue devtools_

---

_Observed: 2026-08-27T09:05:17.992Z_
