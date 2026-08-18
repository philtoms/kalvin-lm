---
type: source
title: "Observation: No `python -m kscript` entrypoint exists"
tags:
  - kscript
  - cli
  - entrypoint
status: observation
created: 2026-08-11
updated: 2026-08-11
slug: obs-2026-08-11-no-python-m-kscript-entrypoint-exists
relevance: high
observed_at: 2026-08-11T14:57:51.661Z
---

# ⭐ Observation: No `python -m kscript` entrypoint exists

KScript has no standalone compiler CLI. There is no `src/kscript/` module and no `ks/__main__.py`. KScript is compiled in-process via `ks.compiler.compile_source(source: str)`, as used by `scripts/kalvin_test.py` and by the dialogue and training harnesses. Any doc claiming `python -m kscript script.ks -out output.json` is wrong.

*Relevance: high*
*Tags: kscript cli entrypoint*

---
*Observed: 2026-08-11T14:57:51.661Z*
