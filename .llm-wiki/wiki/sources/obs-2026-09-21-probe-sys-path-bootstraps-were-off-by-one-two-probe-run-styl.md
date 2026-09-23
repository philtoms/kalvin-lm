---
type: source
title: "Observation: Probe sys.path bootstraps were off-by-one; two probe run styles"
tags:
  - probes
  - tooling
  - gotcha
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-probe-sys-path-bootstraps-were-off-by-one-two-probe-run-styl
relevance: medium
observed_at: 2026-09-21T15:16:42.157Z
source_context: Refactoring src/dialogue into main code lines
---

# 🔍 Observation: Probe sys.path bootstraps were off-by-one; two probe run styles

Two quirks found during the move: (1) The historical probe bootstrap `_SYS_SRC = Path(__file__).resolve().parent.parent / "src"` was off by one — from dev/dialogue/*.py it computes <root>/dev/src (nonexistent), so it silently no-opped; probes actually ran on PYTHONPATH=src all along. Correct root from a probe is Path(__file__).resolve().parents[2]. probe_rationalise.py, probe_scope0_groups.py, and trace_full.py now carry correct root+src bootstraps. (2) Probes are run two ways: PYTHONPATH=src:. python -m style, or direct .venv/bin/python dev/dialogue/probe_x.py (sys.path[0]=dev/dialogue + CWD-relative "src" insert — must be run from repo root).

*Relevance: medium*
*Context: Refactoring src/dialogue into main code lines*
*Tags: probes tooling gotcha*

---
*Observed: 2026-09-21T15:16:42.157Z*
