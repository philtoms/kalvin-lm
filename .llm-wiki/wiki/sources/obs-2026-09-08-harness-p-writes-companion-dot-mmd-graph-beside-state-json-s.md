---
type: source
title: "Observation: Harness -p writes companion .dot/.mmd graph beside state JSON; save-path bug fixed"
tags:
  - dialogue
  - harness
  - persistence
  - vscode
  - graphviz
  - mermaid
status: observation
created: 2026-09-08
updated: 2026-09-08
slug: obs-2026-09-08-harness-p-writes-companion-dot-mmd-graph-beside-state-json-s
relevance: low
observed_at: 2026-09-08T10:01:50.864Z
source_context: VSCode preview support for persisted dialogue state
---

# 📝 Observation: Harness -p writes companion .dot/.mmd graph beside state JSON; save-path bug fixed

Harness `-p` now also writes a companion graph file next to the saved state JSON: `mhall.json` → `mhall.dot` (default) or `.mmd` (when `--graph mermaid` is given), rendered from end-of-run state via the shared _GRAPH_RENDERERS map. Purpose: VSCode preview — DOT renders via the Graphviz Interactive Preview extension, .mmd via Mermaid preview extensions; VSCode has no native graph rendering so a companion file (not a .json association) is the mechanism. Also fixed a pre-existing bug in the same block: `-p PATH` loaded from PATH but always saved to `data/dialogue/{stem}.json`, contradicting its own help text; save now targets state_path.

*Relevance: low*
*Context: VSCode preview support for persisted dialogue state*
*Tags: dialogue harness persistence vscode graphviz mermaid*

---
*Observed: 2026-09-08T10:01:50.864Z*
