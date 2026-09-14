---
type: source
title: "Observation: connotateY reverse edges bridge WDMH to MHALL"
tags:
  - dialogue
  - cogitator
  - engine
  - feat
status: observation
created: 2026-09-04
updated: 2026-09-04
slug: obs-2026-09-04-connotatey-reverse-edges-bridge-wdmh-to-mhall
relevance: high
observed_at: 2026-09-04T09:22:30.325Z
source_context: "Dialogue-dev: reverse-edge design for connotateY"
---

# ⭐ Observation: connotateY reverse edges bridge WDMH to MHALL

Added reverse connotation edges to Cogitator.connotateY (commit eff2aab on dialogue branch): alongside forward find_sig(signature) edges, traversal now follows klines whose signature shares a word bit with cur (signifies) or that contain cur as a node. This bridges word-level gap nodes (what/did/have — only identity resolutions in memory) to compound-signature klines. With mhall.json memory loaded, wdmh-underfit.ks step 1 now proposes WDMH:[ALL, DH] S3 13 and WDMH:[SVO] S3 2 — the WDMH=>MHALL bridge the user asked for. Decoded: 'what did Mary have' -> Mary (fit) + a-little-lamb (ALL) + did-have (DH). mhall.ks canonical run unchanged. Prototype in dev/dialogue/probe_reverse_edges.py.

*Relevance: high*
*Context: Dialogue-dev: reverse-edge design for connotateY*
*Tags: dialogue cogitator engine feat*

---
*Observed: 2026-09-04T09:22:30.325Z*
