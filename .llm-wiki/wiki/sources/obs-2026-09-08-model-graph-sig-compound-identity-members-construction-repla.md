---
type: source
title: "Observation: Model graph: sig→compound→identity-members construction replaces node fan-out"
tags:
  - dialogue
  - harness
  - graph
  - compound
  - canon
  - identity
  - mts
status: observation
created: 2026-09-08
updated: 2026-09-08
slug: obs-2026-09-08-model-graph-sig-compound-identity-members-construction-repla
relevance: medium
observed_at: 2026-09-08T12:59:16.698Z
source_context: Compound-based model graph rendering
---

# 🔍 Observation: Model graph: sig→compound→identity-members construction replaces node fan-out

Reworked the harness model-graph per user design: multi-node klines no longer fan out to individual nodes. A multi-node kline SIG:[n1..nk] renders as one edge SIG --class--> compound, where compound = signifier.signature_of(nodes) (OR-reduced value, label = concatenated node labels: SVO -> SubjectVerbObject, DH -> didhave). Identity-held members hang beneath the compound as UNLABELLED edges (user: "the edge is not labelled for identities"), each showing the identity kline's layer glyphs; members without held identities are omitted. Identity klines therefore no longer render as standalone self-loop blocks when their value is a compound member; compound-less identities (Mark:[Mark] in the curriculum) keep an `id [X]` fallback block. Single-node klines switched to the same arrow style (`denotation  ---> Det  W`); unknowns render `---> ∅`. Compound vertices are derived, not held: distinct ids (c{value:x} in DOT/mermaid), dashed white boxes/`:::compound` class. Key fix: compound names must use the composed KNode.label, never the labels dict — for canons the compound value equals the signature's value, so the script label (SVO) would shadow the concatenated label (SubjectVerbObject). The S2 residual becomes visible: WDMH underfit renders WDMH ---> DHwhat (compound ≠ sig).

*Relevance: medium*
*Context: Compound-based model graph rendering*
*Tags: dialogue harness graph compound canon identity mts*

---
*Observed: 2026-09-08T12:59:16.698Z*
