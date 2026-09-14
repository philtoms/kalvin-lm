---
type: source
title: "Model-state graph design: klines deduped with layer glyphs, strongest-layer styling"
status: insight
category: design
created: 2026-09-08
updated: 2026-09-08
slug: model-state-graph-layer-rendering
---

# Model-state graph design: klines deduped with layer glyphs, strongest-layer styling

Design pattern for graphing a tiered memory model (EngineState: ltm/frame/work_list/refused) where the same kline can live in several tiers at once. Do NOT cluster the graph by tier — membership overlaps and would duplicate nodes. Instead: dedupe klines by (signature, nodes), accumulate tier glyphs (e.g. "LF") on the kline, and union per-value tier sets over both roles (head and node). For renderers with single-valued attributes (DOT fill, mermaid classDef), style by "strongest" tier under a fixed priority order (L > F > W > R). Semantic details that earn their place: mark ask signatures with "?" (signifier.is_ask), star nodes that head their own klines (the cross-reference edge), and list values that are referenced but never headed in a separate trailing section — that residue is exactly what a trainer diagnoses. Unknown klines ({S: []}) correctly render as isolated vertices: no edges, nothing held. Related: the `dialogue-dev` skill, [[CONTEXT.md]] glossary.

*Category: design*

---
*Captured: 2026-09-08*

## Related

_Add links to related pages._
